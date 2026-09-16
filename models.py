"""
models.py — a basic class-conditional DCGAN, sized dynamically for any target patch_size.
Now features a Projected GAN architecture for the Discriminator.

Generator: latent vector + class embedding -> project to a small 4x4 feature map ->
repeated Upsample(nearest)+Conv2d blocks (each doubles spatial size, halves channels)
until we reach or exceed patch_size.

Discriminator (Projected): Passes both real and fake images through a frozen, 
pretrained DenseNet201 backbone to extract robust texture features (ImageNet-normalized).
The class label embedding is broadcast and concatenated to these features, which are 
then passed through a small, trainable CNN head with Spectral Norm and Minibatch-StdDev 
to evaluate realness. This prevents mode collapse while massively accelerating learning.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def num_upsample_blocks(target_size: int, base_size: int = 4) -> int:
    """How many x2 upsampling blocks are needed so base_size * 2**k >= target_size."""
    if target_size <= base_size:
        return 0
    return math.ceil(math.log2(target_size / base_size))


class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2, embed_dim=64, patch_size=128,
                 base_channels=512, img_channels=3, base_size=4, min_channels=32):
        super().__init__()
        self.patch_size = patch_size
        self.base_size = base_size
        self.base_channels = base_channels
        self.n_blocks = num_upsample_blocks(patch_size, base_size)

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, base_channels * base_size * base_size),
            nn.BatchNorm1d(base_channels * base_size * base_size),
            nn.ReLU(inplace=True),
        )

        blocks = []
        ch = base_channels
        for _ in range(self.n_blocks):
            out_ch = max(ch // 2, min_channels)
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.upsample = nn.Sequential(*blocks)
        self.final_channels = ch

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        y = self.label_embed(labels)
        x = torch.cat([z, y], dim=1)
        x = self.project(x)
        x = x.view(-1, self.base_channels, self.base_size, self.base_size)
        x = self.upsample(x)
        x = self.to_rgb(x)
        if x.shape[-2:] != (self.patch_size, self.patch_size):
            x = F.interpolate(x, size=(self.patch_size, self.patch_size),
                               mode="bilinear", align_corners=False)
        return x


class MinibatchStdDev(nn.Module):
    """Appends one extra channel containing the batch's feature-map stddev, averaged
    down to a single scalar and broadcast spatially. Countermeasure against mode collapse."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)  # (C, H, W)
        mean_std = std.mean().view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, num_classes=2, patch_size=128, base_channels=64, img_channels=3,
                 use_spectral_norm=True, min_spatial=4, max_channels=512):
        super().__init__()
        self.patch_size = patch_size

        def sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        # --- PRETRAINED BACKBONE (FROZEN) ---
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).features
        
        # Slice up to transition1 (inclusive) to get mid-level texture features
        # Structure: 0:conv0, 1:norm0, 2:relu0, 3:pool0, 4:denseblock1, 5:transition1
        self.backbone = nn.Sequential(*list(densenet.children())[:6])
        
        # CRITICAL: Freeze the backbone weights to save VRAM and prevent them from changing
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # ImageNet normalization tensors for the GAN inputs (which are in [-1, 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # DenseNet201 [:6] slice always outputs exactly 128 channels. 
        backbone_out_ch = 128
        
        # --- TRAINABLE HEAD ---
        # 1 learned scalar per class, broadcast as an extra constant channel over the feature map
        self.label_embed = nn.Embedding(num_classes, 1)

        in_ch = backbone_out_ch + 1  # 128 DenseNet channels + 1 Class Label channel
        
        blocks = [
            sn(nn.Conv2d(in_ch, base_channels * 2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        ch = base_channels * 2
        # We start at patch_size // 8 due to DenseNet's internal pooling (pool0 + transition1)
        cur_size = patch_size // 8 
        while cur_size > min_spatial:
            out_ch = min(ch * 2, max_channels)
            blocks += [
                sn(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch
            cur_size //= 2
            
        self.features = nn.Sequential(*blocks)
        self.minibatch_stddev = MinibatchStdDev()
        self.pool = nn.AdaptiveAvgPool2d(min_spatial)
        # +1 input channel because minibatch_stddev layer appends one extra feature map
        self.classifier = sn(nn.Linear((ch + 1) * min_spatial * min_spatial, 1))

    def train(self, mode=True):
        """Override train to ensure the frozen DenseNet backbone always stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, img, labels):
        # 1. Normalize GAN outputs from [-1, 1] to ImageNet [0, 1] standard
        x = (img + 1.0) / 2.0
        x = (x - self.mean) / self.std
        
        # 2. Extract features through frozen DenseNet
        # (Notice there is NO torch.no_grad() here. We need gradients to pass 
        # *through* the backbone so the Generator can learn from them).
        x = self.backbone(x)
        
        # 3. Inject the Class Embedding as an extra spatial channel
        y = self.label_embed(labels).view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, y], dim=1)
        
        # 4. Trainable Head
        x = self.features(x)
        x = self.minibatch_stddev(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    # quick shape smoke test, run with: python models.py
    NUM_CLASSES = 2
    for size in (64, 100, 128, 256, 512):
        g = Generator(patch_size=size, num_classes=NUM_CLASSES)
        d = Discriminator(patch_size=size, num_classes=NUM_CLASSES)
        z = torch.randn(2, g.project[0].in_features - g.label_embed.embedding_dim)
        labels = torch.randint(0, NUM_CLASSES, (2,))
        fake = g(z, labels)
        assert fake.shape == (2, 3, size, size), fake.shape
        score = d(fake, labels)
        assert score.shape == (2, 1), score.shape
        print(f"patch_size={size:4d}: generator out {tuple(fake.shape)}, "
              f"discriminator out {tuple(score.shape)}, G blocks={g.n_blocks} OK")
    print("All shape checks passed.")