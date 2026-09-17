"""
models.py — a basic class-conditional DCGAN, sized dynamically for any target patch_size.

Generator: latent vector + class embedding + attention-score embedding -> project to
a small 4x4 feature map -> repeated Upsample(nearest)+Conv2d blocks (each doubles
spatial size, halves channels) until we reach or exceed patch_size -> a final resize +
refinement conv locks in the exact requested size (so patch_size doesn't need to be a
power of two). We use Upsample+Conv2d rather than ConvTranspose2d specifically to
avoid the checkerboard artifacts transposed convolutions are known to produce (uneven
kernel overlap) -- this was showing up clearly in earlier sample grids.

Discriminator: mirrors this with strided Conv2d blocks, conditioned on class AND
attention score by concatenating one constant channel per condition to the image. A
minibatch-stddev layer is inserted before the final classifier: it appends one extra
channel containing the batch's feature stddev, so the discriminator can directly
notice when a whole batch of generator outputs looks suspiciously uniform -- a direct
countermeasure against mode collapse. AdaptiveAvgPool2d at the end means the whole
thing also works for any patch_size without manual size bookkeeping. Spectral norm is
applied to the conv/linear layers by default -- it's a one-line addition that
meaningfully helps stability once you push resolution up (e.g. towards 512x512),
which plain DCGAN struggles with.

ATTENTION-SCORE CONDITIONING: both G and D now take a per-patch attention score
(from your AB-MIL model, see attention_manifest.py) alongside the class label, as a
continuous value rather than a discrete one. The class label uses a learned embedding
table (nn.Embedding) since there are only a handful of discrete classes; the
attention score instead goes through a small learned projection (a Linear layer,
`attn_embed` in the Generator / `attn_proj` in the Discriminator) since it's a
continuous scalar -- same conditioning role, different mechanism for a different kind
of input. Expected as a `(batch,)` or `(batch, 1)` float tensor, in whatever range
your manifest stores it in (see attention_manifest.py's docstring for a caveat about
raw AB-MIL attention weights not being comparable across slides with different patch
counts -- worth normalizing upstream if that hasn't already been done).
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


def _as_column(x: torch.Tensor) -> torch.Tensor:
    """Accepts an attention-score tensor shaped (batch,) or (batch, 1) and returns
    (batch, 1), so callers don't have to worry about which one they were handed."""
    return x.view(-1, 1).float()


class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2, embed_dim=64, attn_embed_dim=32,
                 patch_size=128, base_channels=512, img_channels=3, base_size=4,
                 min_channels=32):
        super().__init__()
        self.patch_size = patch_size
        self.base_size = base_size
        self.base_channels = base_channels
        self.n_blocks = num_upsample_blocks(patch_size, base_size)

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.attn_embed = nn.Sequential(
            nn.Linear(1, attn_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim + attn_embed_dim, base_channels * base_size * base_size),
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

    def forward(self, z, labels, attn_scores):
        y = self.label_embed(labels)
        a = self.attn_embed(_as_column(attn_scores))
        x = torch.cat([z, y, a], dim=1)
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
    down to a single scalar and broadcast spatially. Simplified (single-group) version
    of the ProGAN/StyleGAN minibatch-stddev layer -- lets the discriminator directly
    notice when a whole batch of generator outputs is suspiciously uniform, which is
    exactly what mode collapse looks like."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)  # (C, H, W)
        mean_std = std.mean().view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)

class DenseNetPerceptualLoss(nn.Module):
    def __init__(self):
        super(DenseNetPerceptualLoss, self).__init__()
        # Load pre-trained DenseNet201 features
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).features
        
        # DenseNet features structure: 
        # 0:conv0, 1:norm0, 2:relu0, 3:pool0, 4:denseblock1, 5:transition1, 6:denseblock2...
        # We slice up to index 6 to capture early-to-mid level textures (perfect for histology)
        self.slice = nn.Sequential(*list(densenet.children())[:6])
        
        # CRITICAL: Freeze the weights to prevent massive VRAM consumption
        for param in self.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss() # L1 loss produces sharper images than MSE

    def forward(self, input_images, target_images):
        # VGG and DenseNet expect ImageNet normalization. 
        # Assuming your GAN outputs [-1, 1], we map it to [0, 1] then normalize.
        input_images = (input_images + 1) / 2
        target_images = (target_images + 1) / 2
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_images.device)
        
        input_images = (input_images - mean) / std
        target_images = (target_images - mean) / std
        
        # Extract features and calculate distance
        input_features = self.slice(input_images)
        target_features = self.slice(target_images)
        
        return self.criterion(input_features, target_features)


class Discriminator(nn.Module):
    def __init__(self, num_classes=2, patch_size=128, base_channels=64, img_channels=3,
                 use_spectral_norm=True, min_spatial=4, max_channels=512):
        super().__init__()
        self.patch_size = patch_size

        def sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        # one learned scalar per class, broadcast as an extra constant channel
        self.label_embed = nn.Embedding(num_classes, 1)
        # same role for the continuous attention score: a learned affine recalibration
        # (rather than an embedding table, since there's no fixed set of discrete
        # values to look up), also broadcast as one constant channel
        self.attn_proj = nn.Linear(1, 1)

        in_ch = img_channels + 2
        blocks = [
            sn(nn.Conv2d(in_ch, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        cur_size = patch_size // 2
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
        # +1 input channel: the minibatch-stddev layer appends one extra feature map
        self.classifier = sn(nn.Linear((ch + 1) * min_spatial * min_spatial, 1))

    def forward(self, img, labels, attn_scores):
        y = self.label_embed(labels).view(-1, 1, 1, 1).expand(-1, 1, img.shape[2], img.shape[3])
        a = self.attn_proj(_as_column(attn_scores)).view(-1, 1, 1, 1).expand(-1, 1, img.shape[2], img.shape[3])
        x = torch.cat([img, y, a], dim=1)
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
        latent_dim = g.project[0].in_features - g.label_embed.embedding_dim - g.attn_embed[0].out_features
        z = torch.randn(2, latent_dim)
        labels = torch.randint(0, NUM_CLASSES, (2,))

        for attn_scores in (torch.rand(2), torch.rand(2, 1)):  # both (batch,) and (batch,1) accepted
            fake = g(z, labels, attn_scores)
            assert fake.shape == (2, 3, size, size), fake.shape
            score = d(fake, labels, attn_scores)
            assert score.shape == (2, 1), score.shape

        # gradients should reach both new conditioning paths, not just the class embedding
        fake = g(z, labels, torch.rand(2, requires_grad=False))
        score = d(fake, labels, torch.rand(2))
        score.sum().backward()
        assert g.attn_embed[0].weight.grad is not None and g.attn_embed[0].weight.grad.abs().sum() > 0
        assert d.attn_proj.weight.grad is not None and d.attn_proj.weight.grad.abs().sum() > 0

        print(f"patch_size={size:4d}: generator out {tuple(fake.shape)}, "
              f"discriminator out {tuple(score.shape)}, G blocks={g.n_blocks}, "
              f"attn conditioning gradients flow OK")
    print("All shape checks passed.")