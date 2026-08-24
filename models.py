"""
models.py — a basic class-conditional DCGAN, sized dynamically for any target patch_size.

Generator: latent vector + class embedding -> project to a small 4x4 feature map ->
repeated ConvTranspose2d blocks (each doubles spatial size, halves channels) until we
reach or exceed patch_size -> a final resize + refinement conv locks in the exact
requested size (so patch_size doesn't need to be a power of two).

Discriminator: mirrors this with strided Conv2d blocks, conditioned on class by
concatenating a constant per-class channel to the image. AdaptiveAvgPool2d at the end
means it also works for any patch_size without manual size bookkeeping. Spectral norm
is applied to the conv/linear layers by default -- it's a one-line addition that
meaningfully helps stability once you push resolution up (e.g. towards 512x512), which
plain DCGAN struggles with.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def num_upsample_blocks(target_size: int, base_size: int = 4) -> int:
    """How many x2 upsampling blocks are needed so base_size * 2**k >= target_size."""
    if target_size <= base_size:
        return 0
    return math.ceil(math.log2(target_size / base_size))


class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=4, embed_dim=64, patch_size=128,
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
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
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


class Discriminator(nn.Module):
    def __init__(self, num_classes=4, patch_size=128, base_channels=64, img_channels=3,
                 use_spectral_norm=True, min_spatial=4, max_channels=512):
        super().__init__()
        self.patch_size = patch_size

        def sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        # one learned scalar per class, broadcast as an extra constant channel
        self.label_embed = nn.Embedding(num_classes, 1)

        in_ch = img_channels + 1
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
        self.pool = nn.AdaptiveAvgPool2d(min_spatial)
        self.classifier = sn(nn.Linear(ch * min_spatial * min_spatial, 1))

    def forward(self, img, labels):
        y = self.label_embed(labels).view(-1, 1, 1, 1).expand(-1, 1, img.shape[2], img.shape[3])
        x = torch.cat([img, y], dim=1)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    # quick shape smoke test, run with: python models.py
    for size in (64, 100, 128, 256, 512):
        g = Generator(patch_size=size)
        d = Discriminator(patch_size=size)
        z = torch.randn(2, g.project[0].in_features - g.label_embed.embedding_dim)
        labels = torch.randint(0, 4, (2,))
        fake = g(z, labels)
        assert fake.shape == (2, 3, size, size), fake.shape
        score = d(fake, labels)
        assert score.shape == (2, 1), score.shape
        print(f"patch_size={size:4d}: generator out {tuple(fake.shape)}, "
              f"discriminator out {tuple(score.shape)}, G blocks={g.n_blocks} OK")
    print("All shape checks passed.")
