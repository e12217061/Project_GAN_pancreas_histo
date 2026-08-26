"""
models.py — pix2pix-style architecture for mask-conditioned image synthesis.

Generator: a U-Net. The one-hot mask (num_classes, H, W) goes in, gets downsampled by
strided convs to a 1x1 bottleneck, then upsampled back out -- with skip connections
concatenating each encoder stage to its matching decoder stage, so fine spatial detail
(exact region boundaries from the mask) survives all the way to the output instead of
being squeezed through the bottleneck. This is the key architectural difference from
the earlier class-conditional Generator: that one only had a *global* vector to work
from (no path for per-pixel conditioning); this one takes the mask as a spatial input
end-to-end. Upsampling uses Upsample+Conv2d rather than ConvTranspose2d, same
checkerboard-avoidance reasoning as before. Requires patch_size to be a power of two
(64/128/256/512) -- U-Net skip connections need exact spatial alignment between encoder
and decoder stages, so this doesn't have the same "any size" flexibility the earlier
class-conditional model had.

Discriminator: a PatchGAN (the classic pix2pix discriminator). It looks at (mask,
image) pairs and outputs a grid of real/fake predictions -- one per local patch of the
image -- rather than a single scalar for the whole image. This is what gives pix2pix
its sharp local texture: it's judging "does this patch look real," many times over,
instead of one global verdict. Spectral norm and the minibatch-stddev layer carry over
from what stabilized training on your small dataset before.

Loss (in train.py): adversarial + a weighted L1 term between generated and real image.
The L1 term is what actually teaches the network to match the mask's specific layout --
adversarial loss alone only pushes toward "looks realistic," not "matches this mask."
"""
import math

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsamples `x` and halves its channel count to out_ch. The caller concatenates
    the corresponding encoder skip connection onto the result afterward -- that
    concatenation happens in UNetGenerator.forward, not in here, since this block has
    no way to know which skip tensor belongs with it."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, num_classes=4, img_channels=3, patch_size=512, noise_channels=0,
                 base_channels=64, max_channels=512, dropout=0.5, n_dropout_layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.noise_channels = noise_channels
        self.patch_size = patch_size
        in_ch = num_classes + noise_channels
        self.n_blocks = int(math.log2(patch_size))
        if 2 ** self.n_blocks != patch_size:
            raise ValueError(f"UNetGenerator needs patch_size to be a power of 2 "
                              f"(64/128/256/512/...), got {patch_size}")

        # encoder channel schedule, e.g. n_blocks=9 (patch_size=512) ->
        # [64,128,256,512,512,512,512,512,512]
        enc_channels = []
        ch = base_channels
        for _ in range(self.n_blocks):
            enc_channels.append(ch)
            ch = min(ch * 2, max_channels)
        self.enc_channels = enc_channels

        self.downs = nn.ModuleList()
        prev_ch = in_ch
        for i, out_ch in enumerate(enc_channels):
            # no norm on the outermost layer (i==0, standard pix2pix convention) or the
            # innermost/bottleneck layer (i==n_blocks-1): InstanceNorm on a 1x1 feature
            # map is degenerate -- mean equals the single value, so it normalizes every
            # activation to exactly zero regardless of input, destroying the bottleneck.
            normalize = i != 0 and i != self.n_blocks - 1
            self.downs.append(DownBlock(prev_ch, out_ch, normalize=normalize))
            prev_ch = out_ch

        # decoder: n_blocks - 1 up-blocks with skip connections, then a final
        # upsample + conv back to img_channels at full resolution (no skip at that
        # last step -- there's no encoder stage at the original input resolution).
        self.ups = nn.ModuleList()
        cur_ch = enc_channels[-1]
        for i in range(self.n_blocks - 1):
            skip_ch = enc_channels[self.n_blocks - 2 - i]
            use_dropout = dropout > 0 and i < n_dropout_layers
            self.ups.append(UpBlock(cur_ch, skip_ch, dropout=dropout if use_dropout else 0.0))
            cur_ch = skip_ch + skip_ch  # out_ch of this up-block + the concatenated skip

        self.final_up = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cur_ch, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, mask_onehot, noise=None):
        x = mask_onehot
        if self.noise_channels > 0:
            if noise is None:
                noise = torch.randn(x.size(0), self.noise_channels, x.size(2), x.size(3),
                                     device=x.device, dtype=x.dtype)
            x = torch.cat([x, noise], dim=1)

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for i, up in enumerate(self.ups):
            x = up(x)
            skip = skips[self.n_blocks - 2 - i]
            x = torch.cat([x, skip], dim=1)

        return self.final_up(x)


class MinibatchStdDev(nn.Module):
    """Appends one extra channel containing the batch's feature-map stddev, averaged
    down to a single scalar and broadcast spatially. Same layer used in the earlier
    class-conditional Discriminator -- carried over since it directly helps against
    mode collapse regardless of what the rest of the architecture looks like."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        mean_std = std.mean().view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


class PatchDiscriminator(nn.Module):
    """The classic pix2pix 70x70 PatchGAN, conditioned on the mask by channel
    concatenation. Fully convolutional -- works at any input resolution, no
    patch_size-dependent bookkeeping needed (unlike the Generator)."""

    def __init__(self, num_classes=4, img_channels=3, base_channels=64,
                 use_spectral_norm=True):
        super().__init__()

        def sn(module):
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        in_ch = num_classes + img_channels
        self.features = nn.Sequential(
            sn(nn.Conv2d(in_ch, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),

            sn(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=False),

            sn(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=False),

            sn(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.minibatch_stddev = MinibatchStdDev()
        self.out_conv = sn(nn.Conv2d(base_channels * 8 + 1, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, mask_onehot, img):
        x = torch.cat([mask_onehot, img], dim=1)
        x = self.features(x)
        x = self.minibatch_stddev(x)
        return self.out_conv(x)  # (batch, 1, h', w') patch-level logits


if __name__ == "__main__":
    # quick shape smoke test, run with: python models.py
    for size in (64, 128, 256, 512):
        g = UNetGenerator(num_classes=4, patch_size=size)
        d = PatchDiscriminator(num_classes=4)
        mask = torch.zeros(2, 4, size, size)
        mask[:, 0] = 1.0  # dummy: everything labeled class 0
        img = torch.randn(2, 3, size, size)
        fake = g(mask)
        assert fake.shape == (2, 3, size, size), fake.shape
        score_real = d(mask, img)
        score_fake = d(mask, fake)
        assert score_real.shape == score_fake.shape
        print(f"patch_size={size:4d}: generator out {tuple(fake.shape)}, "
              f"discriminator out {tuple(score_real.shape)}, G blocks={g.n_blocks} OK")
    print("All shape checks passed.")
