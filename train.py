"""
train.py — train the class-conditional patch GAN end to end.

Minimal example:
    python train.py --data_dir /path/to/data --patch_size 128 --num_classes 4

See `python train.py --help` for all options. Also see README.md for the expected
folder layout and how to adapt dataset.py if your data isn't organized that way.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from dataset import PatchDataset
from models import Generator, Discriminator

# --- ADDED FOR FID ---
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None
# ---------------------


def parse_args():
    p = argparse.ArgumentParser(description="Basic class-conditional patch GAN")
    # data
    p.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing images/ and masks/ subfolders (see README.md)")
    p.add_argument("--images_subdir", type=str, default="images")
    p.add_argument("--masks_subdir", type=str, default="masks")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--background_label", type=int, default=None,
                    help="Mask pixel value to treat as 'no class' / ignore, e.g. 0. "
                         "Leave unset if every pixel belongs to one of the classes.")
    # model / image size
    p.add_argument("--patch_size", type=int, default=None,
                    help="Output resolution. Defaults to auto-detect from the first "
                         "image in your dataset (your images are already cropped to "
                         "this size, e.g. 512). Set explicitly to override. 512 needs "
                         "a lot of GPU memory and is much less stable to train than "
                         "smaller sizes -- consider starting smaller while iterating.")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--g_base_channels", type=int, default=512)
    p.add_argument("--d_base_channels", type=int, default=64)
    p.add_argument("--no_spectral_norm", action="store_true",
                    help="Disable spectral norm in the discriminator (on by default; "
                         "helps stability, especially at larger patch sizes).")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--label_smoothing", type=float, default=0.9,
                    help="'Real' label value fed to the discriminator loss (1.0 = off).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                    help="'cuda', 'mps', or 'cpu'. Auto-detected if not set.")
    # output
    p.add_argument("--output_dir", type=str, default="./gan_outputs")
    p.add_argument("--sample_every", type=int, default=1, help="Save a sample grid every N epochs.")
    p.add_argument("--checkpoint_every", type=int, default=10, help="Save a checkpoint every N epochs.")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt to resume from.")
    p.add_argument("--samples_per_class", type=int, default=4)
    
    # --- ADDED FOR FID ---
    p.add_argument("--eval_fid_every", type=int, default=5, 
                    help="Evaluate FID every N epochs (0 to disable). Requires torchmetrics.")
    p.add_argument("--fid_samples", type=int, default=2048, 
                    help="Max number of samples to use for computing FID (standard is 50k, but 2k-5k is faster for intermediate checks).")
    # ---------------------
    return p.parse_args()


def auto_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_sample_grid(generator, num_classes, samples_per_class, latent_dim, device, path):
    """Generate a few patches per class and save them as one PNG grid (no torchvision dep)."""
    generator.eval()
    with torch.no_grad():
        rows = []
        for c in range(num_classes):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
            imgs = generator(z, labels).cpu()
            imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()  # NHWC uint8
            rows.append(imgs)
    generator.train()

    patch = rows[0].shape[1]
    grid = np.zeros((num_classes * patch, samples_per_class * patch, 3), dtype=np.uint8)
    for r, imgs in enumerate(rows):
        for cidx, img in enumerate(imgs):
            grid[r * patch:(r + 1) * patch, cidx * patch:(cidx + 1) * patch] = img
    Image.fromarray(grid).save(path)


# --- ADDED FOR FID ---
def evaluate_fid(generator, dataloader, fid_metric, latent_dim, device, max_samples=2048):
    """Calculates FID score by generating fake images corresponding to the real dataloader distributions."""
    print(f"  [FID] Computing over ~{max_samples} samples... this may take a moment.")
    generator.eval()
    fid_metric.reset()
    
    samples_processed = 0
    with torch.no_grad():
        for real_imgs, labels in dataloader:
            if samples_processed >= max_samples:
                break
                
            real_imgs = real_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = real_imgs.size(0)
            
            # Generate fake images conditioned on the REAL labels (to match dataset distribution)
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = generator(z, labels)
            
            # TorchMetrics FID requires uint8 images in [0, 255]
            real_imgs_uint8 = ((real_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            fake_imgs_uint8 = ((fake_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            
            # Update metric iteratively
            fid_metric.update(real_imgs_uint8, real=True)
            fid_metric.update(fake_imgs_uint8, real=False)
            
            samples_processed += bs
            
    # Compute final scalar
    fid_score = fid_metric.compute().item()
    generator.train()
    
    return fid_score
# ---------------------


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = auto_device(args.device)
    print(f"[train] Using device: {device}")

    # --- ADDED FOR FID ---
    if args.eval_fid_every > 0:
        if FrechetInceptionDistance is None:
            raise ImportError("Please install torchmetrics to evaluate FID: pip install \"torchmetrics[image]\"")
        # Initialize FID using the standard 2048-dim feature layer from InceptionV3
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    # ---------------------

    out_dir = Path(args.output_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = PatchDataset(
        args.data_dir, num_classes=args.num_classes,
        images_subdir=args.images_subdir, masks_subdir=args.masks_subdir,
        background_label=args.background_label,
    )

    if args.patch_size is None:
        sample_img, _ = dataset[0]
        args.patch_size = sample_img.shape[-1]
        print(f"[train] --patch_size not set, auto-detected {args.patch_size} "
              f"from the first image in your dataset.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True,
                         pin_memory=(device.type == "cuda"))

    G = Generator(latent_dim=args.latent_dim, num_classes=args.num_classes,
                  patch_size=args.patch_size, base_channels=args.g_base_channels).to(device)
    D = Discriminator(num_classes=args.num_classes, patch_size=args.patch_size,
                       base_channels=args.d_base_channels,
                       use_spectral_norm=not args.no_spectral_norm).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] Resumed from {args.resume} at epoch {start_epoch}")

    print(f"[train] {len(dataset)} patches/epoch, batch_size={args.batch_size}, "
          f"patch_size={args.patch_size}, num_classes={args.num_classes}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running_d, running_g, n_batches = 0.0, 0.0, 0

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = real_imgs.size(0)

            real_target = torch.full((bs, 1), args.label_smoothing, device=device)
            fake_target = torch.zeros((bs, 1), device=device)

            # --- Discriminator step ---
            opt_d.zero_grad(set_to_none=True)
            z = torch.randn(bs, args.latent_dim, device=device)
            with torch.no_grad():
                fake_imgs = G(z, labels)
            d_real = D(real_imgs, labels)
            d_fake = D(fake_imgs, labels)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(bs, args.latent_dim, device=device)
            fake_imgs = G(z, labels)
            d_fake_for_g = D(fake_imgs, labels)
            g_loss = criterion(d_fake_for_g, torch.full((bs, 1), 1.0, device=device))
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            n_batches += 1

        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] D_loss={running_d/n_batches:.4f} "
              f"G_loss={running_g/n_batches:.4f} ({dt:.1f}s)")

        if (epoch + 1) % args.sample_every == 0:
            sample_path = out_dir / "samples" / f"epoch_{epoch+1:04d}.png"
            save_sample_grid(G, args.num_classes, args.samples_per_class,
                              args.latent_dim, device, sample_path)
            print(f"  saved sample grid -> {sample_path}")

        # --- ADDED FOR FID ---
        if args.eval_fid_every > 0 and (epoch + 1) % args.eval_fid_every == 0:
            fid_score = evaluate_fid(G, loader, fid_metric, args.latent_dim, device, args.fid_samples)
            print(f"  --> FID Score: {fid_score:.4f}")
        # ---------------------

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = out_dir / "checkpoints" / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    print("[train] Done.")


if __name__ == "__main__":
    main()