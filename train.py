"""
train.py — train the class-conditional patch GAN end to end.

Minimal example:
    python train.py --data_dir /path/to/data --patch_size 128

The number of classes is no longer a flag you set -- it's auto-detected from the
number of class subfolders under --data_dir/--images_subdir (e.g. healthy/, tumor/),
the same way --patch_size auto-detects from the first image if left unset.

See `python train.py --help` for all options. Also see README.md for the expected
folder layout and how to adapt dataset.py if your data isn't organized that way.
"""
import argparse
import time
import os
import csv
import copy
import logging
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from dataset import PatchDataset
from models import Generator, Discriminator
from gitlogger import GitHubLogger
from dotenv import load_dotenv
import sys

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
                    help="Folder containing an images/ subfolder with one subdirectory "
                         "per class, e.g. images/healthy/, images/tumor/ (see README.md)")
    p.add_argument("--images_subdir", type=str, default="images")
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
    p.add_argument("--r1_gamma", type=float, default=10.0,
                    help="R1 gradient penalty weight on the discriminator (Mescheder et "
                         "al. 2018). Pulls D's gradients toward zero near real images "
                         "instead of letting it form sharp, easily-exploitable boundaries "
                         "around a handful of memorized images. Set to 0 to disable.")
    p.add_argument("--r1_every", type=int, default=16,
                    help="Apply the R1 penalty only every N discriminator steps (lazy "
                         "regularization, scaled to compensate) -- cheaper than every "
                         "step with a very similar effect. Set to 1 to apply every step.")
    p.add_argument("--ema_decay", type=float, default=0.995,
                    help="Decay for the exponential moving average of G's weights. The "
                         "EMA copy is what's used for sample grids, FID, and the "
                         "best-FID checkpoint -- it smooths out exactly the kind of "
                         "epoch-to-epoch quality swings a raw GAN checkpoint can have. "
                         "Lower (e.g. 0.99) adapts faster if you only get a few hundred "
                         "total steps; higher (0.999) is standard for long runs.")
    p.add_argument("--lr_decay_start_frac", type=float, default=0.5,
                    help="Fraction of total --epochs after which LR starts linearly "
                         "decaying (both G and D). 0.5 with --epochs 100 means constant "
                         "LR for 50 epochs, then decay.")
    p.add_argument("--lr_min_factor", type=float, default=0.1,
                    help="LR at the final epoch, as a fraction of the initial --lr.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches_per_epoch", type=int, default=5,
                    help="For quick dev runs: process at most this many batches per epoch (0 = no limit)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                    help="'cuda', 'mps', or 'cpu'. Auto-detected if not set.")
    # output
    p.add_argument("--output_dir", type=str, default="./gan_outputs")
    p.add_argument("--sample_every", type=int, default=1, help="Save a sample grid every N epochs.")
    p.add_argument("--checkpoint_every", type=int, default=3, help="Save a checkpoint every N epochs.")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt to resume from.")
    p.add_argument("--samples_per_class", type=int, default=4)
    
    # --- ADDED FOR FID ---
    p.add_argument("--eval_fid_every", type=int, default=1, 
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


# --- ADDED FOR R1 PENALTY ---
def r1_penalty(real_imgs, d_real):
    """Gradient penalty on D's output w.r.t. real images (Mescheder et al. 2018).
    Requires real_imgs.requires_grad_(True) before D(real_imgs, ...) was called."""
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(), inputs=real_imgs, create_graph=True
    )[0]
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
# ---------------------


# --- ADDED FOR EMA ---
def update_ema(ema_model, model, decay):
    """In-place EMA update of ema_model's parameters toward model's current parameters.
    Buffers (e.g. BatchNorm running stats) are copied directly rather than EMA'd, since
    they're already a running average themselves."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)
# ---------------------


# --- ADDED FOR CLASS-EMBEDDING COLLAPSE MONITORING ---
def embedding_collapse_metric(embed_weight):
    """Mean pairwise cosine similarity between class embedding vectors (off-diagonal
    only). Close to 1.0 means the classes are becoming indistinguishable to the
    generator -- a leading indicator of the cross-class texture collapse seen when D
    overpowers G (same output regardless of the class label fed in)."""
    w = F.normalize(embed_weight, dim=1)
    sim = w @ w.t()
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    off_diag_sum = sim.sum() - sim.diagonal().sum()
    return (off_diag_sum / (n * (n - 1))).item()
# ---------------------


# --- ADDED FOR LR DECAY ---
def make_lr_lambda(total_epochs, decay_start_frac, min_factor):
    decay_start_epoch = int(total_epochs * decay_start_frac)

    def lr_lambda(epoch):
        if total_epochs <= decay_start_epoch or epoch < decay_start_epoch:
            return 1.0
        progress = (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
        return max(1.0 - (1.0 - min_factor) * progress, min_factor)

    return lr_lambda
# ---------------------


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


    load_dotenv()
    gh_token =  os.environ.get("GITHUB_TOKEN")

    gitlogger = GitHubLogger(token=gh_token, repo_name="e12217061/Project_GAN_pancreas_histo", issue_number=2)

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

    # Logging setup: both console and file
    log_path = out_dir / "train.log"
    csv_path = out_dir / "training_log.csv"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_path, mode="a")])
    logger = logging.getLogger("train")
    logger.info(f"Logging to {log_path}")

    # Write PID file so users can check if the process is running
    pid_path = out_dir / "train.pid"
    try:
        pid_path.write_text(str(os.getpid()))
    except Exception:
        logger.debug("Could not write PID file.")

    dataset = PatchDataset(args.data_dir, images_subdir=args.images_subdir)
    args.num_classes = dataset.num_classes

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

    # --- ADDED FOR EMA ---
    # G_ema is a smoothed copy of G's weights, never trained directly via gradients --
    # it's what we sample from, evaluate FID on, and save as the "best" checkpoint.
    G_ema = copy.deepcopy(G).to(device)
    G_ema.eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    # ---------------------

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()

    # --- ADDED FOR LR DECAY ---
    lr_lambda = make_lr_lambda(args.epochs, args.lr_decay_start_frac, args.lr_min_factor)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
    # ---------------------

    start_epoch = 0
    best_fid = float("inf")  # --- ADDED FOR BEST-FID CHECKPOINTING ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        # backward-compatible: older checkpoints (before this update) won't have these
        if "G_ema" in ckpt:
            G_ema.load_state_dict(ckpt["G_ema"])
        else:
            G_ema.load_state_dict(G.state_dict())
            logger.warning("Checkpoint has no G_ema -- reinitialized EMA from raw G.")
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
            sched_d.load_state_dict(ckpt["sched_d"])
        best_fid = ckpt.get("best_fid", float("inf"))
        print(f"[train] Resumed from {args.resume} at epoch {start_epoch} "
              f"(best_fid so far: {best_fid:.4f})")

    logger.info(f"[train] {len(dataset)} patches/epoch, batch_size={args.batch_size}, "
                f"patch_size={args.patch_size}, num_classes={args.num_classes}")

    # Setup CSV header if needed
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "d_loss", "g_loss", "r1", "fid", "embed_sim", "lr",
                              "epoch_time_s"])
    logger.info(f"Training CSV log at {csv_path}")

    # Prefer tqdm for a nicer progress bar if available; fallback to simple loop
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    epoch_iter = range(start_epoch, args.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="epochs", unit="epoch")

    global_step = 0  # counts discriminator steps, used to schedule the lazy R1 penalty

    for epoch in epoch_iter:
        t0 = time.time()
        running_d, running_g, n_batches = 0.0, 0.0, 0
        running_r1, n_r1 = 0.0, 0

        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")

        # per-batch progress bar (if tqdm available)
        batch_iter = loader
        if tqdm is not None:
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = None
            batch_iter = tqdm(loader, desc=f"epoch {epoch+1}", total=total_batches, leave=False)

        for real_imgs, labels in batch_iter:
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

            # --- ADDED FOR R1 PENALTY ---
            apply_r1 = args.r1_gamma > 0 and (global_step % args.r1_every == 0)
            if apply_r1:
                real_imgs.requires_grad_(True)
            # ---------------------

            d_real = D(real_imgs, labels)
            d_fake = D(fake_imgs, labels)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)

            # --- ADDED FOR R1 PENALTY ---
            if apply_r1:
                r1 = r1_penalty(real_imgs, d_real)
                # scaled by r1_every to compensate for only applying it periodically
                d_loss = d_loss + (args.r1_gamma / 2) * r1 * args.r1_every
                running_r1 += r1.item()
                n_r1 += 1
            # ---------------------

            d_loss.backward()
            opt_d.step()
            global_step += 1

            # --- Generator step ---
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(bs, args.latent_dim, device=device)
            fake_imgs = G(z, labels)
            d_fake_for_g = D(fake_imgs, labels)
            g_loss = criterion(d_fake_for_g, torch.full((bs, 1), 1.0, device=device))
            g_loss.backward()
            opt_g.step()

            # --- ADDED FOR EMA ---
            update_ema(G_ema, G, args.ema_decay)
            # ---------------------

            running_d += d_loss.item()
            running_g += g_loss.item()
            n_batches += 1
            # optional early-exit for dev runs
            if args.max_batches_per_epoch > 0 and n_batches >= args.max_batches_per_epoch:
                break

        dt = time.time() - t0
        mean_d = running_d / n_batches if n_batches else float('nan')
        mean_g = running_g / n_batches if n_batches else float('nan')
        mean_r1 = running_r1 / n_r1 if n_r1 else float('nan')

        # --- ADDED FOR LR DECAY ---
        sched_g.step()
        sched_d.step()
        current_lr = opt_g.param_groups[0]["lr"]
        # ---------------------

        # --- ADDED FOR CLASS-EMBEDDING COLLAPSE MONITORING ---
        embed_sim = embedding_collapse_metric(G.label_embed.weight)
        # ---------------------

        logger.info(f"[epoch {epoch+1}/{args.epochs}] D_loss={mean_d:.4f} G_loss={mean_g:.4f} "
                    f"R1={mean_r1:.4f} embed_sim={embed_sim:.4f} lr={current_lr:.6f} ({dt:.1f}s)")

        # --- ADDED FOR FID (now evaluated on the EMA generator) ---
        fid_score = None
        if args.eval_fid_every > 0 and (epoch + 1) % args.eval_fid_every == 0:
            try:
                fid_score = evaluate_fid(G_ema, loader, fid_metric, args.latent_dim, device, args.fid_samples)
                logger.info(f"  --> FID Score (EMA): {fid_score:.4f}")
            except Exception as e:
                logger.exception("Error while computing FID")
        # ---------------------

        # Update progress bar postfix if available
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            try:
                epoch_iter.set_postfix({"D_loss": f"{mean_d:.4f}", "G_loss": f"{mean_g:.4f}",
                                         "R1": f"{mean_r1:.4f}" if n_r1 else "NA",
                                         "FID": f"{fid_score if fid_score is not None else 'NA'}"})
            except Exception:
                pass

        # Append to CSV log
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{mean_d:.6f}", f"{mean_g:.6f}",
                              f"{mean_r1:.6f}" if n_r1 else "",
                              f"{fid_score:.6f}" if fid_score is not None else "",
                              f"{embed_sim:.6f}", f"{current_lr:.8f}", f"{dt:.3f}"])

        if (epoch + 1) % args.sample_every == 0:
            sample_path = out_dir / "samples" / f"epoch_{epoch+1:04d}.png"
            save_sample_grid(G_ema, args.num_classes, args.samples_per_class,
                              args.latent_dim, device, sample_path)
            logger.info(f"  saved sample grid (EMA) -> {sample_path}")

        # GITHUB LOGGING
        gitlogger.log_epoch(epoch=epoch, d_loss=mean_d, g_loss=mean_g, fid_score=fid_score)

        local_file_path = f"gan_outputs/samples/epoch_{epoch+1:04d}.png"
        repo_destination_path = f"gan_outputs/samples/epoch_{epoch+1:04d}.png"
        gitlogger.commit_file(local_file_path, repo_destination_path, commit_message=f"cGAN: Sample Upload: Epoch {epoch+1:04d}")

        def make_ckpt_dict():
            return {
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "G_ema": G_ema.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                "best_fid": best_fid, "args": vars(args),
            }

        # --- ADDED FOR BEST-FID CHECKPOINTING ---
        if fid_score is not None and fid_score < best_fid:
            best_fid = fid_score
            best_path = out_dir / "checkpoints" / "best_fid.pt"
            ckpt_dict = make_ckpt_dict()
            ckpt_dict["best_fid"] = best_fid
            torch.save(ckpt_dict, best_path)
            logger.info(f"  New best FID ({best_fid:.4f}) -> saved {best_path}")
        # ---------------------

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = out_dir / "checkpoints" / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save(make_ckpt_dict(), ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    print("[train] Done.")


if __name__ == "__main__":
    main()