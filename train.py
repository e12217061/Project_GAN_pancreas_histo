"""
train.py — train the pix2pix-style mask-conditioned GAN end to end.

Minimal example:
    python train.py --data_dir /path/to/data --num_classes 4

See `python train.py --help` for all options, and README.md for the folder layout.
"""
import argparse
import time
import os
import csv
import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from dataset import SpatialMaskDataset
from models import UNetGenerator, PatchDiscriminator
import sys
from gitlogger import GitHubLogger
from dotenv import load_dotenv

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None


# a fixed color per class index, used only for visualizing masks in sample grids
CLASS_COLORS = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description="pix2pix-style mask-conditioned GAN")
    # data
    p.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing images/ and masks/ subfolders (see README.md)")
    p.add_argument("--images_subdir", type=str, default="images")
    p.add_argument("--masks_subdir", type=str, default="masks")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--background_label", type=int, default=0,
                    help="Mask pixel value treated as 'no class' (all-zero one-hot). "
                         "Other pixel values are expected in [1, num_classes].")
    # model
    p.add_argument("--patch_size", type=int, default=None,
                    help="Must be a power of 2 (64/128/256/512) -- the U-Net needs "
                         "exact spatial alignment between its encoder and decoder "
                         "stages. Defaults to auto-detect from the first image.")
    p.add_argument("--noise_channels", type=int, default=0,
                    help="Extra random channels concatenated to the mask input for "
                         "some output stochasticity. Off by default: with the L1 loss "
                         "pix2pix uses, injected noise is well known to get mostly "
                         "ignored by the network anyway -- it's here if you want to "
                         "experiment, not because it's expected to do much on its own.")
    p.add_argument("--g_base_channels", type=int, default=64)
    p.add_argument("--g_max_channels", type=int, default=512)
    p.add_argument("--d_base_channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--n_dropout_layers", type=int, default=3,
                    help="How many of the innermost decoder layers use dropout "
                         "(matches the original pix2pix paper's setup).")
    p.add_argument("--no_spectral_norm", action="store_true")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--label_smoothing", type=float, default=0.9)
    p.add_argument("--l1_lambda", type=float, default=100.0,
                    help="Weight on the L1 reconstruction loss between generated and "
                         "real image. This is what actually teaches the generator to "
                         "match the mask's specific layout, not just 'look realistic' "
                         "-- pix2pix's standard value is 100.")
    p.add_argument("--r1_gamma", type=float, default=10.0,
                    help="R1 gradient penalty weight on D. Set to 0 to disable.")
    p.add_argument("--r1_every", type=int, default=16,
                    help="Apply R1 only every N discriminator steps (lazy, scaled to "
                         "compensate). Set to 1 for every step.")
    p.add_argument("--ema_decay", type=float, default=0.995,
                    help="EMA decay for G's weights. Sample grids, FID, and the "
                         "best-FID checkpoint all use the EMA generator.")
    p.add_argument("--lr_decay_start_frac", type=float, default=0.5,
                    help="Fraction of total --epochs after which LR starts linearly "
                         "decaying (both G and D).")
    p.add_argument("--lr_min_factor", type=float, default=0.1,
                    help="LR at the final epoch, as a fraction of the initial --lr.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches_per_epoch", type=int, default=0,
                    help="For quick dev runs: cap batches per epoch (0 = no limit).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    # output
    p.add_argument("--output_dir", type=str, default="./gan_outputs")
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_sample_pairs", type=int, default=4,
                    help="How many fixed (mask, image) pairs to show in each sample "
                         "grid -- the same ones every epoch, for a fair before/after "
                         "comparison.")
    p.add_argument("--eval_fid_every", type=int, default=1)
    p.add_argument("--fid_samples", type=int, default=2048)
    return p.parse_args()


def auto_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- R1 PENALTY ---
def r1_penalty(real_imgs, d_real):
    """Gradient penalty on D's output w.r.t. real images (Mescheder et al. 2018).
    Requires real_imgs.requires_grad_(True) before D(mask, real_imgs) was called."""
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(), inputs=real_imgs, create_graph=True
    )[0]
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


# --- EMA ---
def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# --- LR DECAY ---
def make_lr_lambda(total_epochs, decay_start_frac, min_factor):
    decay_start_epoch = int(total_epochs * decay_start_frac)

    def lr_lambda(epoch):
        if total_epochs <= decay_start_epoch or epoch < decay_start_epoch:
            return 1.0
        progress = (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
        return max(1.0 - (1.0 - min_factor) * progress, min_factor)

    return lr_lambda


def colorize_mask(onehot):
    """onehot: (num_classes, H, W) numpy array -> (H, W, 3) uint8 visualization."""
    num_classes, h, w = onehot.shape
    vis = np.full((h, w, 3), 20, dtype=np.uint8)  # dark gray = background/unlabeled
    for c in range(num_classes):
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        vis[onehot[c] > 0.5] = color
    return vis


def tensor_to_uint8(img_t):
    """(3, H, W) tensor in [-1, 1] -> (H, W, 3) uint8."""
    return ((img_t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()


def save_sample_grid(generator, masks, reals, device, path):
    """masks/reals: fixed (N, C, H, W) tensors, the same ones every call. Saves a grid
    with one row per pair: [colorized mask | generated image | real image]."""
    generator.eval()
    with torch.no_grad():
        fakes = generator(masks.to(device)).cpu()
    generator.train()

    n = masks.size(0)
    patch = reals.shape[-1]
    grid = np.zeros((n * patch, 3 * patch, 3), dtype=np.uint8)
    for i in range(n):
        mask_vis = colorize_mask(masks[i].numpy())
        fake_vis = tensor_to_uint8(fakes[i])
        real_vis = tensor_to_uint8(reals[i])
        grid[i * patch:(i + 1) * patch, 0:patch] = mask_vis
        grid[i * patch:(i + 1) * patch, patch:2 * patch] = fake_vis
        grid[i * patch:(i + 1) * patch, 2 * patch:3 * patch] = real_vis
    Image.fromarray(grid).save(path)


def evaluate_fid(generator, dataloader, fid_metric, device, max_samples=2048):
    print(f"  [FID] Computing over ~{max_samples} samples...")
    generator.eval()
    fid_metric.reset()
    samples_processed = 0
    with torch.no_grad():
        for real_imgs, masks in dataloader:
            if samples_processed >= max_samples:
                break
            real_imgs = real_imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            fake_imgs = generator(masks)

            real_u8 = ((real_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            fake_u8 = ((fake_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            fid_metric.update(real_u8, real=True)
            fid_metric.update(fake_u8, real=False)
            samples_processed += real_imgs.size(0)
    fid_score = fid_metric.compute().item()
    generator.train()
    return fid_score


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = auto_device(args.device)
    print(f"[train] Using device: {device}")


    load_dotenv()
    gh_token =  os.environ.get("GITHUB_TOKEN")

    gitlogger = GitHubLogger(token=gh_token, repo_name="e12217061/Project_GAN_pancreas_histo", issue_number=2)


    if args.eval_fid_every > 0:
        if FrechetInceptionDistance is None:
            raise ImportError("Please install torchmetrics to evaluate FID: pip install \"torchmetrics[image]\"")
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    out_dir = Path(args.output_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    csv_path = out_dir / "training_log.csv"
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s: %(message)s",
                         handlers=[logging.StreamHandler(sys.stdout),
                                   logging.FileHandler(log_path, mode="a")])
    logger = logging.getLogger("train")
    logger.info(f"Logging to {log_path}")

    pid_path = out_dir / "train.pid"
    try:
        pid_path.write_text(str(os.getpid()))
    except Exception:
        logger.debug("Could not write PID file.")

    dataset = SpatialMaskDataset(
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

    # fixed pairs for the sample grid -- same ones every epoch
    n_sample = min(args.num_sample_pairs, len(dataset))
    sample_reals, sample_masks = zip(*[dataset[i] for i in range(n_sample)])
    sample_reals = torch.stack(sample_reals)
    sample_masks = torch.stack(sample_masks)

    G = UNetGenerator(num_classes=args.num_classes, patch_size=args.patch_size,
                       noise_channels=args.noise_channels, base_channels=args.g_base_channels,
                       max_channels=args.g_max_channels, dropout=args.dropout,
                       n_dropout_layers=args.n_dropout_layers).to(device)
    D = PatchDiscriminator(num_classes=args.num_classes, base_channels=args.d_base_channels,
                            use_spectral_norm=not args.no_spectral_norm).to(device)

    G_ema = copy.deepcopy(G).to(device)
    G_ema.eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()

    lr_lambda = make_lr_lambda(args.epochs, args.lr_decay_start_frac, args.lr_min_factor)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    start_epoch = 0
    best_fid = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        if "G_ema" in ckpt:
            G_ema.load_state_dict(ckpt["G_ema"])
        else:
            G_ema.load_state_dict(G.state_dict())
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
            sched_d.load_state_dict(ckpt["sched_d"])
        best_fid = ckpt.get("best_fid", float("inf"))
        print(f"[train] Resumed from {args.resume} at epoch {start_epoch} "
              f"(best_fid so far: {best_fid:.4f})")

    logger.info(f"[train] {len(dataset)} pairs/epoch, batch_size={args.batch_size}, "
                f"patch_size={args.patch_size}, num_classes={args.num_classes}, "
                f"l1_lambda={args.l1_lambda}")

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "d_loss", "g_adv_loss", "g_l1_loss", "r1", "fid",
                              "lr", "epoch_time_s"])

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    epoch_iter = range(start_epoch, args.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="epochs", unit="epoch")

    global_step = 0
    for epoch in epoch_iter:
        t0 = time.time()
        running_d, running_g_adv, running_g_l1, n_batches = 0.0, 0.0, 0.0, 0
        running_r1, n_r1 = 0.0, 0

        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")

        batch_iter = loader
        if tqdm is not None:
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = None
            batch_iter = tqdm(loader, desc=f"epoch {epoch+1}", total=total_batches, leave=False)

        for real_imgs, masks in batch_iter:
            real_imgs = real_imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # --- Discriminator step ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_imgs = G(masks)

            apply_r1 = args.r1_gamma > 0 and (global_step % args.r1_every == 0)
            if apply_r1:
                real_imgs.requires_grad_(True)

            d_real = D(masks, real_imgs)
            d_fake = D(masks, fake_imgs)
            real_target = torch.full_like(d_real, args.label_smoothing)
            fake_target = torch.zeros_like(d_fake)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)

            if apply_r1:
                r1 = r1_penalty(real_imgs, d_real)
                d_loss = d_loss + (args.r1_gamma / 2) * r1 * args.r1_every
                running_r1 += r1.item()
                n_r1 += 1

            d_loss.backward()
            opt_d.step()
            global_step += 1

            # --- Generator step ---
            opt_g.zero_grad(set_to_none=True)
            fake_imgs = G(masks)
            d_fake_for_g = D(masks, fake_imgs)
            adv_target = torch.full_like(d_fake_for_g, 1.0)
            g_adv_loss = criterion(d_fake_for_g, adv_target)
            g_l1_loss = F.l1_loss(fake_imgs, real_imgs)
            g_loss = g_adv_loss + args.l1_lambda * g_l1_loss
            g_loss.backward()
            opt_g.step()

            update_ema(G_ema, G, args.ema_decay)

            running_d += d_loss.item()
            running_g_adv += g_adv_loss.item()
            running_g_l1 += g_l1_loss.item()
            n_batches += 1
            if args.max_batches_per_epoch > 0 and n_batches >= args.max_batches_per_epoch:
                break

        dt = time.time() - t0
        mean_d = running_d / n_batches if n_batches else float('nan')
        mean_g_adv = running_g_adv / n_batches if n_batches else float('nan')
        mean_g_l1 = running_g_l1 / n_batches if n_batches else float('nan')
        mean_r1 = running_r1 / n_r1 if n_r1 else float('nan')

        sched_g.step()
        sched_d.step()
        current_lr = opt_g.param_groups[0]["lr"]

        logger.info(f"[epoch {epoch+1}/{args.epochs}] D_loss={mean_d:.4f} "
                    f"G_adv={mean_g_adv:.4f} G_L1={mean_g_l1:.4f} R1={mean_r1:.4f} "
                    f"lr={current_lr:.6f} ({dt:.1f}s)")

        fid_score = None
        if args.eval_fid_every > 0 and (epoch + 1) % args.eval_fid_every == 0:
            try:
                fid_score = evaluate_fid(G_ema, loader, fid_metric, device, args.fid_samples)
                logger.info(f"  --> FID Score (EMA): {fid_score:.4f}")
            except Exception:
                logger.exception("Error while computing FID")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{mean_d:.6f}", f"{mean_g_adv:.6f}", f"{mean_g_l1:.6f}",
                              f"{mean_r1:.6f}" if n_r1 else "",
                              f"{fid_score:.6f}" if fid_score is not None else "",
                              f"{current_lr:.8f}", f"{dt:.3f}"])

        if (epoch + 1) % args.sample_every == 0:
            sample_path = out_dir / "samples" / f"pix2pix_epoch_{epoch+1:04d}.png"
            save_sample_grid(G_ema, sample_masks, sample_reals, device, sample_path)
            logger.info(f"  saved sample grid (mask | generated | real) -> {sample_path}")

    #GitLogger
        gitlogger.log_epoch(epoch=epoch+1, d_loss=mean_d, g_adv=mean_g_adv, g_l1=mean_g_l1, r1=mean_r1, lr=current_lr, fid_score=fid_score)

        local_file_path = f"gan_outputs/samples/pix2pix_epoch_{epoch+1:04d}.png"
        repo_destination_path = f"gan_outputs/samples/pix2pix_epoch_{epoch+1:04d}.png"
        gitlogger.commit_file(local_file_path, repo_destination_path, commit_message="Upload")

        def make_ckpt_dict():
            return {
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "G_ema": G_ema.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                "best_fid": best_fid, "args": vars(args),
            }

        if fid_score is not None and fid_score < best_fid:
            best_fid = fid_score
            best_path = out_dir / "checkpoints" / "best_fid.pt"
            ckpt_dict = make_ckpt_dict()
            ckpt_dict["best_fid"] = best_fid
            torch.save(ckpt_dict, best_path)
            logger.info(f"  New best FID ({best_fid:.4f}) -> saved {best_path}")

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = out_dir / "checkpoints" / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save(make_ckpt_dict(), ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    print("[train] Done.")


if __name__ == "__main__":
    main()
