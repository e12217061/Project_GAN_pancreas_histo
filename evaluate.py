"""
evaluate.py — Generate a synthetic dataset for TSTR and compute FID, KID, and IS.

Usage:
    # Balanced (equal samples per class):
    python evaluate.py --checkpoint gan_outputs/checkpoints/best_fid.pt \
                       --data_dir ./dataset \
                       --out_dir ./synthetic-data \
                       --samples_per_class 25000

    # Imbalanced / Custom per-class counts (e.g., 20000 Class 0, 73000 Class 1):
    python evaluate.py --checkpoint gan_outputs/checkpoints/best_fid.pt \
                       --data_dir ./dataset \
                       --out_dir ./synthetic-data \
                       --samples_per_class 20000 73000
"""
import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models import Generator
from dataset import PatchDataset
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_fid.pt")
    p.add_argument("--data_dir", type=str, required=True, help="Path to real dataset (for FID/KID comparison)")
    p.add_argument("--out_dir", type=str, required=True, help="Where to save the synthetic images")
    p.add_argument(
        "--samples_per_class", 
        type=int, 
        nargs="+", 
        default=[25000], 
        help="Number of synthetic images to generate. Provide 1 number to apply equally across all classes, "
             "or one number per class in order (e.g., --samples_per_class 20000 73000)."
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]

    # Initialize Generator using the same arguments used during training
    G = Generator(
        latent_dim=train_args["latent_dim"], 
        num_classes=train_args["num_classes"],
        patch_size=train_args["patch_size"], 
        base_channels=train_args["g_base_channels"]
    ).to(device)
    
    # Use EMA weights for evaluation if available
    if "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"])
        print("Loaded EMA weights for optimal quality.")
    else:
        G.load_state_dict(ckpt["G"])
        print("Loaded standard Generator weights.")
    
    G.eval()

    # Load real dataset to obtain class index-to-name mapping and real distributions
    print("Preparing real dataset...")
    real_dataset = PatchDataset(args.data_dir, augment=False, stain_normalize=False)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    class_names = real_dataset.class_names
    num_classes = len(class_names)

    # Resolve per-class generation counts
    if len(args.samples_per_class) == 1:
        class_sample_targets = {i: args.samples_per_class[0] for i in range(num_classes)}
    elif len(args.samples_per_class) == num_classes:
        class_sample_targets = {i: count for i, count in enumerate(args.samples_per_class)}
    else:
        raise ValueError(
            f"Invalid number of values for --samples_per_class. Expected either 1 value or "
            f"{num_classes} values (one per class in {class_names}), got {len(args.samples_per_class)}: {args.samples_per_class}"
        )

    print("\nGeneration targets per class:")
    for idx, name in enumerate(class_names):
        print(f"  Class {idx} ({name}): {class_sample_targets[idx]:,} patches")

    # Initialize Metrics
    print("\nInitializing metrics (InceptionV3)...")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)
    inception = InceptionScore().to(device)

    # Setup output directories
    out_dir = Path(args.out_dir) / "images"
    for c_name in class_names:
        (out_dir / c_name).mkdir(parents=True, exist_ok=True)

    # 1. Generate Synthetic Images per Class
    print("\nGenerating synthetic samples...")
    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            target_count = class_sample_targets[class_idx]
            generated_count = 0
            pbar = tqdm(total=target_count, desc=f"Generating {class_name} (Class {class_idx})")
            
            while generated_count < target_count:
                bs = min(args.batch_size, target_count - generated_count)
                
                z = torch.randn(bs, train_args["latent_dim"], device=device)
                labels = torch.full((bs,), class_idx, dtype=torch.long, device=device)
                fake_imgs = G(z, labels)
                
                # Convert to uint8 for saving and Inception metrics
                fake_imgs_uint8 = ((fake_imgs.clamp(-1, 1) + 1) * 127.5).byte()
                
                # Save to disk for TSTR
                for i in range(bs):
                    img_np = fake_imgs_uint8[i].permute(1, 2, 0).cpu().numpy()
                    img_path = out_dir / class_name / f"synth_{generated_count + i:06d}.png"
                    Image.fromarray(img_np).save(img_path)
                
                # Update metric buffers
                fid.update(fake_imgs_uint8, real=False)
                kid.update(fake_imgs_uint8, real=False)
                inception.update(fake_imgs_uint8)
                
                generated_count += bs
                pbar.update(bs)
            pbar.close()

    # 2. Extract Real Features for FID/KID
    total_fake_samples = sum(class_sample_targets.values())
    target_real_samples = min(total_fake_samples, len(real_dataset))
    real_processed = 0
    
    print(f"\nExtracting features from real images (target: {target_real_samples:,} images)...")
    with torch.no_grad():
        pbar = tqdm(total=target_real_samples, desc="Processing Real Images")
        for real_imgs, _ in real_loader:
            if real_processed >= target_real_samples:
                break
                
            bs = real_imgs.size(0)
            if real_processed + bs > target_real_samples:
                bs = target_real_samples - real_processed
                real_imgs = real_imgs[:bs]
                
            real_imgs = real_imgs.to(device)
            real_imgs_uint8 = ((real_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            
            fid.update(real_imgs_uint8, real=True)
            kid.update(real_imgs_uint8, real=True)
            
            real_processed += bs
            pbar.update(bs)
        pbar.close()

    # 3. Compute and Print Final Scores
    print("\nComputing final metrics (this may take a few minutes)...")
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()

    print("=" * 45)
    print("FINAL EVALUATION METRICS")
    print("=" * 45)
    print(f"FID Score:          {fid_score:.4f}")
    print(f"KID Score:          {kid_mean.item():.6f} ± {kid_std.item():.6f}")
    print(f"Inception Score:    {is_mean.item():.4f} ± {is_std.item():.4f}")
    print("=" * 45)
    print(f"Synthetic dataset saved to: {out_dir.absolute()}")

if __name__ == "__main__":
    main()