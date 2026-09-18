"""
evaluate.py — Generate a synthetic dataset for TSTR and compute FID, KID, and IS.
Handles continuous AB-MIL attention conditioning.

Usage:
    # 1. Generate ONLY (Save 10k healthy, 10k tumor to disk)
    python evaluate.py --mode generate \
                       --checkpoint gan_outputs/checkpoints/best_fid.pt \
                       --data_dir ./dataset \
                       --out_dir ./synthetic-data \
                       --samples_per_class 10000 10000

    # 2. Evaluate ONLY (Compare existing synthetic folder against real data)
    python evaluate.py --mode evaluate \
                       --data_dir ./dataset \
                       --synthetic_dir ./synthetic-data/images

    # 3. Both (Generate data to disk, then immediately calculate metrics)
    python evaluate.py --mode both \
                       --checkpoint gan_outputs/checkpoints/best_fid.pt \
                       --data_dir ./dataset \
                       --out_dir ./synthetic-data \
                       --samples_per_class 10000 10000
"""
import argparse
import os
import torch
import numpy as np
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
    p.add_argument("--mode", type=str, required=True, choices=["generate", "evaluate", "both"], 
                   help="Action to perform: 'generate', 'evaluate', or 'both'.")
    
    # Required for generation
    p.add_argument("--checkpoint", type=str, default=None, help="Path to best_fid.pt")
    p.add_argument("--out_dir", type=str, default=None, help="Where to save the synthetic images")
    p.add_argument("--samples_per_class", type=int, nargs="+", default=[10000], 
                   help="Number of synthetic images to generate. Provide 1 number for all classes, or one per class.")
    
    # Required for evaluation
    p.add_argument("--data_dir", type=str, required=True, help="Path to real dataset")
    p.add_argument("--synthetic_dir", type=str, default=None, help="Path to existing synthetic images (only needed if mode=evaluate)")
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Inherited from dataset.py
    p.add_argument("--attention_manifest", type=str, default=None, help="Path to your scaled attention CSV. Needed to load real dataset.")
    
    return p.parse_args()

@torch.no_grad()
def generate_images(args, device, real_dataset):
    print(f"\n--- GENERATION STAGE ---")
    if not args.checkpoint or not args.out_dir:
        raise ValueError("--checkpoint and --out_dir are required for generation.")
        
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    G = Generator(
        latent_dim=train_args["latent_dim"], 
        num_classes=train_args["num_classes"],
        attn_embed_dim=train_args.get("attn_embed_dim", 32),
        patch_size=train_args["patch_size"], 
        base_channels=train_args["g_base_channels"]
    ).to(device)
    
    if "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"])
        print("Loaded EMA weights for optimal quality.")
    else:
        G.load_state_dict(ckpt["G"])
        
    G.eval()

    class_names = real_dataset.class_names
    num_classes = len(class_names)

    if len(args.samples_per_class) == 1:
        class_sample_targets = {i: args.samples_per_class[0] for i in range(num_classes)}
    elif len(args.samples_per_class) == num_classes:
        class_sample_targets = {i: count for i, count in enumerate(args.samples_per_class)}
    else:
        raise ValueError(f"Invalid --samples_per_class format.")

    out_dir = Path(args.out_dir) / "images"
    for c_name in class_names:
        (out_dir / c_name).mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        target_count = class_sample_targets[class_idx]
        if target_count <= 0: continue
        
        # Determine attention scores for synthetic sampling
        if class_idx == 0 or "healthy" in class_name.lower():
            scores_np = np.zeros(target_count)
        else:
            # 60/40 Stratified Split for Tumor
            n_high = int(target_count * 0.60)
            n_mid = target_count - n_high
            scores_high = np.random.uniform(0.75, 1.0, size=n_high)
            scores_mid = np.random.uniform(0.30, 0.74, size=n_mid)
            scores_np = np.concatenate([scores_high, scores_mid])
            np.random.shuffle(scores_np)
            
        scores_tensor = torch.tensor(scores_np, dtype=torch.float32, device=device)

        generated_count = 0
        pbar = tqdm(total=target_count, desc=f"Generating {class_name}")
        
        while generated_count < target_count:
            bs = min(args.batch_size, target_count - generated_count)
            z = torch.randn(bs, train_args["latent_dim"], device=device)
            labels = torch.full((bs,), class_idx, dtype=torch.long, device=device)
            scores = scores_tensor[generated_count : generated_count + bs]
            
            fake_imgs = G(z, labels, scores)
            fake_imgs_uint8 = ((fake_imgs.clamp(-1, 1) + 1) * 127.5).byte()
            
            for i in range(bs):
                img_np = fake_imgs_uint8[i].permute(1, 2, 0).cpu().numpy()
                img_path = out_dir / class_name / f"synth_{generated_count + i:06d}.png"
                Image.fromarray(img_np).save(img_path)
            
            generated_count += bs
            pbar.update(bs)
        pbar.close()
        
    return out_dir

@torch.no_grad()
def evaluate_images(args, device, real_dataset, synthetic_dir):
    print(f"\n--- EVALUATION STAGE ---")
    print("\nInitializing metrics (InceptionV3)...")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)
    inception = InceptionScore().to(device)

    # 1. Load and process Synthetic Images
    print(f"Loading synthetic images from: {synthetic_dir}")
    synth_dataset = PatchDataset(synthetic_dir, augment=False, stain_normalize=False, attention_fallback=0.0)
    synth_loader = DataLoader(synth_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("Extracting features from SYNTHETIC images...")
    for synth_imgs, _, _ in tqdm(synth_loader, desc="Synthetic Features"):
        synth_imgs = synth_imgs.to(device)
        synth_imgs_uint8 = ((synth_imgs.clamp(-1, 1) + 1) * 127.5).byte()
        
        fid.update(synth_imgs_uint8, real=False)
        kid.update(synth_imgs_uint8, real=False)
        inception.update(synth_imgs_uint8)

    # 2. Load and process Real Images
    target_real_samples = len(synth_dataset) # Match counts for fair comparison
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"\nExtracting features from REAL images (target max: {target_real_samples:,})...")
    real_processed = 0
    for real_imgs, _, _ in tqdm(real_loader, desc="Real Features"):
        if real_processed >= target_real_samples: break
            
        bs = real_imgs.size(0)
        if real_processed + bs > target_real_samples:
            bs = target_real_samples - real_processed
            real_imgs = real_imgs[:bs]
            
        real_imgs = real_imgs.to(device)
        real_imgs_uint8 = ((real_imgs.clamp(-1, 1) + 1) * 127.5).byte()
        
        fid.update(real_imgs_uint8, real=True)
        kid.update(real_imgs_uint8, real=True)
        real_processed += bs

    # 3. Compute and Print Final Scores
    print("\nComputing final metrics (this may take a few minutes)...")
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()

    print("\n" + "=" * 45)
    print("FINAL EVALUATION METRICS")
    print("=" * 45)
    print(f"FID Score:          {fid_score:.4f}")
    print(f"KID Score:          {kid_mean.item():.6f} ± {kid_std.item():.6f}")
    print(f"Inception Score:    {is_mean.item():.4f} ± {is_std.item():.4f}")
    print("=" * 45 + "\n")

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    if args.attention_manifest is None and args.mode != "evaluate":
         print("Warning: No attention_manifest provided. Healthy slides will default to 0.0, but Tumor slides may fail to load in PatchDataset if missing.")

    # We always need the real dataset for FID/KID comparison
    print("Loading real dataset framework...")
    real_dataset = PatchDataset(args.data_dir, augment=False, stain_normalize=False, 
                                attention_manifest=args.attention_manifest, attention_fallback=0.0)
    
    synthetic_dir = args.synthetic_dir

    if args.mode in ["generate", "both"]:
        synthetic_dir = generate_images(args, device, real_dataset)
        
    if args.mode in ["evaluate", "both"]:
        if not synthetic_dir or not Path(synthetic_dir).exists():
            raise ValueError("Could not find synthetic directory to evaluate. Pass --synthetic_dir or use mode 'both'.")
        evaluate_images(args, device, real_dataset, synthetic_dir)

if __name__ == "__main__":
    main()