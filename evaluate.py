import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from models import Generator

@torch.no_grad()
def generate_stratified_dataset(
    ckpt_path, out_dir, n_healthy, n_tumor, batch_size,
    patch_size, latent_dim, base_channels, device
):
    out_dir = Path(out_dir)
    dir_healthy = out_dir / "class_0_healthy"
    dir_tumor = out_dir / "class_1_cancer"
    dir_healthy.mkdir(parents=True, exist_ok=True)
    dir_tumor.mkdir(parents=True, exist_ok=True)

    print(f"Loading Generator from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Initialize Generator
    # Note: num_classes is hardcoded to 2 based on your architecture
    G = Generator(
        latent_dim=latent_dim, 
        num_classes=2, 
        patch_size=patch_size, 
        base_channels=base_channels
    ).to(device)

    # Always prefer G_ema for generation (vastly superior image quality)
    if "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"])
        print("Successfully loaded EMA weights for generation.")
    elif "G" in ckpt:
        G.load_state_dict(ckpt["G"])
        print("Loaded standard G weights.")
    else:
        G.load_state_dict(ckpt)
    
    G.eval()

    manifest_records = []

    # ==========================================
    # 1. GENERATE HEALTHY PATCHES (Score: 0.0)
    # ==========================================
    print(f"\nGenerating {n_healthy} Healthy patches (Score = 0.0)...")
    for i in tqdm(range(0, n_healthy, batch_size), desc="Healthy"):
        bs = min(batch_size, n_healthy - i)
        z = torch.randn(bs, latent_dim, device=device)
        labels = torch.zeros(bs, dtype=torch.long, device=device)
        scores = torch.zeros(bs, dtype=torch.float32, device=device)

        imgs = G(z, labels, scores)
        # Convert [-1, 1] tensor to [0, 255] uint8 numpy array
        imgs_np = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

        for j in range(bs):
            idx = i + j
            filename = f"synthetic_healthy_{idx:06d}.png"
            Image.fromarray(imgs_np[j]).save(dir_healthy / filename)
            manifest_records.append({
                "filename": filename,
                "label": 0,
                "attention_score": 0.0,
                "tumor_zone": "N/A"
            })

    # ==========================================
    # 2. GENERATE TUMOR PATCHES (60/40 Split)
    # ==========================================
    print(f"\nGenerating {n_tumor} Tumor patches (Stratified 60/40 Split)...")
    
    # Calculate exact counts for 60/40 split
    n_high = int(n_tumor * 0.60)
    n_mid = n_tumor - n_high
    
    # Pre-sample the continuous scores uniformly within their respective bounds
    scores_high = np.random.uniform(0.75, 1.0, size=n_high)
    scores_mid = np.random.uniform(0.30, 0.74, size=n_mid)
    
    # Combine and shuffle so the batches are mixed
    tumor_scores_np = np.concatenate([scores_high, scores_mid])
    np.random.shuffle(tumor_scores_np)
    tumor_scores_tensor = torch.tensor(tumor_scores_np, dtype=torch.float32, device=device)

    for i in tqdm(range(0, n_tumor, batch_size), desc="Tumor"):
        bs = min(batch_size, n_tumor - i)
        z = torch.randn(bs, latent_dim, device=device)
        labels = torch.ones(bs, dtype=torch.long, device=device)
        scores = tumor_scores_tensor[i : i + bs]

        imgs = G(z, labels, scores)
        imgs_np = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

        for j in range(bs):
            idx = i + j
            score_val = float(scores[j].item())
            zone = "Core" if score_val >= 0.75 else "Transition"
            filename = f"synthetic_tumor_{idx:06d}_score{score_val:.2f}.png"
            
            Image.fromarray(imgs_np[j]).save(dir_tumor / filename)
            manifest_records.append({
                "filename": filename,
                "label": 1,
                "attention_score": score_val,
                "tumor_zone": zone
            })

    # ==========================================
    # 3. SAVE TRACKING MANIFEST
    # ==========================================
    manifest_path = out_dir / "synthetic_manifest.csv"
    pd.DataFrame(manifest_records).to_csv(manifest_path, index=False)
    
    print("\n" + "="*50)
    print("Generation Complete!")
    print(f"Total Images Saved: {len(manifest_records)}")
    print(f"Tracking Manifest Saved: {manifest_path}")
    print("="*50 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to best_fid.pt or ckpt_epoch_xxxx.pt")
    p.add_argument("--out_dir", type=str, default="./synthetic_dataset")
    p.add_argument("--n_healthy", type=int, default=25000, help="Number of healthy patches to generate")
    p.add_argument("--n_tumor", type=int, default=25000, help="Number of tumor patches to generate")
    p.add_argument("--batch_size", type=int, default=64, help="Generation batch size (reduce if OOM)")
    p.add_argument("--patch_size", type=int, default=512, help="Must match your training patch size")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--g_base_channels", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generate_stratified_dataset(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        n_healthy=args.n_healthy,
        n_tumor=args.n_tumor,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        base_channels=args.g_base_channels,
        device=device
    )