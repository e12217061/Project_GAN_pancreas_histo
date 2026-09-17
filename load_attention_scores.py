"""
Load per-patch attention scores (from your trained AB-MIL model) out of a
manifest CSV, keyed by image filename.
"""

import argparse
from pathlib import Path
from typing import Dict
import pandas as pd


def load_attention_manifest(csv_path: str, path_col: str = "filename", score_col: str = "scaled_attention_score") -> Dict[str, float]:
    """Returns {basename: attention_score} for every row in the manifest."""
    df = pd.read_csv(csv_path)

    # Hard-fail if the expected columns aren't there
    if path_col not in df.columns or score_col not in df.columns:
        raise ValueError(
            f"Manifest is missing required columns! "
            f"Expected '{path_col}' and '{score_col}'. Found: {list(df.columns)}"
        )

    scores: Dict[str, float] = {}
    for _, row in df.iterrows():
        # Path().name ensures we strip out any folders and just keep "patch_001.png"
        basename = Path(str(row[path_col])).name
        scores[basename] = float(row[score_col])
        
    return scores


# =====================================================================
# LIVE FILE PARSER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and verify the real patch manifest.")
    parser.add_argument("--manifest", required=True, help="Path to your scaled patch manifest CSV")
    args = parser.parse_args()

    try:
        print(f"Loading manifest from: {args.manifest}")
        # 1. Load the dictionary
        scores_dict = load_attention_manifest(args.manifest)
        
        # 2. Print Summary
        print(f"\n✅ Successfully loaded {len(scores_dict)} patches into memory!")
        
        # 3. Print a quick sample of the first 5 items
        print("\n--- Quick Sanity Check (First 5 patches) ---")
        for i, (basename, score) in enumerate(scores_dict.items()):
            print(f"{basename} -> {score:.4f}")
            if i >= 4:
                break
                
        # 4. Verify Min/Max scaling
        if scores_dict:
            max_patch = max(scores_dict, key=scores_dict.get)
            min_patch = min(scores_dict, key=scores_dict.get)
            print("\n--- Distribution Check ---")
            print(f"Highest Score Patch: {max_patch} -> {scores_dict[max_patch]:.6f}")
            print(f"Lowest Score Patch:  {min_patch} -> {scores_dict[min_patch]:.6f}")
            
    except Exception as e:
        print(f"\n❌ Error loading manifest: {e}")