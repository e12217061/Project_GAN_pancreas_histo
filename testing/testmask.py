import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

mask_dir = Path(__file__).resolve().parent / "dataset" / "masks" / "stroma"

if not mask_dir.exists():
    raise FileNotFoundError(f"Folder not found: {mask_dir}")

# only consider common image extensions
files = sorted([
    p
    for p in mask_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
])
if not files:
    raise FileNotFoundError(f"No image files found in {mask_dir}")


def testmask():
    # Load all mask images into numpy arrays (grayscale)
    arrays = []
    for p in files:
        with Image.open(p) as img:
            arr = np.array(img.convert("L"))
        arrays.append(arr)

    # If all masks share the same shape, stack into a single ndarray
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        masks = np.stack(arrays)  # shape (N, H, W)
    else:
        masks = np.array(arrays, dtype=object)

    print(f"Loaded {len(arrays)} mask files from {mask_dir}")
    print(f"masks type: {type(masks)}, shape: {getattr(masks, 'shape', None)}, dtype: {masks.dtype}")
    print(f"Example: first file {files[0].name} -> shape={arrays[0].shape}, min={arrays[0].min()}, max={arrays[0].max()}")

def testmask_onehot():
    # Load all mask images into numpy arrays (grayscale)
    arrays = []
    for p in files:
        with Image.open(p) as img:
            arr = np.array(img.convert("L"))
        arrays.append(arr)

    # If all masks share the same shape, stack into a single ndarray
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        masks = np.stack(arrays)  # shape (N, H, W)
    else:
        masks = np.array(arrays, dtype=object)

    # Convert to one-hot encoding
    num_classes = 2  # background and tumor
    onehot_masks = np.zeros((len(masks), num_classes, *masks[0].shape), dtype=np.float32)
    for i, mask in enumerate(masks):
        onehot_masks[i, 0] = (mask == 0).astype(np.float32)  # background
        onehot_masks[i, 1] = (mask > 0).astype(np.float32)   # tumor

    print(f"One-hot encoded masks shape: {onehot_masks.shape}, dtype: {onehot_masks.dtype}")


def testvalue():
    # Load all mask images into numpy arrays (grayscale)
    arrays = []
    i=0
    for p in files:
        with Image.open(p) as img:
            arr = np.array(img.convert("L"))
            arr = arr * 60
            print(arr)
        i += 1
        if i > 5:
            break

def main():
    testvalue()

if __name__ == "__main__":
    main()