"""
dataset.py — loads a 512x512 (or whatever size your data already is) image and its
same-named mask from a folder. No cropping, no resizing: your images are already the
right size, so this just matches each image to its mask and reads a class label off
the mask.

EXPECTED FOLDER LAYOUT (edit find_pairs() below if yours differs):

    <data_dir>/
        images/
            slide_001.png
            slide_002.png
            ...
        masks/
            slide_001.png      <- same filename as its image
            slide_002.png
            ...

Each mask's pixel value encodes the class at that location (e.g. 0/1/2/3 for a
4-class problem). Since your images are already selected/cropped to be dominated by
one class, the label is just the majority value across the whole mask.

Random selection itself is handled by PyTorch: train.py creates the DataLoader with
shuffle=True, which already draws a random image each step -- no custom randomness
needed in here.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_pairs(data_dir, images_subdir="images", masks_subdir="masks"):
    """Return a list of (image_path, mask_path) pairs matched by filename stem."""
    data_dir = Path(data_dir)
    img_dir = data_dir / images_subdir
    mask_dir = data_dir / masks_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Expected an images folder at: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected a masks folder at: {mask_dir}")

    mask_lookup = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}

    pairs, missing = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_lookup.get(p.stem)
        if mask_path is None:
            missing.append(p.name)
            continue
        pairs.append((p, mask_path))

    if missing:
        print(f"[dataset] Warning: {len(missing)} image(s) had no matching mask and were "
              f"skipped, e.g. {missing[:5]}")
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found under {data_dir}. "
                            f"Check --images_subdir/--masks_subdir and filenames.")
    return pairs


class PatchDataset(Dataset):
    """One item = one (already-cropped) image + its majority-vote class label."""

    def __init__(self, data_dir, num_classes=4, images_subdir="images", masks_subdir="masks",
                 background_label=None):
        self.pairs = find_pairs(data_dir, images_subdir, masks_subdir)
        self.num_classes = num_classes
        self.background_label = background_label
        print(f"[dataset] Found {len(self.pairs)} image/mask pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode not in ("L", "I", "P"):
            mask = mask.convert("L")

        image_arr = np.array(image)
        mask_arr = np.array(mask)
        if image_arr.shape[:2] != mask_arr.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {img_path.name}: "
                              f"{image_arr.shape[:2]} vs {mask_arr.shape[:2]}")

        vals, counts = np.unique(mask_arr, return_counts=True)
        if self.background_label is not None:
            keep = vals != self.background_label
            vals, counts = vals[keep], counts[keep]
        if counts.size == 0:
            raise ValueError(f"Mask for {img_path.name} has no non-background pixels.")
        label = int(vals[np.argmax(counts)])
        if not (0 <= label < self.num_classes):
            raise ValueError(f"Mask for {img_path.name} has dominant class value {label}, "
                              f"outside expected range [0, {self.num_classes}). Check "
                              f"--num_classes / --background_label.")

        image_t = torch.from_numpy(image_arr.astype(np.float32) / 127.5 - 1.0)
        image_t = image_t.permute(2, 0, 1).contiguous()  # CHW, scaled to [-1, 1]
        return image_t, label
