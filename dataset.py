"""
dataset.py — paired (mask, image) loader for spatial mask-conditioned image synthesis
(pix2pix-style). Unlike the class-conditional version, this returns the FULL mask as a
one-hot spatial tensor (num_classes, H, W) instead of collapsing it to one label per
image -- the generator learns "put class C here" from the mask's actual layout.

EXPECTED FOLDER LAYOUT (same as your class-conditional dataset.py, unchanged):

    <data_dir>/
        images/
            adjacent_benign/
                slide_001.png
            stroma/
                slide_002.png
            ...
        masks/
            adjacent_benign/
                slide_001.png      <- same filename as its image
            stroma/
                slide_002.png
            ...

Mask pixel values are expected to be 0 (background/unlabeled) and 1..num_classes for
the actual classes -- the same convention your class-conditional dataset.py used
(`label = pixel_value - 1`). If your masks use a different convention, adjust
mask_to_onehot() below; it's the only place that needs to change.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_pairs(data_dir, images_subdir="images", masks_subdir="masks"):
    """Return a list of (image_path, mask_path) pairs matched by filename stem within
    class subdirectories. Unchanged from the class-conditional dataset.py -- the file
    layout doesn't need to change for the switch to spatial conditioning, only how the
    mask is turned into a tensor once loaded (see mask_to_onehot / __getitem__ below)."""
    data_dir = Path(data_dir)
    img_dir = data_dir / images_subdir
    mask_dir = data_dir / masks_subdir

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Expected an images folder at: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected a masks folder at: {mask_dir}")

    pairs, missing = [], []
    for class_dir in sorted(img_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        mask_class_dir = mask_dir / class_name
        if not mask_class_dir.is_dir():
            print(f"[dataset] Warning: Mask subdirectory '{mask_class_dir}' not found. "
                  f"Skipping class '{class_name}'.")
            continue

        mask_lookup = {p.stem: p for p in mask_class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_lookup.get(p.stem)
            if mask_path is None:
                missing.append(f"{class_name}/{p.name}")
                continue
            pairs.append((p, mask_path))

    if missing:
        print(f"[dataset] Warning: {len(missing)} image(s) had no matching mask and were "
              f"skipped, e.g. {missing[:5]}")
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found under {data_dir}. "
                            f"Check --images_subdir/--masks_subdir, folder structure, and filenames.")
    return pairs


def mask_to_onehot(mask_arr, num_classes, background_label=0):
    """Convert an integer-valued (H, W) mask into a (num_classes, H, W) float32
    one-hot tensor. Pixels equal to background_label get an all-zero column across
    every channel (i.e. "no class" -- the generator sees nothing there to condition
    on); every other pixel value v is expected in [1, num_classes] and lands in
    channel (v - 1)."""
    h, w = mask_arr.shape[:2]
    onehot = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        onehot[c] = (mask_arr == (c + 1)).astype(np.float32)
    return onehot


class SpatialMaskDataset(Dataset):
    """One item = one (image, one-hot mask) pair, both at their native resolution --
    your images are already cropped/selected, so there's no resizing here."""

    def __init__(self, data_dir, num_classes=4, images_subdir="images", masks_subdir="masks",
                 background_label=0):
        self.pairs = find_pairs(data_dir, images_subdir, masks_subdir)
        self.num_classes = num_classes
        self.background_label = background_label
        print(f"[dataset] Found {len(self.pairs)} image/mask pairs across subdirectories.")

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

        onehot = mask_to_onehot(mask_arr, self.num_classes, self.background_label)
        mask_t = torch.from_numpy(onehot)  # (num_classes, H, W)

        image_t = torch.from_numpy(image_arr.astype(np.float32) / 127.5 - 1.0)
        image_t = image_t.permute(2, 0, 1).contiguous()  # (3, H, W), scaled to [-1, 1]

        return image_t, mask_t
