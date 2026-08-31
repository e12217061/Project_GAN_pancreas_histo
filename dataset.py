"""
dataset.py — loads a 512x512 (or whatever size your data already is) image from a
folder and reads its class label directly off the name of the subfolder it lives in.
No cropping, no resizing: your images are already the right size, and (for now) the
label comes purely from which class folder the image is filed under -- masks are not
read or used at all here.

EXPECTED FOLDER LAYOUT:

    <data_dir>/
        images/
            healthy/
                slide_001.png
            tumor/
                slide_002.png

Every image under images/<class_name>/ gets label = index of <class_name> in the
sorted list of class folder names found on disk (so with just "healthy" and "tumor",
alphabetical order gives healthy -> 0, tumor -> 1). Pass class_names explicitly if you
want to pin down the order yourself instead of relying on alphabetical sort, or to
restrict/validate against a known set of classes.

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


def find_images(data_dir, images_subdir="images"):
    """Return a list of (image_path, class_name) pairs, one per image, with the class
    name taken directly from the image's parent subdirectory, e.g.
    images/healthy/slide_001.png -> class_name "healthy"."""
    data_dir = Path(data_dir)
    img_dir = data_dir / images_subdir

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Expected an images folder at: {img_dir}")

    items = []
    for class_dir in sorted(img_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            items.append((p, class_name))

    if not items:
        raise RuntimeError(f"No images found under {img_dir}. Check --images_subdir "
                           f"and folder structure (expects one subfolder per class).")
    return items


class PatchDataset(Dataset):
    """One item = one (already-cropped) image + a class label taken from its folder."""

    def __init__(self, data_dir, images_subdir="images", class_names=None):
        self.items = find_images(data_dir, images_subdir)

        # Build a stable class_name -> integer label mapping. By default this is just
        # the sorted list of class folder names found on disk; pass class_names to
        # pin down the order yourself (and to catch unexpected folders early).
        discovered = sorted({class_name for _, class_name in self.items})
        if class_names is not None:
            missing = set(discovered) - set(class_names)
            if missing:
                raise ValueError(f"Found class folder(s) {sorted(missing)} that aren't "
                                 f"in the given class_names {list(class_names)}.")
            self.class_names = list(class_names)
        else:
            self.class_names = discovered

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        print(f"[dataset] Found {len(self.items)} images across {self.num_classes} "
              f"class folder(s): {self.class_to_idx}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, class_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image_arr = np.array(image)

        label = self.class_to_idx[class_name]

        image_t = torch.from_numpy(image_arr.astype(np.float32) / 127.5 - 1.0)
        image_t = image_t.permute(2, 0, 1).contiguous()  # CHW, scaled to [-1, 1]
        return image_t, label
