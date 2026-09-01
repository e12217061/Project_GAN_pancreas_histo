"""
dataset.py — loads a 512x512 (or whatever size your data already is) image from a
folder and reads its class label directly off the name of the subfolder it lives in.
No cropping, no resizing: your images are already the right size, and the label
comes purely from which class folder the image is filed under.

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

Optional per-item processing (all off by default except augment):
- augment: random dihedral (D4) augmentation -- one of the four 90-degree rotations,
  each optionally mirrored horizontally/vertically. No interpolation is involved (exact
  pixel remapping), which matters for tissue patches: there's no canonical orientation,
  but arbitrary-angle rotation would introduce blur/border artifacts that 90-degree
  steps avoid. Assumes square patches -- __init__ warns if they aren't.
- stain_normalize: Macenko (default) or Vahadane stain-color normalization, see
  stain_norm.py. Every image is remapped onto one reference stain appearance, fit
  either from --stain_target_image or (if that's not given) from the first image
  found in the dataset.

Class imbalance (e.g. far more "healthy" than "tumor" patches) isn't handled in here
-- see train.py's --no_weighted_sampler / WeightedRandomSampler, which operates on
whichever images this file discovers rather than duplicating any data on disk.

Random selection itself is handled by PyTorch: train.py creates the DataLoader with
shuffle=True (or a WeightedRandomSampler), which already draws a random image each
step -- no custom randomness needed in here beyond the augmentation itself.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from stain_norm import build_stain_normalizer

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

    def __init__(self, data_dir, images_subdir="images", class_names=None,
                 augment=True, stain_normalize=False, stain_method="macenko",
                 stain_target_image=None):
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
        # Flat list of int labels, same order as self.items -- handy for anything
        # that needs the full label distribution without touching disk again (e.g.
        # train.py's WeightedRandomSampler).
        self.labels = [self.class_to_idx[c] for _, c in self.items]

        counts = {name: self.labels.count(idx) for name, idx in self.class_to_idx.items()}
        print(f"[dataset] Found {len(self.items)} images across {self.num_classes} "
              f"class folder(s): {self.class_to_idx} (counts: {counts})")

        self.augment = augment
        if self.augment:
            with Image.open(self.items[0][0]) as sample_img:
                w, h = sample_img.size
            if w != h:
                print(f"[dataset] Warning: images aren't square ({w}x{h}); 90-degree "
                      f"rotation augmentation will swap width/height per-sample, which "
                      f"most training loops don't expect. Pass augment=False if your "
                      f"patches aren't square.")

        self.stain_normalizer = None
        if stain_normalize:
            if stain_target_image is not None:
                with Image.open(stain_target_image) as target_img:
                    target_arr = np.array(target_img.convert("RGB"))
            else:
                target_path = self.items[0][0]
                with Image.open(target_path) as target_img:
                    target_arr = np.array(target_img.convert("RGB"))
                print(f"[dataset] No stain_target_image given; using {target_path} as "
                      f"the stain-normalization reference.")
            self.stain_normalizer = build_stain_normalizer(stain_method, target_arr)
            print(f"[dataset] Stain normalization enabled ({stain_method}).")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _augment(image_arr):
        """Random dihedral (D4) augmentation: one of the four 90-degree rotations,
        each optionally mirrored. Exact pixel remapping, no interpolation."""
        k = np.random.randint(0, 4)
        image_arr = np.rot90(image_arr, k)
        if np.random.rand() < 0.5:
            image_arr = np.fliplr(image_arr)
        if np.random.rand() < 0.5:
            image_arr = np.flipud(image_arr)
        return np.ascontiguousarray(image_arr)

    def __getitem__(self, idx):
        img_path, class_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image_arr = np.array(image)

        if self.stain_normalizer is not None:
            try:
                image_arr = self.stain_normalizer.normalize(image_arr)
            except Exception as e:
                print(f"[dataset] Warning: stain normalization failed for "
                      f"{img_path.name} ({e}); using the original image instead.")

        if self.augment:
            image_arr = self._augment(image_arr)

        label = self.class_to_idx[class_name]

        image_t = torch.from_numpy(image_arr.astype(np.float32) / 127.5 - 1.0)
        image_t = image_t.permute(2, 0, 1).contiguous()  # CHW, scaled to [-1, 1]
        return image_t, label
