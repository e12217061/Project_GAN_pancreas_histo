# Basic class-conditional patch GAN

A minimal, class-conditional DCGAN-style GAN for generating image patches, sized
dynamically for whatever resolution your data is — including 512x512 (see the
honesty note below).

## Files
- `dataset.py` — matches each image to its same-named mask and reads off a class
  label. No cropping or resizing: your images are already the right size.
- `models.py` — the Generator and Discriminator.
- `train.py` — CLI training script (this is what you run).
- `make_dummy_data.py` — generates a tiny synthetic dataset so you can smoke-test the
  pipeline before pointing it at real data.
- `requirements.txt` — `pip install -r requirements.txt`.

## Assumed folder layout — **please check this first**

```
<data_dir>/
    images/
        slide_001.png
        slide_002.png
        ...
    masks/
        slide_001.png     <- same filename as its image
        slide_002.png
        ...
```

Each mask is a single-channel image where the **pixel value is the class id**
(0, 1, 2, 3 for a 4-class problem). Since your images are already selected/cropped to
be dominated by one class, the label for an image is just the majority pixel value
across its whole mask.

**If your data is organized differently** — e.g. images and masks in one shared
folder, a `_mask` filename suffix, class from a subfolder name instead of pixel
values — edit `find_pairs()` in `dataset.py`. It's ~25 lines and is the only place
that needs to change.

If part of your mask is "unlabeled" rather than one of the classes, pass
`--background_label` (e.g. `--background_label 0`) to exclude those pixels from the
majority vote.

Random selection is handled by the DataLoader (`shuffle=True` in `train.py`) — there's
no custom sampling logic to configure.

## Quick start

```bash
pip install -r requirements.txt

# 1) sanity-check the whole pipeline on synthetic data first
python make_dummy_data.py --out ./dummy_data --num_images 16 --size 512 --num_classes 4
python train.py --data_dir ./dummy_data --epochs 2 --batch_size 4

# 2) once that runs cleanly, point it at your real data
python train.py --data_dir /path/to/your/data --num_classes 4 \
                 --epochs 200 --batch_size 16
```

`--patch_size` defaults to auto-detecting from the first image in your dataset (so if
your crops are 512x512, you don't need to set anything). Pass it explicitly only if
you want to override.

Outputs land in `./gan_outputs/samples/` (a PNG grid: one row per class) and
`./gan_outputs/checkpoints/` (resumable with `--resume path/to/ckpt.pt`).

Run `python train.py --help` for every option, and `python models.py` for a standalone
shape smoke-test of the Generator/Discriminator at several resolutions.

## About reaching 512x512

The architecture builds and runs at 512x512. Being upfront about the tradeoff: a
plain small-scale conditional DCGAN (which is what this is — "very basic" was the
brief) tends to get unstable and mode-collapse-prone at that resolution, no matter
whose implementation it is. Spectral norm is on by default in the discriminator since
it's a cheap, standard way to buy some extra stability. Practical path that tends to
work:
- First confirm the model learns at all by running a batch or two at a smaller size
  (temporarily downsample a few images to 128 for a quick check) if 512 training looks
  unstable, to isolate whether it's a resolution problem or a data/setup problem.
- Watch the sample grids each epoch for collapse (all patches within a class look
  identical) or divergence (pure noise).
- If 512x512 stays unstable, the standard next steps are progressive-growing or a
  StyleGAN-style architecture — meaningfully more complex than "basic," so only worth
  it once you've confirmed the simple version works.

## A note on how this was built

I don't have PyTorch available in the sandbox I write code in, so I could not execute
`train.py` or `models.py` end-to-end myself. What I *did* verify:
- The image/mask matching and label logic in `dataset.py` — tested against real files
  produced by `make_dummy_data.py`.
- The Generator/Discriminator shape math (upsampling block count, channel sizes,
  padding) — traced by hand for sizes 64/100/128/256/512.
- All files pass a Python syntax check.

Please run the dummy-data smoke test above as your first step on your own machine —
it's the fastest way to catch anything my sandbox couldn't.
