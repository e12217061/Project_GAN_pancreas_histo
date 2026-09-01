"""
stain_norm.py — stain-color normalization for H&E histology patches.

H&E slides vary a lot in color from scanner to scanner and lab to lab (different
stain batches, protocols, scanner color profiles). Stain normalization remaps every
patch's stain colors onto one fixed reference appearance, so the GAN spends its
capacity modeling tissue structure rather than partly modeling stain-color noise
that has nothing to do with the (healthy/tumor) class.

Two methods are offered, matching what train.py exposes via --stain_method:

- "macenko" (default): Macenko et al., 'A method for normalizing histology slides
  for quantitative analysis', ISBI 2009. Implemented here from scratch in numpy --
  no extra dependency beyond what this project already needs.

- "vahadane": Vahadane et al., 'Structure-Preserving Color Normalization and Sparse
  Stain Separation for Histological Images', IEEE TMI 2016. This relies on sparse
  non-negative matrix factorization, which we don't reimplement here -- it's
  delegated to the `staintools` package (which in turn needs `spams`, a compiled
  dependency). Install both if you want it:
      pip install staintools spams
  If that install is a hassle in your environment, "macenko" needs nothing extra
  and is a reasonable default.
"""
import numpy as np


class MacenkoNormalizer:
    """Normalizes an RGB H&E image's stain colors to a reference image's stain
    appearance, following Macenko et al. (ISBI 2009)."""

    def __init__(self, od_threshold=0.15, angle_percentile=1.0):
        # od_threshold: optical-density threshold ("beta" in the paper) below which a
        # pixel is treated as background/no-stain and excluded from the stain fit.
        # angle_percentile: percentile ("alpha") used to pick the two extreme stain
        # directions -- 1.0 means the 1st/99th percentile, the paper's standard,
        # robust choice (avoids single outlier pixels dominating the fit).
        self.od_threshold = od_threshold
        self.angle_percentile = angle_percentile
        self.target_stain_matrix = None  # (2, 3): rows are unit stain vectors (H, E)
        self.target_max_c = None         # (2,): 99th-percentile concentration per stain

    @staticmethod
    def _rgb_to_od(rgb):
        """RGB uint8 [0,255] -> optical density (Beer-Lambert)."""
        rgb = np.clip(rgb.astype(np.float64), 1.0, 255.0)  # avoid log(0)
        return -np.log10(rgb / 255.0)

    def _estimate_stain_matrix(self, od_tissue):
        """od_tissue: (N, 3) OD values for tissue-only pixels. Returns a (2, 3) stain
        matrix whose rows are the unit Hematoxylin and Eosin vectors."""
        cov = np.cov(od_tissue.T)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
        top2 = eigvecs[:, -2:]  # the 2 eigenvectors with the largest eigenvalues

        proj = od_tissue @ top2
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        min_angle = np.percentile(angles, self.angle_percentile)
        max_angle = np.percentile(angles, 100 - self.angle_percentile)
        v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # Order as (Hematoxylin, Eosin) using the standard convention: the vector
        # with the larger red-channel OD is Eosin (it's the pinker/redder stain).
        if v_min[0] > v_max[0]:
            stain_matrix = np.stack([v_min, v_max])
        else:
            stain_matrix = np.stack([v_max, v_min])

        norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Degenerate stain vector (zero norm) -- image likely has "
                             "too little color variation to fit a stain matrix.")
        return stain_matrix / norms

    @staticmethod
    def _concentrations(od_pixels, stain_matrix):
        """Solve od_pixels.T ≈ stain_matrix.T @ C for C. Returns (N, 2)."""
        c, *_ = np.linalg.lstsq(stain_matrix.T, od_pixels.T, rcond=None)
        return c.T

    def fit(self, target_rgb):
        """Fit the reference stain appearance from an RGB (H,W,3) uint8 image."""
        od = self._rgb_to_od(np.asarray(target_rgb)).reshape(-1, 3)
        tissue = od[(od > self.od_threshold).any(axis=1)]
        if tissue.shape[0] < 10:
            raise ValueError("Target/reference image doesn't have enough non-background "
                             "pixels to fit a reference stain matrix -- pick a patch "
                             "with more visible tissue.")
        self.target_stain_matrix = self._estimate_stain_matrix(tissue)
        concentrations = self._concentrations(tissue, self.target_stain_matrix)
        max_c = np.percentile(concentrations, 99, axis=0)
        max_c[max_c == 0] = 1.0
        self.target_max_c = max_c
        return self

    def normalize(self, image_rgb):
        """Normalize an RGB (H,W,3) uint8 image to the fitted target's stain appearance."""
        if self.target_stain_matrix is None:
            raise RuntimeError("Call fit(target_image) before normalize().")

        image_rgb = np.asarray(image_rgb)
        h, w = image_rgb.shape[:2]
        od = self._rgb_to_od(image_rgb).reshape(-1, 3)
        tissue_mask = (od > self.od_threshold).any(axis=1)
        tissue = od[tissue_mask]
        if tissue.shape[0] < 10:
            # No real tissue in this patch (e.g. an all-background crop) -- leave it
            # as-is rather than fitting a stain matrix to background noise.
            return image_rgb.astype(np.uint8)

        source_stain_matrix = self._estimate_stain_matrix(tissue)
        concentrations = self._concentrations(od, source_stain_matrix)  # (H*W, 2)
        max_c_source = np.percentile(concentrations[tissue_mask], 99, axis=0)
        max_c_source[max_c_source == 0] = 1.0
        concentrations = concentrations * (self.target_max_c / max_c_source)

        od_norm = concentrations @ self.target_stain_matrix
        rgb_norm = 255.0 * np.power(10.0, -od_norm)
        rgb_norm = np.clip(rgb_norm, 0, 255).astype(np.uint8)
        return rgb_norm.reshape(h, w, 3)


class VahadaneNormalizer:
    """Thin wrapper around staintools' Vahadane normalizer (Vahadane et al., 2016).
    Needs `pip install staintools spams`. If that's not installed, use
    method="macenko" instead -- no extra dependencies there."""

    def __init__(self):
        try:
            import staintools
        except ImportError as e:
            raise ImportError(
                "method='vahadane' needs the `staintools` package (and its `spams` "
                "dependency): pip install staintools spams. Or use method='macenko', "
                "which needs nothing beyond numpy."
            ) from e
        self._staintools = staintools
        self._normalizer = staintools.StainNormalizer(method="vahadane")

    def fit(self, target_rgb):
        target_rgb = self._staintools.LuminosityStandardizer.standardize(
            np.asarray(target_rgb, dtype=np.uint8))
        self._normalizer.fit(target_rgb)
        return self

    def normalize(self, image_rgb):
        image_rgb = self._staintools.LuminosityStandardizer.standardize(
            np.asarray(image_rgb, dtype=np.uint8))
        return self._normalizer.transform(image_rgb)


def build_stain_normalizer(method, target_rgb):
    """Build and fit a stain normalizer of the given method against a reference image."""
    method = method.lower()
    if method == "macenko":
        normalizer = MacenkoNormalizer()
    elif method == "vahadane":
        normalizer = VahadaneNormalizer()
    else:
        raise ValueError(f"Unknown stain normalization method '{method}'. "
                         f"Choose 'macenko' or 'vahadane'.")
    normalizer.fit(target_rgb)
    return normalizer
