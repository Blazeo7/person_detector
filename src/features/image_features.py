import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from scipy.ndimage import rotate


class ImageProcessor:
    @staticmethod
    def augment(img):
        # Add noise (50% chance)
        if np.random.random() < 0.5:
            img = img + np.random.normal(0, 0.05, img.shape)

        # Rotate (50% chance)
        if np.random.random() < 0.5:
            img = rotate(img, np.random.uniform(-15, 15), order=1, mode="constant", reshape=False)

        # Horizontal flip (60% chance)
        if np.random.random() < 0.6:
            img = np.fliplr(img)

        # Clip at the very end to ensure valid image bounds
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def extract_hog(img, cfg):
        return hog(
            img,
            orientations=cfg.orientations,
            pixels_per_cell=cfg.pixels_per_cell,
            cells_per_block=cfg.cells_per_block,
            feature_vector=True,
        )

    @staticmethod
    def extract_lbp(img, cfg):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lbp = local_binary_pattern(img, cfg.n_points, cfg.radius, cfg.method)
        n_bins = (cfg.n_points + 2) if cfg.method == "uniform" else (2**cfg.n_points)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist

    def process(self, img, augment=False, feature_cfg=None):
        if img.ndim == 3:
            img = rgb2gray(img)
        if augment:
            img = self.augment(img)

        if feature_cfg is None or feature_cfg.type == "raw":
            return img.ravel()

        feats = []
        if "hog" in feature_cfg.type:
            feats.append(self.extract_hog(img, feature_cfg.hog))
        if "lbp" in feature_cfg.type:
            feats.append(self.extract_lbp(img, feature_cfg.lbp))

        return np.concatenate(feats) if feats else img.ravel()
