import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import Iterator
from dataclasses import dataclass
from joblib import Parallel, delayed

from dataset.data_loader import DataLoader, Sample
from features.image_features import ImageProcessor
from features.audio_features import AudioProcessor


@dataclass
class FoldData:
    train_samples: list[Sample]
    val_samples: list[Sample]
    train_images: np.ndarray  # (N_train * (1 + n_aug), feat_dim)
    val_images: np.ndarray  # (N_val, feat_dim)
    train_audios: np.ndarray  # (N_train * (1 + n_aug), feat_dim)
    val_audios: np.ndarray  # (N_val, feat_dim)
    train_labels: np.ndarray  # (N_train * (1 + n_aug),)
    val_labels: np.ndarray  # (N_val,)


def make_session_groups(samples: list[Sample]) -> np.ndarray:
    keys = [f"{s.person_id}_{s.session_id}" for s in samples]
    unique_keys = sorted(set(keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    return np.array([key_to_idx[k] for k in keys])


def session_kfold_splits(
    loader: DataLoader,
    n_splits: int = 5,
    seed: int = 42,
    n_augmentations: int = 2,
    image_processor: ImageProcessor = None,
    audio_processor: AudioProcessor = None,
    feature_cfg=None,
    n_jobs: int = -1,
) -> Iterator[FoldData]:
    image_processor = image_processor or ImageProcessor()
    audio_processor = audio_processor or AudioProcessor()

    samples = loader.samples
    images = loader.images  # (N, H, W, C)
    audios = loader.audios  # list of N arrays

    labels = DataLoader.get_labels(samples)
    groups = make_session_groups(samples)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros((len(samples), 1))

    for train_idx, val_idx in sgkf.split(dummy_X, labels, groups):
        val_imgs, _ = _process_images(images[val_idx], image_processor, 0, feature_cfg)
        val_auds, _ = _process_audios([audios[i] for i in val_idx], audio_processor, 0, n_jobs)

        train_imgs, img_aug_map = _process_images(
            images[train_idx], image_processor, n_augmentations, feature_cfg
        )
        train_auds, aud_aug_map = _process_audios(
            [audios[i] for i in train_idx], audio_processor, n_augmentations, n_jobs
        )

        train_labels = labels[train_idx][img_aug_map]

        yield FoldData(
            train_samples=[samples[train_idx[i]] for i in img_aug_map],
            val_samples=[samples[i] for i in val_idx],
            train_images=train_imgs,
            val_images=val_imgs,
            train_audios=train_auds,
            val_audios=val_auds,
            train_labels=train_labels,
            val_labels=labels[val_idx],
        )


def _process_images(
    images: np.ndarray,
    processor: ImageProcessor,
    n_augmentations: int = 0,
    feature_cfg=None,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(images)

    original_feats = np.array(
        [processor.process(img, augment=False, feature_cfg=feature_cfg) for img in images]
    )

    if n_augmentations == 0:
        return original_feats, np.arange(N)

    aug_batches = [original_feats]
    for _ in range(n_augmentations):
        aug_imgs = _augment_images_batch(images)  # (N, H, W) — already greyscale
        aug_feats = np.array(
            [processor.process(img, augment=False, feature_cfg=feature_cfg) for img in aug_imgs]
        )
        aug_batches.append(aug_feats)

    features = np.vstack(aug_batches)
    aug_map = np.tile(np.arange(N), 1 + n_augmentations)
    return features, aug_map


def _process_audios(
    audios: list,
    processor: AudioProcessor,
    n_augmentations: int = 0,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Audio features are inherently sequential (librosa), so we parallelise
    across samples with joblib instead of trying to vectorise the DSP.
    """
    N = len(audios)

    original_feats = np.array(
        Parallel(n_jobs=n_jobs)(delayed(processor.process)(sig, augment=False) for sig in audios)
    )

    if n_augmentations == 0:
        return original_feats, np.arange(N)

    aug_batches = [original_feats]
    for _ in range(n_augmentations):
        aug_feats = np.array(
            Parallel(n_jobs=n_jobs)(delayed(processor.process)(sig, augment=True) for sig in audios)
        )
        aug_batches.append(aug_feats)

    features = np.vstack(aug_batches)
    aug_map = np.tile(np.arange(N), 1 + n_augmentations)
    return features, aug_map


def _augment_images_batch(images: np.ndarray) -> np.ndarray:
    """
    Apply a random augmentation to every image in the batch simultaneously.
    images: (N, H, W)  float64, already greyscaled
    returns: (N, H, W)
    """
    from scipy.ndimage import rotate as nd_rotate

    N, H, W = images.shape
    out = images.copy()

    choices = np.random.choice(["noise", "rotate", "flip"], size=N, p=[0.3, 0.3, 0.4])

    # --- noise (all noise samples at once) ---
    noise_mask = choices == "noise"
    if noise_mask.any():
        noise = np.random.normal(0, 0.05, (noise_mask.sum(), H, W))
        out[noise_mask] = np.clip(out[noise_mask] + noise, 0.0, 1.0)

    # --- flip (vectorised axis flip) ---
    flip_mask = choices == "flip"
    if flip_mask.any():
        out[flip_mask] = out[flip_mask, :, ::-1]

    # --- rotate (still per-sample but only for the subset that needs it) ---
    rotate_mask = np.where(choices == "rotate")[0]
    if len(rotate_mask):
        angles = np.random.uniform(-15, 15, len(rotate_mask))
        for i, angle in zip(rotate_mask, angles):
            out[i] = nd_rotate(out[i], angle, order=1, mode="constant", reshape=False)

    return np.clip(out, 0.0, 1.0)
