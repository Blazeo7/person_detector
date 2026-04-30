import os
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from enum import Enum

import numpy as np
from scipy.io import wavfile
from matplotlib.pyplot import imread


class Modality(Enum):
    IMAGE = 1
    AUDIO = 2


class Label(Enum):
    NON_TARGET = 1
    TARGET = 2


@dataclass
class Sample:
    path: str
    name: str  # filename without extension
    label: Label
    person_id: str  # e.g. 'f401'
    session_id: str  # e.g. '01'  — used for CV splitting
    modality: Modality  # 'image' or 'audio'


def parse_filename(fname: str) -> tuple[str, str]:
    """Extract (person_id, session_id) from filename stem like f401_01_f21_i0_0."""
    parts = Path(fname).stem.split("_")
    person_id = parts[0] if len(parts) > 0 else "unknown"
    session_id = parts[1] if len(parts) > 1 else "00"
    return person_id, session_id


def load_samples(target_dirs: list[str], non_target_dirs: list[str], modality: Modality) -> list[Sample]:
    """
    Load all samples from the given directories.
    Returns a list of Sample objects (no data loaded yet — lazy).
    """
    ext = "*.png" if modality == Modality.IMAGE else "*.wav"
    samples = []

    for directory in target_dirs:
        for path in sorted(glob(os.path.join(directory, ext))):
            person_id, session_id = parse_filename(path)
            samples.append(
                Sample(
                    path=path,
                    name=Path(path).stem,
                    label=Label.TARGET,
                    person_id=person_id,
                    session_id=session_id,
                    modality=modality,
                )
            )

    for directory in non_target_dirs:
        for path in sorted(glob(os.path.join(directory, ext))):
            person_id, session_id = parse_filename(path)
            samples.append(
                Sample(
                    path=path,
                    name=Path(path).stem,
                    label=Label.NON_TARGET,
                    person_id=person_id,
                    session_id=session_id,
                    modality=modality,
                )
            )

    return samples


def load_image(path: str) -> np.ndarray:
    """Load PNG as float64 grayscale array (H, W)."""
    img = imread(path)
    if img.ndim == 3:
        # Convert RGB to grayscale by summing channels (matches baseline)
        img = img.sum(axis=2)
    return img.astype(np.float64)


def load_audio(path: str) -> tuple[int, np.ndarray]:
    """Load WAV, return (sample_rate, signal_float)."""
    rate, sig = wavfile.read(path)
    sig = sig.astype(np.float64)
    return rate, sig


def load_all_images(samples: list[Sample]) -> np.ndarray:
    """Load all image samples into an (N, H*W) matrix."""
    arrays = []
    for s in samples:
        img = s.path
        arrays.append(img.ravel())
    return np.array(arrays)


def load_labels(samples: list[Sample]) -> np.ndarray:
    return np.array([s.label for s in samples])
