import os
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
from matplotlib.pyplot import imread
import librosa
from skimage.color import rgb2gray


@dataclass
class Sample:
    name: str
    label: int
    person_id: str
    session_id: str


class DataLoader:
    """Unified dataloader for dataset discovery, image processing, and audio processing."""

    def __init__(self):
        self.samples: list[Sample] = []
        self.images = np.array([])
        self.audios = []

    @staticmethod
    def parse_filename(fname: str) -> tuple[str, str]:
        """Extract (person_id, session_id) from filename stem."""
        parts = Path(fname).stem.split("_")
        person_id = parts[0] if len(parts) > 0 else "unknown"
        session_id = parts[1] if len(parts) > 1 else "00"
        return person_id, session_id

    def load_data(self, target_dirs: list[str], non_target_dirs: list[str]):
        configs = [(target_dirs, 1), (non_target_dirs, 0)]

        images = []
        audios = []

        for directories, label in configs:
            for directory in directories:
                # Find all stems that have BOTH a .png and .wav file
                png_stems = {Path(p).stem for p in glob(os.path.join(directory, "*.png"))}
                wav_stems = {Path(p).stem for p in glob(os.path.join(directory, "*.wav"))}
                paired_stems = sorted(png_stems & wav_stems)  # intersection, sorted for determinism

                for stem in paired_stems:
                    png_path = os.path.join(directory, stem + ".png")
                    wav_path = os.path.join(directory, stem + ".wav")

                    p_id, s_id = self.parse_filename(stem)
                    self.samples.append(
                        Sample(
                            name=stem,
                            label=label,
                            person_id=p_id,
                            session_id=s_id,
                        )
                    )
                    images.append(self.load_single_image(png_path))
                    _, sig = self.load_single_audio(wav_path)
                    audios.append(sig)

        self.images = np.array(images) if images else np.array([])
        self.audios = audios  # keep as list due to variable length
        return self

    @staticmethod
    def get_labels(samples: list[Sample]) -> np.ndarray:
        """Utility to extract labels from a list of samples."""
        return np.array([s.label for s in samples])

    @staticmethod
    def load_single_image(path: str) -> np.ndarray:
        """Load PNG as float64 array."""
        img = imread(path)

        if img.ndim == 3:
            img = rgb2gray(img)

        return img.astype(np.float64)

    @staticmethod
    def load_single_audio(path: str, target_sr=16000) -> tuple[int, np.ndarray]:
        """Load WAV, resample, and normalize."""
        sig, rate = librosa.load(path, sr=target_sr)
        return rate, sig.astype(np.float64)
