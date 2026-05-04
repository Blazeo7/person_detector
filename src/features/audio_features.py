import librosa
import numpy as np
from scipy.stats import kurtosis, skew


class AudioProcessor:
    def __init__(self, fs=16000, feature_cfg=None):
        self.fs = fs
        # 25ms window for spectral features (MFCC) = 400 samples
        self.n_fft = int(0.025 * fs)
        # 10ms hop length = 160 samples
        self.hop_length = int(0.010 * fs)
        # 64ms window specifically for pitch tracking = 1024 samples
        self.pitch_frame_length = 1024

        # Fallback to all True if no config is passed
        self.cfg = feature_cfg or {
            "mfcc": True,
            "n_mfcc": 21,
            "deltas": True,
            "spectral": True,
            "pitch": True,
        }

    def augment(self, signal):
        # Additive noise (60% chance)
        if np.random.random() < 0.6:
            noise = np.random.randn(len(signal))
            snr_db = np.random.uniform(10, 30)
            noise_scale = np.linalg.norm(signal) / (np.linalg.norm(noise) * (10 ** (snr_db / 20)))
            signal = signal + noise * noise_scale

        # Time stretch (50% chance)
        if np.random.random() < 0.5:
            signal = librosa.effects.time_stretch(y=signal, rate=np.random.uniform(0.85, 1.15))

        # Pitch shift (40% chance)
        if np.random.random() < 0.4:
            signal = librosa.effects.pitch_shift(y=signal, sr=self.fs, n_steps=np.random.uniform(-2, 2))

        # Random gain (50% chance)
        if np.random.random() < 0.5:
            gain_db = np.random.uniform(-6, 6)
            signal = signal * (10 ** (gain_db / 20))

        return signal

    def _extract_frame_features(self, signal):
        """Extracts matrix of features (rows=features, cols=frames)."""
        feats = []

        if self.cfg.get("mfcc", True):
            n_mfcc = self.cfg.get(13)
            mfcc_full = librosa.feature.mfcc(
                y=signal,
                sr=self.fs,
                n_mfcc=n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window="hamming",
            )
            # Drop MFCC-0
            mfcc = mfcc_full[1:, :]
            feats.append(mfcc)

            if self.cfg.get("deltas", True):
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                feats.extend([delta, delta2])

        if self.cfg.get("spectral", True):
            rms = librosa.feature.rms(y=signal, frame_length=self.n_fft, hop_length=self.hop_length)
            zcr = librosa.feature.zero_crossing_rate(signal, hop_length=self.hop_length)
            flat = librosa.feature.spectral_flatness(y=signal, hop_length=self.hop_length)
            centroid = librosa.feature.spectral_centroid(
                y=signal, sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length
            )
            feats.extend([rms, zcr, flat, centroid])

        if not feats:
            raise ValueError("No frame-level audio features selected in config!")

        return np.vstack(feats)

    def process(self, signal, augment=False):
        intervals = librosa.effects.split(signal, top_db=30)
        if len(intervals) > 0:
            signal = np.concatenate([signal[start:end] for start, end in intervals])

        if augment:
            signal = self.augment(signal)

        # Lower preemphasis coefficient for speaker tasks
        signal = librosa.effects.preemphasis(signal, coef=0.95)

        # Frame-level features
        frame_feats = self._extract_frame_features(signal)

        stats = np.concatenate(
            [
                np.mean(frame_feats, axis=1),
                np.std(frame_feats, axis=1),
                skew(frame_feats, axis=1),
                kurtosis(frame_feats, axis=1),
            ]
        )

        final_vector = [stats]

        # Extract Pitch (f0) globally if enabled
        if self.cfg.get("pitch", True):
            f0 = librosa.yin(
                signal,
                fmin=80,
                fmax=600,
                sr=self.fs,
                frame_length=self.pitch_frame_length,
                hop_length=self.hop_length,
            )

            # Filter out unvoiced frames
            f0 = np.nan_to_num(f0)
            voiced_f0 = f0[f0 > 0]

            if len(voiced_f0) > 0:
                f0_stats = [np.mean(voiced_f0), np.std(voiced_f0), skew(voiced_f0), kurtosis(voiced_f0)]
            else:
                f0_stats = [0, 0, 0, 0]  # Fallback

            final_vector.append(f0_stats)

        return np.concatenate(final_vector)
