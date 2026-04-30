import numpy as np
import scipy.fftpack
from scipy.ndimage import rotate, shift
import librosa


def mel_inv(x):
    return (np.exp(x / 1127.0) - 1.0) * 700.0


def mel(x):
    return 1127.0 * np.log(1.0 + x / 700.0)


def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    if not fend:
        fend = 0.5 * fs

    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((nfft // 2 + 1, nbands))
    for ii in range(nbands):
        mfb[cbin[ii] : cbin[ii + 1] + 1, ii] = np.linspace(0.0, 1.0, cbin[ii + 1] - cbin[ii] + 1)
        mfb[cbin[ii + 1] : cbin[ii + 2] + 1, ii] = np.linspace(1.0, 0.0, cbin[ii + 2] - cbin[ii + 1] + 1)
    return mfb


def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window):
        window = np.hamming(window)
    if noverlap is None:
        noverlap = window.size // 2
    if nfft is None:
        nfft = window.size

    x = framing(x, window.size, window.size - noverlap)
    x = scipy.fftpack.fft(x * window, nfft)
    return x[:, : x.shape[1] // 2 + 1]


def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    # Add low level noise (40dB SNR) to avoid log of zeros
    snrdb = 40
    noise = np.random.rand(s.shape[0])
    s = s + noise.dot(np.linalg.norm(s, 2)) / np.linalg.norm(noise, 2) / (10 ** (snrdb / 20))

    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm="ortho")

    S = spectrogram(s, window, noverlap, nfft)
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T


def augment_audio(signal, fs):
    """Randomly apply an augmentation to the audio signal."""
    # Choose a random augmentation type (or none)
    aug_type = np.random.choice(["noise", "speed", "none"], p=[0.4, 0.4, 0.2])

    if aug_type == "noise":
        # Add random Gaussian noise (simulate different mic qualities)
        noise = np.random.randn(len(signal))
        snrdb = np.random.uniform(10, 30)  # Random SNR between 10dB and 30dB
        signal = signal + noise * (np.linalg.norm(signal) / np.linalg.norm(noise)) / (10 ** (snrdb / 20))

    elif aug_type == "speed":
        # Speed up or slow down speech without changing pitch
        rate = np.random.uniform(0.85, 1.15)
        signal = librosa.effects.time_stretch(y=signal, rate=rate)

    return signal


def augment_image(img):
    """Randomly apply an augmentation to the 2D image."""
    aug_type = np.random.choice(["noise", "rotate", "shift", "none"], p=[0.3, 0.3, 0.3, 0.1])

    if aug_type == "noise":
        # Add Gaussian noise
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255)

    elif aug_type == "rotate":
        # Rotate between -15 and 15 degrees
        angle = np.random.uniform(-15, 15)
        img = rotate(img, angle, reshape=False, mode="nearest")

    elif aug_type == "shift":
        # Translate randomly by up to 5 pixels on both axes
        shift_val = np.random.uniform(-5, 5, 2)
        img = shift(img, shift_val, mode="nearest")

    return img


def process_audio(signal, fs, augment=False):
    """
    Complete audio pipeline: Augmentation -> Feature Extraction
    Returns 2D array of MFCC features.
    """
    if augment:
        signal = augment_audio(signal, fs)

    # Standard baseline parameters from 16kHz files
    # mfcc(s, window, noverlap, nfft, fs, nbanks, nceps)
    features = mfcc(signal, 400, 240, 512, fs, 23, 13)
    return features


def process_image(img, augment=False):
    """
    Complete image pipeline: Augmentation -> Flattening/Normalization
    Returns 1D array of image features.
    """
    if augment:
        img = augment_image(img)

    # Flatten the image (as baseline does) and normalize to [0, 1] for better stability
    # especially helpful for linear classifiers and neural networks
    features = img.ravel() / 255.0
    return features
