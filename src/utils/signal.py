import numpy as np
import scipy.fftpack


def mel_inv(x):
    return (np.exp(x / 1127.0) - 1.0) * 700.0


def mel(x):
    return 1127.0 * np.log(1.0 + x / 700.0)


def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    fend = fend or (0.5 * fs)
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
    noverlap = noverlap if noverlap is not None else window.size // 2
    nfft = nfft or window.size

    frames = framing(x, window.size, window.size - noverlap)
    stft = scipy.fftpack.fft(frames * window, nfft)
    return stft[:, : stft.shape[1] // 2 + 1]
