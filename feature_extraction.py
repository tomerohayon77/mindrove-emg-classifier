import numpy as np
from scipy.fftpack import fft

# Time-Domain Features
def mav_feature(emg_signal):
    """Mean Absolute Value (MAV)"""
    return np.mean(np.abs(emg_signal))


def rms_feature(emg_signal):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(emg_signal ** 2))


def wl_feature(emg_signal):
    """Waveform Length (WL)"""
    return np.sum(np.abs(np.diff(emg_signal)))


def zc_feature(emg_signal, threshold=0):
    """Zero Crossing (ZC)"""
    return np.sum(np.diff(np.sign(emg_signal)) != 0)


def ssc_feature(emg_signal, threshold=0):
    """Slope Sign Changes (SSC)"""
    return np.sum(np.diff(np.sign(np.diff(emg_signal))) != 0)


# Frequency-Domain Features
def fft_spectrum(emg_signal, fs):
    """Calculate FFT and return frequency and amplitude"""
    N = len(emg_signal)
    f_values = np.fft.fftfreq(N, 1 / fs)
    f_values = f_values[:N // 2]  # Only return positive frequencies
    fft_values = np.abs(fft(emg_signal))[:N // 2]
    return f_values, fft_values


def mnf_feature(emg_signal, fs):
    """Mean Frequency (MNF)"""
    f_values, fft_values = fft_spectrum(emg_signal, fs)
    return np.sum(f_values * fft_values) / np.sum(fft_values)


def mdf_feature(emg_signal, fs):
    """Median Frequency (MDF)"""
    f_values, fft_values = fft_spectrum(emg_signal, fs)
    cumulative_sum = np.cumsum(fft_values)
    median_freq = f_values[np.where(cumulative_sum >= cumulative_sum[-1] / 2)[0][0]]
    return median_freq

# Full Feature Extraction Pipeline
def extract_features(emg_signal, fs):

    features = {
        'MAV': mav_feature(emg_signal),
        'RMS': rms_feature(emg_signal),
        'WL': wl_feature(emg_signal),
        'ZC': zc_feature(emg_signal),
        'SSC': ssc_feature(emg_signal)
    }

    # Step 3: Frequency-Domain Features
    features['MNF'] = mnf_feature(emg_signal, fs)
    features['MDF'] = mdf_feature(emg_signal, fs)

    return features


