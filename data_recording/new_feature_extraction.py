import numpy as np
from scipy.fftpack import fft
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import cwt, ricker

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
    return (np.sum(np.diff(np.sign(emg_signal)) != 0)) / emg_signal.shape[0]
def ssc_feature(emg_signal, threshold=0):
    """Slope Sign Changes (SSC)"""
    return np.sum(np.diff(np.sign(np.diff(emg_signal))) != 0)/emg_signal.shape[0]

def willison_amplitude(signal, threshold=0.01):
    """Calculate Willison Amplitude (WAMP) of an EMG signal with a threshold."""
    return np.sum(np.abs(np.diff(signal)) > threshold)

def integrated_emg(signal):
    """Calculate Integrated EMG (IEMG) of an EMG signal."""
    return np.sum(np.abs(signal))

def signal_entropy(signal, num_bins=10):
    """Calculate Entropy of an EMG signal."""
    hist, _ = np.histogram(signal, bins=num_bins, density=True)
    return entropy(hist)

def var_feature(emg_signal):
    """Variance (VAR)"""
    return np.var(emg_signal)

def skewness_feature(emg_signal):
    """Skewness"""
    return skew(emg_signal)

def kurtosis_feature(emg_signal):
    """Kurtosis"""
    return kurtosis(emg_signal)

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

def wavelet_energy(emg_signal, widths=np.arange(1, 31)):
    """Wavelet Energy"""
    cwt_matrix = cwt(emg_signal, ricker, widths)
    return np.sum(cwt_matrix ** 2)

def frequency_variance(emg_signal, fs):
    """Frequency Variance"""
    f_values, fft_values = fft_spectrum(emg_signal, fs)
    return np.var(fft_values)

def power_spectrum_ratio(emg_signal, fs, low_freq=20, high_freq=450):
    """Power Spectrum Ratio"""
    f_values, fft_values = fft_spectrum(emg_signal, fs)
    total_power = np.sum(fft_values)
    band_power = np.sum(fft_values[(f_values >= low_freq) & (f_values <= high_freq)])
    return band_power / total_power

def hjorth_parameters(signal):
    """Calculate Hjorth parameters (Activity, Mobility, Complexity) of an EMG signal."""
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal)))
    return activity, mobility, complexity

def extract_wavelet_features_cwt(emg_signal, widths=np.arange(1, 31)):
    """
    Extract continuous wavelet transform (CWT) features from an EMG signal.

    Parameters:
    - emg_signal: 1D array-like, the EMG signal data.
    - widths: 1D array, the range of widths for the wavelet (default: 1 to 30).

    Returns:
    - features: 2D numpy array of wavelet coefficients.
    """
    cwt_matrix = cwt(emg_signal, ricker, widths)
    features = cwt_matrix.flatten()
    return features

# Full Feature Extraction Pipeline
def extract_features(emg_signal, fs):
    abs_emg_signal = np.abs(emg_signal)
    features = {}
    num_channels = emg_signal.shape[0]

    for channel in range(num_channels):
        channel_signal = emg_signal[channel, :]
        abs_channel_signal = abs_emg_signal[channel, :]
        activity, mobility, complexity = hjorth_parameters(channel_signal)
        features.update({
            f'MAV_CH{channel + 1}': mav_feature(abs_channel_signal),  # Time-Domain
            f'RMS_CH{channel + 1}': rms_feature(channel_signal),  # Time-Domain
            f'WL_CH{channel + 1}': wl_feature(abs_channel_signal),  # Time-Domain
            f'ZC_CH{channel + 1}': zc_feature(channel_signal),  # Time-Domain
            f'SSC_CH{channel + 1}': ssc_feature(channel_signal),  # Time-Domain
            #f'MNF_CH{channel + 1}': mnf_feature(channel_signal, fs),  # Frequency-Domain
            #f'MDF_CH{channel + 1}': mdf_feature(channel_signal, fs),  # Frequency-Domain
            #f'HJP_Activity_CH{channel + 1}': activity,  # Time-Domain
            #f'HJP_Mobility_CH{channel + 1}': mobility,  # Time-Domain
            #f'HJP_Complexity_CH{channel + 1}': complexity,  # Time-Domain
            #f'ENT_CH{channel + 1}': signal_entropy(channel_signal),  # Time-Domain
            #f'CWT_CH{channel + 1}': extract_wavelet_features_cwt(channel_signal),  # Frequency-Domain
            #f'VAR_CH{channel + 1}': var_feature(channel_signal),  # Time-Domain
            #f'Skewness_CH{channel + 1}': skewness_feature(channel_signal),  # Time-Domain
            #f'Kurtosis_CH{channel + 1}': kurtosis_feature(channel_signal),  # Time-Domain
            #f'Wavelet_Energy_CH{channel + 1}': wavelet_energy(channel_signal),  # Frequency-Domain
            #f'Frequency_Variance_CH{channel + 1}': frequency_variance(channel_signal, fs),  # Frequency-Domain
            #f'Power_Spectrum_Ratio_CH{channel + 1}': power_spectrum_ratio(channel_signal, fs)  # Frequency-Domain
        })
    return features