from scipy.signal import butter, lfilter, iirnotch
import pandas as pd
import numpy as np
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

# Function to apply a Butterworth filter
def butter_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

# High-pass filter (for baseline drift removal)
def highpass_filter(filtered_emg, cutoff=20, fs=500, order=4):
    return butter_filter(filtered_emg, cutoff, fs, order, filter_type='high')

# Low-pass filter (for muscle activity band isolation)
def lowpass_filter(filtered_emg, cutoff=225, fs=500, order=4):  # Adjusted cutoff frequency
    return butter_filter(filtered_emg, cutoff, fs, order, filter_type='low')

# Notch filter for powerline interference removal (50Hz or 60Hz)
def notch_filter(filtered_emg, notch_freq=50.0, fs=500, Q=30):
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, Q)
    return lfilter(b, a, filtered_emg)

def apply_filters(emg_signals, fs):
    padded_signals = np.pad(emg_signals, ((200, 0), (0, 0)), mode='edge')  # Pad 200 samples
    filtered_emg_signals = emg_signals.copy()
    for i in range(padded_signals.shape[1]):
        filtered_signal = highpass_filter(padded_signals[:, i], cutoff=20, fs=fs)
        filtered_signal = notch_filter(filtered_signal, notch_freq=50, fs=fs)  # For 50 Hz powerline noise
        filtered_signal = lowpass_filter(filtered_signal, cutoff=fs/2-1, fs=fs)  # Adjusted cutoff frequency
        # Remove padding after filtering
        filtered_emg_signals[:, i] = filtered_signal[200:]
    return filtered_emg_signals
