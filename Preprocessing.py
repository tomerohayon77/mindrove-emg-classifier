import os
import pandas as pd
import numpy as np
from FIltering import apply_filters

def segment_emg_signals(emg_signals, labels,  window_size, overlap, fs):
    """
    Segment EMG signals into time windows.

    Parameters:
    emg_signals (numpy.ndarray): The normalized EMG signals.
    lables (numpy.ndarray): The labels of the EMG signals data.
    window_size[ms] (int): The size of the window in milliseconds. Should be between 200 and 500 ms.
    overlap (float): The overlap between windows as a percentage (0 to 1).
    fs (int): The sampling frequency.

    Returns:
    Array: A array of segmented EMG signals.
    """
    num_samples_per_window = round(int(window_size * fs / 1000))
    segmented_emg = []
    segment_labels = []
    step_size = window_size - overlap*window_size * fs / 1000

    for start in range(0, len(emg_signals) - num_samples_per_window , step_size):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        mid_point = start + window_size // 2
        segment_labels.append(labels[mid_point])

    return np.array(segmented_emg), np.array(segment_labels)

def save_mean_std_to_csv(file_path, mean, std, csv_file='mean_std.csv'):
    """Save the mean and std of EMG signals to a CSV file."""
    file_name = file_path.split('\\')[-1]

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        columns = ['file_name'] + [f'mean_{i}' for i in range(mean.shape[1])] + [f'std_{i}' for i in range(std.shape[1])]
        df = pd.DataFrame(columns=columns)

    if file_name not in df['file_name'].values:
        new_row = {'file_name': file_name}
        new_row.update({f'mean_{i}': mean[0, i] for i in range(mean.shape[1])})
        new_row.update({f'std_{i}': std[0, i] for i in range(std.shape[1])})
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        df.to_csv(csv_file, index=False)

def normalize_signals(emg_signals):
    """Normalize EMG signals to have zero mean and unit variance, and return the mean and std."""
    mean = np.mean(emg_signals, axis=0, keepdims=True)
    std = np.std(emg_signals, axis=0, keepdims=True)
    normalized_signals = (emg_signals - mean) / std
    return normalized_signals, mean, std

if __name__ == "__main__":
    # Load the recorded data
    fs = 500  # Sampling frequency in Hz
    file_path = r'Record\recorded_data.csv'
    data = pd.read_csv(file_path)
    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = data[[f'EMG_{i}' for i in range(8)]].values
    gy_signals = data[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    acc_signals = data[['Acc_X', 'Acc_Y', 'Acc_Z']].values

    emg_titles = [f'EMG Channel {i + 1}' for i in range(emg_signals.shape[1])]
    gy_titles = [f'Gyroscope Channel {i + 1}' for i in range(gy_signals.shape[1])]
    acc_titles = [f'Accelerometer Channel {i + 1}' for i in range(acc_signals.shape[1])]
    vbat = data['VBAT'].values

    # Apply filters to EMG signals
    filtered_Emg = apply_filters(emg_signals, fs)[200:]
    normalized_emg, mean, std = normalize_signals(filtered_Emg)
    save_mean_std_to_csv(file_path, mean, std)
    segmented_normalized_emg = segment_emg_signals(normalized_emg, window_size=300, fs=fs)
