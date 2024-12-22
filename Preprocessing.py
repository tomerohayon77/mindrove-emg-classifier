import os
import pandas as pd
import numpy as np
from FIltering import apply_filters
from feature_extraction import extract_features

def segment_emg_signals(emg_signals, labels, window_size, overlap, fs):
    """
    Segment EMG signals into time windows.

    Parameters:
    emg_signals (numpy.ndarray): The normalized EMG signals.
    labels (numpy.ndarray): The labels of the EMG signals data.
    window_size (int): The size of the window in milliseconds. Should be between 200 and 500 ms.
    overlap (float): The overlap between windows as a percentage (0 to 1).
    fs (int): The sampling frequency.

    Returns:
    numpy.ndarray: An array of segmented EMG signals.
    numpy.ndarray: An array of segment labels.
    """
    num_samples_per_window = round(int(window_size * fs / 1000))
    segmented_emg = []
    segment_labels = []
    step_size = window_size - overlap * window_size * fs / 1000

    for start in range(0, len(emg_signals) - num_samples_per_window, step_size):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        mid_point = start + window_size // 2
        segment_labels.append(labels[mid_point])

    return np.array(segmented_emg), np.array(segment_labels)

def normalize_signals(emg_signals):
    """
    Normalize EMG signals using min-max normalization.

    Parameters:
    emg_signals (numpy.ndarray): The EMG signals to normalize.

    Returns:
    numpy.ndarray: The normalized EMG signals.
    """
    min_val = np.min(emg_signals, axis=0, keepdims=True)
    max_val = np.max(emg_signals, axis=0, keepdims=True)
    normalized_signals = (emg_signals - min_val) / (max_val - min_val)
    return normalized_signals

def process_csv_file(file_path, fs=500):
    # Load the recorded data
    data = pd.read_csv(file_path)

    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = data[[f'EMG_{i}' for i in range(8)]].values
    labels = data['Label'].values

    # Apply filters to EMG signals
    filtered_emg = apply_filters(emg_signals, fs)[200:]

    # Normalize the EMG signals
    normalized_emg = normalize_signals(filtered_emg)

    # Segment the normalized EMG signals into time windows
    segmented_emg, segment_labels = segment_emg_signals(normalized_emg, labels, window_size=300, overlap=0.5, fs=fs)

    # Extract features from each segment
    features_list = []
    for segment in segmented_emg:
        features = extract_features(segment, fs)
        features_list.append(features)

    # Create a DataFrame with features and labels
    features_df = pd.DataFrame(features_list)
    features_df['Label'] = segment_labels

    # Save the DataFrame to a CSV file
    output_file = os.path.join('Processed_Patient_Records', os.path.basename(file_path))
    features_df.to_csv(output_file, index=False)

def process_all_csv_files(directory):
    if not os.path.exists('Processed_Patient_Records'):
        os.makedirs('Processed_Patient_Records')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_csv_file(file_path)

if __name__ == "__main__":
    process_all_csv_files('Patient_Records')