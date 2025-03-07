import os
import pandas as pd
import numpy as np
from FIltering import apply_filters
from feature_extraction import extract_features
#
def segment_emg_signals(emg_signals, labels, session_id,  window_size, overlap, fs):
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
    num_samples_per_window = round(int(window_size * (fs / 1000)))
    segmented_emg = []
    segment_labels = []
    step_size = num_samples_per_window *(1 - overlap)
    for start in range(0, len(emg_signals) - num_samples_per_window, int(step_size)):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        mid_point = start + window_size // 2
        segment_labels.append(labels[mid_point])

    return np.array(segmented_emg), np.array(segment_labels), session_id

def normalize_signals(emg_data):
    min_val = np.min(emg_data, axis=0, keepdims=True)
    max_val = np.max(emg_data, axis=0, keepdims=True)
    normalized_signals = (emg_data - min_val) / (max_val - min_val)
    return normalized_signals

def average_reference(emg_signals):
    """
    Perform average referencing by subtracting the mean of all channels from each channel.

    Parameters:
    emg_signals (numpy.ndarray): The EMG signals to average reference.

    Returns:
    numpy.ndarray: The average referenced EMG signals.
    """
    mean_signal = np.mean(emg_signals, axis=1, keepdims=True)
    referenced_signals = emg_signals - mean_signal
    return referenced_signals

def process_csv_file(file_path, fs=500):
    # Load the recorded data
    df = pd.read_csv(file_path)

    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = df[[f'EMG{i + 1}' for i in range(8)]].values
    emg_signals = average_reference(emg_signals)
    #labels = df['label'].values
    labels = df['Task_number'].values
   # session_id = df['SessionID'].values
    session_id = df['Angle_sample'].values

    # Identify the columns representing EMG channels (e.g., CH1, CH2, ...)
    #channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']
    channels = [f'EMG{i + 1}' for i in range(8)]
    # Replace zeros with NaN
    df[channels] = df[channels].replace(0, np.nan)

    # Interpolate missing values (linear interpolation)
    df[channels] = df[channels].interpolate(method='linear', axis=0)

    # Forward-fill remaining NaNs (if at the start of the data)
    df[channels] = df[channels].bfill()

    # Apply filters to EMG signals
    filtered_emg = apply_filters(emg_signals, fs)

    # Normalize the EMG signals
    normalized_emg = normalize_signals(filtered_emg)

    # Segment the normalized EMG signals into time windows
    segmented_emg, segment_labels, session_id = segment_emg_signals(normalized_emg, labels, session_id, window_size=200, overlap=0.5, fs=fs)

    # Extract features from each segment
    features_list = []
    for segment in segmented_emg:  # Iterate over each segment
        features = extract_features(segment, fs)
        features_list.append(features)

    # Create a DataFrame with features and labels
    features_df = pd.DataFrame(features_list)
    features_df['Label'] = segment_labels
    features_df['SessionID'] = session_id[:len(segment_labels)]

    # Save the DataFrame to a CSV file
    output_file = os.path.join(os.path.dirname(file_path), f"features_{os.path.basename(file_path)}")
    features_df.to_csv(output_file, index=False)

def process_all_csv_files(directory):
    if not os.path.exists('Processed_Patient_Records'):
        os.makedirs('Processed_Patient_Records')

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                process_csv_file(file_path)

if __name__ == "__main__":
    process_all_csv_files(r'C:\Users\User\PycharmProjects\Project_A\EMG')
    #process_csv_file(r'C:\Technion\Project_A\Project_A\Patient_Records\liad_olier_left\liad_olier_left_labeled.csv')
