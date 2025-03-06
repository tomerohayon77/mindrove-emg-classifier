import os
import pandas as pd
import numpy as np
from FIltering import apply_filters
from feature_extraction import extract_features
#

def split_by_label_streaks_data_labels_with_chunks(data_arr, label_arr, chunk_size=200):
    """
    Splits the data and label arrays into segments based on consecutive streaks of the same label value.
    If the label is 0, it further splits the data segment into chunks of 200 samples, ensuring no chunk has less than 200 samples (unless it's the last segment).

    Parameters:
    data_arr (numpy.ndarray): The input data array.
    label_arr (numpy.ndarray): The input label array.
    chunk_size (int): The size of chunks to split label 0 segments into.

    Returns:
    tuple: A tuple containing two lists:
        - The first list is a list of data segments (numpy arrays).
        - The second list is a list of labels, where each label corresponds to a segment.
    """
    data_segments = []
    segment_labels = []  # Will hold one label for each segment
    current_data_segment = []
    current_label = None

    # Iterate through the arrays
    for i in range(len(data_arr)):
        if not current_data_segment:  # Start a new segment if the current one is empty
            current_data_segment.append(data_arr[i])
            current_label = label_arr[i]  # Set the label for the new segment
        else:
            # Check if the label is the same as the last one in the segment
            if label_arr[i] == current_label:
                current_data_segment.append(data_arr[i])  # Continue the streak for data
            else:
                # If the label changes, save the current segment and start a new one
                data_segments.append(np.array(current_data_segment))  # Convert to numpy array
                segment_labels.append(current_label)  # Store the label for the current segment
                current_data_segment = [data_arr[i]]  # Start a new data segment
                current_label = label_arr[i]  # Set the new label

    # Add the last segment
    if current_data_segment:
        data_segments.append(np.array(current_data_segment))
        segment_labels.append(current_label)

    # Now handle splitting label 0 segments into chunks of 200 samples
    final_data_segments = []
    final_label_segments = []

    for i, data_segment in enumerate(data_segments):
        label = segment_labels[i]
        if label == 0:
            # Split the data segment into chunks of 200 samples
            while len(data_segment) >= chunk_size:
                final_data_segments.append(data_segment[:chunk_size])
                final_label_segments.append(label)  # The label for these chunks is 0
                data_segment = data_segment[chunk_size:]  # Remove the chunk from the original segment

            # If there are fewer than chunk_size samples left, merge it with the last chunk
            if len(data_segment) > 0:
                if final_data_segments:
                    final_data_segments[-1] = np.concatenate([final_data_segments[-1], data_segment])
                    final_label_segments[-1] = label  # The last chunk gets label 0
                else:
                    final_data_segments.append(data_segment)
                    final_label_segments.append(label)
        else:
            # For segments with label != 0, just add them as they are
            final_data_segments.append(data_segment)
            final_label_segments.append(label)

    return final_data_segments, np.array(final_label_segments)  # Return labels as a numpy array


def segment_emg_signals(emg_signals, labels,  window_size, overlap, fs):
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
    df = pd.read_csv(file_path)

    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = df[[f'CH{i+1}' for i in range(8)]].values
    labels = df['Label'].values
    session_id = df['SessionID'].values

    # Identify the columns representing EMG channels (e.g., CH1, CH2, ...)
    channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']

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
    segmented_emg, segment_labels = split_by_label_streaks_data_labels_with_chunks(data_arr=normalized_emg, label_arr=labels)

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
    output_file = os.path.join(os.path.dirname(file_path), f"features_per_move_{os.path.basename(file_path)}")
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
    #process_all_csv_files(r'C:\Users\User\PycharmProjects\Project_A\Paitient_records_for_features')
    process_csv_file(r'C:\Technion\Project_A\Project_A\Paitient_records_for_features\shira_hazrati_try1_labeled.csv')
