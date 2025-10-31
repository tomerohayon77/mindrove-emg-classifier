import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data analysis')  # note the space
sys.path.insert(0, DATA_DIR)   # so 'feature_extraction' is importable
sys.path.insert(0, BASE_DIR)   # so 'filtering_per_window' is importable
import pandas as pd
import numpy as np
from filtering_per_window import apply_filters
from feature_extraction import extract_features


#
def segment_emg_signals(emg_signals, labels, session_id, window_size, overlap, fs):
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

    step_size = int(num_samples_per_window * (1 - overlap))
    if step_size < 1:
        step_size = 1

    for start in range(0, len(emg_signals) - num_samples_per_window, step_size):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        # mid-point in SAMPLES (not milliseconds)
        mid_point = start + num_samples_per_window // 2
        segment_labels.append(labels[mid_point])

    return np.array(segmented_emg), np.array(segment_labels), session_id


def normalize_signals(emg_signals):
    """
    Normalize EMG signals using RMS normalization.

    Parameters:
    emg_signals (numpy.ndarray): The EMG signals to normalize.

    Returns:
    numpy.ndarray: The RMS normalized EMG signals.
    """
    rms_value = np.sqrt(np.mean(emg_signals ** 2, axis=0, keepdims=True))
    normalized_signals = emg_signals / rms_value
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

    # ---- pick EMG columns (support CH1..CH8 or EMG1..EMG8)
    emg_cols_candidates = [
        [f'CH{i+1}'  for i in range(8)],
        [f'EMG{i+1}' for i in range(8)],
    ]
    emg_cols = None
    for cand in emg_cols_candidates:
        if set(cand).issubset(df.columns):
            emg_cols = cand
            break
    if emg_cols is None:
        raise ValueError(f"Could not find EMG columns. Have: {list(df.columns)}")

    emg_signals = df[emg_cols].values

    # ---- labels (support 'Label' or 'Task_number'; default zeros)
    if 'Label' in df.columns:
        labels = df['Label'].values
    elif 'Task_number' in df.columns:
        labels = df['Task_number'].values
    else:
        labels = np.zeros(len(df), dtype=int)

    # ---- session id-ish (best-effort)
    if 'SessionID' in df.columns:
        session_id = df['SessionID'].values
    elif 'Angle_sample' in df.columns:
        session_id = df['Angle_sample'].values
    else:
        session_id = np.arange(len(df))

    # Replace zeros with NaN for EMG, then interpolate/bfill
    df[emg_cols] = df[emg_cols].replace(0, np.nan)
    df[emg_cols] = df[emg_cols].interpolate(method='linear', axis=0)
    df[emg_cols] = df[emg_cols].bfill()

    # Average reference → filter → normalize
    emg_signals = emg_signals - np.mean(emg_signals, axis=1, keepdims=True)
    filtered_emg = apply_filters(emg_signals, fs)
    rms_value = np.sqrt(np.mean(filtered_emg ** 2, axis=0, keepdims=True))
    normalized_emg = filtered_emg / rms_value

    # Windowing (200 ms, 50% overlap)
    segmented_emg, segment_labels, session_id = segment_emg_signals(
        normalized_emg, labels, session_id, window_size=200, overlap=0.5, fs=fs
    )

    # Feature extraction per segment
    features_list = []
    for seg in segmented_emg:
        feats = extract_features(seg, fs)
        features_list.append(feats)

    # Build DF
    features_df = pd.DataFrame(features_list)
    features_df['Label'] = segment_labels
    # keep same length for session_id if it’s per-sample:
    if len(session_id) >= len(segment_labels):
        features_df['SessionID'] = session_id[:len(segment_labels)]
    else:
        features_df['SessionID'] = np.arange(len(segment_labels))

    # Save near the input file
    output_file = os.path.join(os.path.dirname(file_path), f"features_{os.path.basename(file_path)}")
    print("Saving to:", output_file)
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
    process_csv_file(
        r'C:\Users\Lenovo\Desktop\project_yad\Mindrove_armband_EMG_classifier\Patient_Records\Tomer_TEST_RUN_2\Tomer_TEST_RUN_2_20251031_222854.csv', fs=500)

