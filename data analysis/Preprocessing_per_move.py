# preprocessing_per_move.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from FIltering import apply_filters
from feature_extraction import extract_features


# ---------- Repo-root discovery ----------
def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "Patient_Records").is_dir():
            return p
        if (p / ".git").exists():
            return p
    return start.parents[1]


REPO_ROOT = _find_repo_root(Path(__file__))
PATIENT_DIR = REPO_ROOT / "Patient_Records"
OUTPUT_DIR = REPO_ROOT / "Processed_Patient_Records_Per_Move"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def split_by_label_streaks_data_labels_with_chunks(data_arr, label_arr, chunk_size=200):
    """
    Split into segments by consecutive identical labels.
    Label==0 segments are split to fixed chunks (chunk_size) with tail merged into the last chunk.
    data_arr shape: [N, n_ch] or [N] per-sample records.
    """
    data_segments = []
    segment_labels = []
    current = []
    current_label = None

    for i in range(len(data_arr)):
        if not current:
            current.append(data_arr[i])
            current_label = label_arr[i]
        else:
            if label_arr[i] == current_label:
                current.append(data_arr[i])
            else:
                data_segments.append(np.array(current))
                segment_labels.append(current_label)
                current = [data_arr[i]]
                current_label = label_arr[i]

    if current:
        data_segments.append(np.array(current))
        segment_labels.append(current_label)

    final_data_segments = []
    final_label_segments = []

    for seg, lbl in zip(data_segments, segment_labels):
        if lbl == 0:
            # chunk into fixed windows
            while len(seg) >= chunk_size:
                final_data_segments.append(seg[:chunk_size])
                final_label_segments.append(lbl)
                seg = seg[chunk_size:]
            # merge remainder into the last chunk
            if len(seg) > 0:
                if final_data_segments:
                    final_data_segments[-1] = np.concatenate([final_data_segments[-1], seg])
                else:
                    final_data_segments.append(seg)
                final_label_segments.append(lbl) if not final_label_segments else None
        else:
            final_data_segments.append(seg)
            final_label_segments.append(lbl)

    return final_data_segments, np.array(final_label_segments)


def segment_emg_signals(emg_signals, labels, window_size, overlap, fs):
    num_samples_per_window = round(int(window_size * (fs / 1000)))
    segmented_emg = []
    segment_labels = []
    step_size = max(1, int(num_samples_per_window * (1 - overlap)))
    for start in range(0, len(emg_signals) - num_samples_per_window, step_size):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        mid_point = start + num_samples_per_window // 2
        segment_labels.append(labels[mid_point])
    return np.array(segmented_emg), np.array(segment_labels)


def normalize_signals(emg_signals):
    rms_value = np.sqrt(np.mean(emg_signals ** 2, axis=0, keepdims=True))
    rms_value[rms_value == 0] = 1.0
    return emg_signals / rms_value


# ---------- Core ----------
def process_csv_file(file_path, fs=500):
    df = pd.read_csv(file_path)

    # EMG columns (supports CH1..CH8; extend if needed)
    emg_cols = [f'CH{i+1}' for i in range(8)]
    if not set(emg_cols).issubset(df.columns):
        raise ValueError(f"Missing EMG columns in {file_path}. Columns: {list(df.columns)}")

    # Labels / SessionID
    labels = df['Label'].values if 'Label' in df.columns else np.zeros(len(df), dtype=int)
    session_id = df['SessionID'].values if 'SessionID' in df.columns else np.arange(len(df))

    # Clean EMG in DataFrame (zeros->NaN->interpolate->bfill)
    df[emg_cols] = df[emg_cols].replace(0, np.nan)
    df[emg_cols] = df[emg_cols].interpolate(method='linear', axis=0)
    df[emg_cols] = df[emg_cols].bfill()

    # IMPORTANT: refresh after cleaning
    emg_signals = df[emg_cols].values  # shape [N, 8]

    # Filter
    filtered_emg = apply_filters(emg_signals, fs)

    # Split into segments per label streaks (with chunking for label 0)
    segmented_emg, segment_labels = split_by_label_streaks_data_labels_with_chunks(
        data_arr=filtered_emg, label_arr=labels
    )

    # Feature extraction
    features_list = []
    for seg in segmented_emg:
        feats = extract_features(seg, fs)  # expects dict-like
        features_list.append(feats)

    features_df = pd.DataFrame(features_list)
    features_df['Label'] = segment_labels
    features_df['SessionID'] = session_id[:len(segment_labels)]

    # Save to repo-level folder
    out_path = OUTPUT_DIR / f"features_per_move_{os.path.basename(file_path)}"
    print("Saving to:", out_path)
    features_df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


def process_all_csv_files(patient_dir: Path):
    """Recursively process all CSVs under Patient_Records/."""
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient_Records not found at: {patient_dir}")
    for csv_path in patient_dir.rglob("*.csv"):
        if csv_path.name.startswith("features_per_move_"):
            continue
        print(f"Processing {csv_path}")
        process_csv_file(str(csv_path))


# ---------- Entry point ----------
if __name__ == "__main__":
    print(f"[repo_root] {REPO_ROOT}")
    print(f"[input_dir] {PATIENT_DIR}")
    print(f"[output_dir] {OUTPUT_DIR}")
    process_all_csv_files(PATIENT_DIR)
