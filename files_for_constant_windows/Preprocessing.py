# preprocessing.py
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------- Repo-root discovery (machine-agnostic) ----------
def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` to find the repo root that contains Patient_Records/ (preferred)
    or a .git/ folder (fallback). If neither is found, return the parent of the script's parent.
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "Patient_Records").is_dir():
            return p
        if (p / ".git").exists():
            return p
    return start.parents[1]

REPO_ROOT = _find_repo_root(Path(__file__))
PATIENT_DIR = REPO_ROOT / "Patient_Records"
OUTPUT_DIR = REPO_ROOT / "Processed_Patient_Records"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Make local project modules importable ----------
# Your feature code lives under "data analysis"
DATA_DIR = REPO_ROOT / "data analysis"  # note the space
# Add both the repo root and "data analysis" to sys.path (idempotent)
for p in (str(REPO_ROOT), str(DATA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now imports resolve regardless of where you run this file from
from filtering_per_window import apply_filters  # noqa: E402
from feature_extraction import extract_features  # noqa: E402


# ---------- Helpers ----------
def segment_emg_signals(emg_signals, labels, session_id, window_size, overlap, fs):
    """
    Segment EMG signals into time windows.
    window_size in milliseconds (e.g., 200), overlap in [0..1].
    Returns segmented_emg [n_segments, win_len, n_ch], segment_labels [n_segments], session_id (unchanged).
    """
    num_samples_per_window = int(round(window_size * (fs / 1000)))
    step_size = max(1, int(num_samples_per_window * (1 - overlap)))

    segmented_emg = []
    segment_labels = []

    for start in range(0, len(emg_signals) - num_samples_per_window, step_size):
        segment = emg_signals[start:start + num_samples_per_window]
        segmented_emg.append(segment)
        mid_point = start + num_samples_per_window // 2  # index (samples)
        segment_labels.append(labels[mid_point])

    return np.array(segmented_emg), np.array(segment_labels), session_id


def average_reference(emg_signals):
    """Average reference: subtract the per-sample mean across channels (axis=1)."""
    return emg_signals - np.mean(emg_signals, axis=1, keepdims=True)


# ---------- Core processing ----------
def process_csv_file(file_path, fs=500):
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # --- Robust EMG columns detection (CH1..CH8 or EMG1..EMG8)
    emg_cols_candidates = [
        [f"CH{i+1}" for i in range(8)],
        [f"EMG{i+1}" for i in range(8)],
    ]
    emg_cols = None
    for cand in emg_cols_candidates:
        if set(cand).issubset(df.columns):
            emg_cols = cand
            break
    if emg_cols is None:
        raise ValueError(f"Could not find EMG columns. Columns present: {list(df.columns)}")

    # --- Labels (support 'Label' or 'Task_number', else zeros)
    if "Label" in df.columns:
        labels = df["Label"].values
    elif "Task_number" in df.columns:
        labels = df["Task_number"].values
    else:
        labels = np.zeros(len(df), dtype=int)

    # --- Session ID best-effort (used for tracing back)
    if "SessionID" in df.columns:
        session_id = df["SessionID"].values
    elif "Angle_sample" in df.columns:
        session_id = df["Angle_sample"].values
    else:
        session_id = np.arange(len(df))

    # --- Clean EMG in DataFrame (replace zeros -> NaN -> interpolate -> bfill)
    df[emg_cols] = df[emg_cols].replace(0, np.nan)
    df[emg_cols] = df[emg_cols].interpolate(method="linear", axis=0)
    df[emg_cols] = df[emg_cols].bfill()

    # IMPORTANT: refresh numpy array after cleaning the DataFrame
    emg_signals = df[emg_cols].values  # shape [n_samples, n_ch]

    # --- Average reference -> filter -> RMS normalize (per channel)
    emg_signals = average_reference(emg_signals)
    filtered_emg = apply_filters(emg_signals, fs)  # expects shape [n_samples, n_ch]
    rms_value = np.sqrt(np.mean(filtered_emg ** 2, axis=0, keepdims=True))  # [1, n_ch]
    # Avoid divide-by-zero
    rms_value[rms_value == 0] = 1.0
    normalized_emg = filtered_emg / rms_value

    # --- Windowing (200 ms, 50% overlap)
    segmented_emg, segment_labels, session_id = segment_emg_signals(
        normalized_emg, labels, session_id, window_size=200, overlap=0.5, fs=fs
    )

    # --- Feature extraction per segment
    features_list = []
    for seg in segmented_emg:  # seg shape [win_len, n_ch]
        feats = extract_features(seg, fs)  # should return dict-like
        features_list.append(feats)

    # --- Build features DF
    features_df = pd.DataFrame(features_list)
    features_df["Label"] = segment_labels
    # keep same length for session_id if it’s per-sample:
    if len(session_id) >= len(segment_labels):
        features_df["SessionID"] = session_id[: len(segment_labels)]
    else:
        features_df["SessionID"] = np.arange(len(segment_labels))

    # --- Save into repo_root/Processed_Patient_Records ---
    output_file = OUTPUT_DIR / f"features{file_path.name}"
    print("Saving to:", output_file)
    features_df.to_csv(output_file, index=False)
    print(f"[saved] {output_file}")


def process_all_csv_files(patient_dir: Path):
    """
    Recursively process all CSVs under Patient_Records/ on any machine,
    writing outputs to Processed_Patient_Records/.
    """
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient_Records folder not found at: {patient_dir}")

    for csv_path in patient_dir.rglob("*.csv"):
        # Skip already-processed outputs if ever placed under Patient_Records
        if csv_path.name.startswith("processed_"):
            continue
        print(f"Processing {csv_path}")
        process_csv_file(csv_path)


# ---------- Entry point ----------
if __name__ == "__main__":
    print(f"[repo_root] {REPO_ROOT}")
    print(f"[input_dir] {PATIENT_DIR}")
    print(f"[output_dir] {OUTPUT_DIR}")
    process_all_csv_files(PATIENT_DIR)
