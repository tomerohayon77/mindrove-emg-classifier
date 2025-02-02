from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from new_filtering import apply_filters
from sklearn.svm import SVC
import joblib
import numpy as np
from new_feature_extraction import extract_features
import time
import pandas as pd

model_path = r"C:\Technion\Project_A\Project_A\models\svm_model_2.pkl"
svm_model = joblib.load(model_path)
num_points = 50
memory_points = 1000
gyro_threshold = 2000
window_size = 200
def handeling_nans(array):
    array = np.nan_to_num(array, nan=0)
    return array
def normalize_signals(emg_data):
    min_val = np.min(emg_data, axis=0, keepdims=True)
    max_val = np.max(emg_data, axis=0, keepdims=True)
    normalized_signals = (emg_data - min_val) / (max_val - min_val)
    return normalized_signals
def extracting_features(emg_data,sampling_rate):
    # Apply filters to EMG signals
    filtered_emg_data = apply_filters(emg_data, sampling_rate)
    # Normalize the EMG signals
    normalized_emg = normalize_signals(filtered_emg_data)
    # Extract features
    return extract_features(normalized_emg, sampling_rate)

def movement_from_model(emg_data,sampling_rate):
    features_emg_data = extracting_features(emg_data,sampling_rate)
    features_array = np.array(list(features_emg_data.values())).reshape(1, -1)
    features_array = handeling_nans(features_array)
    return svm_model.predict(features_array)


def real_time_classify():
    BoardShim.enable_dev_board_logger()
    params = MindRoveInputParams()
    board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
    try:
        # Prepare session
        board_shim.prepare_session()
        board_shim.start_stream()
        print("start streaming")
        emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
        gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
        time.sleep(3)  # Wait for the board to start streaming
        all_emg_data = np.empty((8,0))

        print("start classifying")
        while True:
            if board_shim.get_board_data_count() >= num_points:
                new_data = board_shim.get_board_data()
                emg_data = np.array(new_data[emg_channels])
                all_emg_data = np.hstack((all_emg_data, emg_data))
                all_emg_data = all_emg_data[:, -memory_points:]
                gyro_data = np.array(new_data[gyro_channels])

                if np.any(gyro_data > gyro_threshold) and all_emg_data.shape[1] >= window_size:
                    movement = movement_from_model(all_emg_data[:,-window_size:], sampling_rate)
                    print("the move is ", movement)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == "__main__":
    real_time_classify()