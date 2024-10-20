import csv
import time
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

def record_data(duration=60, filename='recorded_data.csv'):
    BoardShim.enable_dev_board_logger()
    params = MindRoveInputParams()
    board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        time.sleep(5)  # Wait for the board to start streaming

        emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
        accel_channels = BoardShim.get_accel_channels(board_shim.board_id)
        gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
        battery_channel = BoardShim.get_battery_channel(board_shim.board_id)

        sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
        print(f'Sampling Rate (Fs): {sampling_rate} Hz')  # Print the sampling rate
        num_points = duration * sampling_rate

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['N'] + [f'EMG_{i}' for i in emg_channels] + \
                     ['Acc_X', 'Acc_Y', 'Acc_Z'] + \
                     ['Gyro_X', 'Gyro_Y', 'Gyro_Z'] + ['VBAT']
            writer.writerow(header)

            start_time = time.time()
            count = 1
            while time.time() - start_time < duration:
                data = board_shim.get_board_data()  # Get all available data since the last call
                num_samples = data.shape[1]
                for i in range(num_samples):
                    row = [count] + \
                          [data[channel][i] for channel in emg_channels] + \
                          [data[channel][i] for channel in accel_channels] + \
                          [data[channel][i] for channel in gyro_channels] + \
                          [data[battery_channel][i]]
                    writer.writerow(row)
                    count += 1
                time.sleep(1)  # Adjust the sleep time as needed

    except Exception as e:
        print(f'Error: {e}')
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == '__main__':
    record_data(duration=20, filename='recorded_data.csv')