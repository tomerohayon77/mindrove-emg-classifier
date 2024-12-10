import csv
import time
import os
import threading
import cv2
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
#to add the name of the recorded person
def record_video(duration, filename):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 27.0, (640, 480)) #put your camera fps

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

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

        os.makedirs('Record', exist_ok=True)
        csv_filename = os.path.join('Record', filename)
        video_filename = os.path.join('Record', 'recorded_video.avi')

        video_thread = threading.Thread(target=record_video, args=(duration, video_filename))
        video_thread.start()

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['N'] + [f'CH{i+1}' for i in emg_channels] + \
                     ['AccX', 'AccY', 'AccZ'] + \
                     ['GyX', 'GyY', 'GyZ'] + ['VBAT']
            writer.writerow(header)

            start_time = time.time()
            count = 1
            i = 0
            while count <= num_points:
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

                if (round(time.time()) - round(start_time) == i):
                    print(time.time() - start_time)
                    i+= 1
                time.sleep(1 / sampling_rate)  # Adjust the sleep time to match the sampling rate


        video_thread.join()

    except Exception as e:
        print(f'Error: {e}')
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == '__main__':
    record_data(duration=10, filename='recorded_data.csv')