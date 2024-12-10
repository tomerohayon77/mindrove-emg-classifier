import csv
import time
import os
import threading
import cv2
from datetime import datetime
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# Updated recording protocol with consistent labels for variations
PROTOCOL = [
    ("Rest", 10, 0),
    ("Open Hand Soft", 5, 1),
    ("Open Hand Hard", 5, 1),
    ("Open Hand Fast", 5, 1),
    ("Open Hand Slow", 5, 1),
    ("Close Hand Soft", 5, 2),
    ("Close Hand Hard", 5, 2),
    ("Close Hand Fast", 5, 2),
    ("Close Hand Slow", 5, 2),
    ("Rotate Hand Right Soft", 5, 3),
    ("Rotate Hand Right Hard", 5, 3),
    ("Rotate Hand Right Slow", 5, 3),
    ("Rotate Hand Right Fast", 5, 3),
    ("Rotate Hand Left Soft", 5, 4),
    ("Rotate Hand Left Hard", 5, 4),
    ("Rotate Hand Left Slow", 5, 4),
    ("Rotate Hand Left Fast", 5, 4),
    ("Random Movements", 10, 5),
    ("Rest Transitions", 3, 0)
]

def record_video(duration, filename):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 27.0, (640, 480))  # Adjust FPS and resolution as needed

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def record_data_and_protocol(duration, csv_filename, protocol):
    """Record data while showing the protocol."""
    BoardShim.enable_dev_board_logger()
    params = MindRoveInputParams()
    board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

    try:
        # Prepare session
        board_shim.prepare_session()
        board_shim.start_stream()
        time.sleep(5)  # Wait for the board to start streaming

        # Get sensor channels
        emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
        accel_channels = BoardShim.get_accel_channels(board_shim.board_id)
        gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
        battery_channel = BoardShim.get_battery_channel(board_shim.board_id)

        # Sampling rate
        sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
        print(f'Sampling Rate (Fs): {sampling_rate} Hz')
        num_points = duration * sampling_rate

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['SessionID', 'N', 'Timestamp'] + \
                     [f'CH{i+1}' for i in emg_channels] + \
                     ['AccX', 'AccY', 'AccZ', 'GyX', 'GyY', 'GyZ', 'VBAT'] + \
                     ['Label', 'Action Name', 'ActionTime']
            writer.writerow(header)

            count = 1
            session_id = os.path.basename(csv_filename).split('.')[0]
            for action, action_duration, label in protocol:
                # Print the action at the start of the section
                print(f"\n{action}")
                action_start_time = time.time()
                action_time = 0  # Track time for the action
                elapsed = 0  # Track elapsed seconds for printing once per second

                while time.time() - action_start_time < action_duration:
                    data = board_shim.get_board_data()  # Get all available data since the last call
                    num_samples = data.shape[1]
                    for i in range(num_samples):
                        if count > num_points:
                            break
                        elapsed_time = time.time() - action_start_time  # Time from the start of the action

                        # Map elapsed time to Label and ActionName
                        action_name = action
                        action_time = elapsed_time

                        # Write data to CSV, including SessionID, Label, Action Name, and ActionTime
                        row = [session_id, count, elapsed_time] + \
                              [data[channel][i] for channel in emg_channels] + \
                              [data[channel][i] for channel in accel_channels] + \
                              [data[channel][i] for channel in gyro_channels] + \
                              [data[battery_channel][i]] + \
                              [label, action_name, action_time]
                        writer.writerow(row)
                        count += 1

                    # Print elapsed time once per second
                    current_elapsed = int(time.time() - action_start_time)
                    if current_elapsed > elapsed:
                        elapsed = current_elapsed
                        print(f"{elapsed}s")  # Print the elapsed time for the action

                # Pause for a moment to ensure smooth transition between actions
                time.sleep(0.5)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == '__main__':
    # Get person name
    person_name = input("Enter the name of the recorded person: ")

    # File paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join('Patient_Records', person_name)
    os.makedirs(directory, exist_ok=True)
    csv_filename = os.path.join(directory, f"{person_name}_{timestamp}.csv")
    video_filename = os.path.join(directory, f"{person_name}_{timestamp}.avi")

    # Calculate total recording time
    total_duration = sum([duration for _, duration, _ in PROTOCOL])

    # Start video recording in a separate thread
    video_thread = threading.Thread(target=record_video, args=(total_duration, video_filename))
    video_thread.start()

    # Record data and display protocol
    record_data_and_protocol(total_duration, csv_filename, PROTOCOL)

    # Ensure video recording finishes
    video_thread.join()
