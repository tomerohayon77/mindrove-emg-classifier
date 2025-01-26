from multiprocessing import Manager
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import time

def check_move(shared_data):
    while True:
        if shared_data['connected'] == 1:
            start_time = time.time()

            BoardShim.enable_dev_board_logger()
            params = MindRoveInputParams()
            board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

            board_shim.prepare_session()
            board_shim.start_stream()

            emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
            accel_channels = BoardShim.get_accel_channels(board_shim.board_id)
            gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
            battery_channel = BoardShim.get_battery_channel(board_shim.board_id)
            sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)

            if time.time() - start_time > 0.2:
                start_time = time.time()
                data = board_shim.get_board_data()  # Get all available data since the last call
                print("checking now the move")


if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=check_move, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[check_move] Stopped by user.")