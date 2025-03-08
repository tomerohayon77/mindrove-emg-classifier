# main.py
from multiprocessing import Process, Manager
import time
import keyboard_classify
import commands
import real_time_classify_per_move
import wifi_connect_windows

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['move'] = 'NONE'  # Initialize shared variable , what move we got?
        shared_data['action'] = 0  # Initialize shared variable , did we just got a movement?
        shared_data['start'] = 0  # Initialize shared variable , can we start classifying?
        shared_data['connected'] = 0  # Initialize shared variable, are we connected to the armband?

        p1 = Process(target=wifi_connect_windows.wifi_connect_windows, args=(shared_data,))
        p2 = Process(target=keyboard_classify.keyboard_classify, args=(shared_data,))
        p3 = Process(target=commands.commands, args=(shared_data,))
        p4 = Process(target=real_time_classify_per_move.real_time_classify_per_move, args=(shared_data,))


        p1.start()
        p2.start()
        p3.start()
        p4.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()

            print("Processes stopped by user.")
