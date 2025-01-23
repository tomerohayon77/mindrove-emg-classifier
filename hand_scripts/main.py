# main.py
from multiprocessing import Process, Manager
import time
import classify
import commands

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['move'] = 'NONE'  # Initialize shared variable
        shared_data['action'] = 0  # Initialize shared variable
        shared_data['start'] = 0  # Initialize shared variable

        #p1 = Process(target=preset.preset, args=(shared_data,))
        p1 = Process(target=classify.classify, args=(shared_data,))
        p2 = Process(target=commands.commands, args=(shared_data,))

        #p1.start()
        p1.start()
        p2.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            p1.terminate()
            p2.terminate()
            print("Processes stopped by user.")
