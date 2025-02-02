# main.py
from multiprocessing import Process, Manager
import time
import keyboard_classify
import commands
import wi_fi
import real_time_classify
if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['move'] = 'NONE'  # Initialize shared variable
        shared_data['action'] = 0  # Initialize shared variable
        shared_data['start'] = 0  # Initialize shared variable
        shared_data['connected'] = 0  # Initialize shared variable

        p1 = Process(target=wi_fi.wi_fi, args=(shared_data,))
        p2 = Process(target=keyboard_classify.keyboard_classify, args=(shared_data,))
        p3 = Process(target=real_time_classify.real_time_classify, args=(shared_data,))
        p4 = Process(target=commands.commands, args=(shared_data,))


        p1.start()
        p2.start()
        p3.start()
        #p4.start()


        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            #p4.terminate()
            print("Processes stopped by user.")
