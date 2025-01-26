import time
from multiprocessing import Manager


def hand_classify(shared_data):
    start_time = time.time()
    while True:
        if shared_data['start'] == 1 and time.time()-start_time > 0.2:
            start_time = time.time()
            if shared_data['move'] != 'NONE':
                shared_data['action'] = 1
                print("You did  - ", shared_data['move'])


if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=hand_classify, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[hand_classify] Stopped by user.")
