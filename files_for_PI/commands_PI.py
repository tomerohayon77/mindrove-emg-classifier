# commands_PI.py
from multiprocessing import Manager
import asyncio
from bleak import BleakScanner, BleakClient
import protocol as prt

hand = "B8:D6:1A:40:EF:D6"
presets = [prt.PRESET0_UUID, prt.PRESET1_UUID, prt.PRESET2_UUID, prt.PRESET3_UUID, prt.PRESET4_UUID, prt.PRESET5_UUID, prt.PRESET6_UUID, prt.PRESET7_UUID
           , prt.PRESET8_UUID, prt.PRESET9_UUID, prt.PRESET10_UUID, prt.PRESET11_UUID]
commands1 = prt.commands

import protocol as prt
def commands_PI(shared_data):
    async def run():
        try:
            async with BleakClient(hand) as client:
                print("Connected to ", hand)
                await asyncio.sleep(3)
                print("You may start")
                shared_data['start'] = 1
                while True:
                    if shared_data['action'] == 1:
                        shared_data['action'] = 0

                        if shared_data['move'] == 'up' or shared_data['move'] == 'open':
                            await client.write_gatt_char(prt.DIRECT_UUID, bytes(commands1["open"]))
                            print("Sending OPEN command")

                        elif shared_data['move'] == 'down' or shared_data['move'] == 'close':
                            await client.write_gatt_char(prt.DIRECT_UUID, bytes(commands1["close"]))
                            print("Sending CLOSE command")

                        elif shared_data['move'] == 'left':
                            await client.write_gatt_char(prt.DIRECT_UUID, bytes(commands1["left"]))
                            print("Sending LEFT command")

                        elif shared_data['move'] == 'right':
                            await client.write_gatt_char(prt.DIRECT_UUID, bytes(commands1["right"]))
                            print("Sending RIGHT command")

                        # in the comments are trigger commands if we want to use them
                        #elif shared_data['move'].isdigit():
                         #   await client.write_gatt_char(prt.TRIGGER_UUID, bytes([int(shared_data['move'])]))
                          #  print("Sending trigger number ", int(shared_data['move']))

                        #elif shared_data['move'] == 'a':
                         #   await client.write_gatt_char(prt.TRIGGER_UUID, bytes([10]))
                          #  print("Sending trigger number 10")

                        #elif shared_data['move'] == 'b':
                         #   await client.write_gatt_char(prt.TRIGGER_UUID, bytes([11]))
                          #  print("Sending trigger number 11")

                        else:
                            print("Invalid command", shared_data['move'])

        except Exception as e:
            print(f"Error: {e}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=commands_PI, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[commands] Stopped by user.")
