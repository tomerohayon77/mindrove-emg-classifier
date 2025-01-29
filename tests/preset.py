import asyncio
from bleak import BleakScanner, BleakClient
import protocol as prt
from multiprocessing import Manager

hand = "B8:D6:1A:43:60:52"  # hand's MAC

presets = [prt.PRESET0_UUID, prt.PRESET1_UUID, prt.PRESET2_UUID, prt.PRESET3_UUID, prt.PRESET4_UUID, prt.PRESET5_UUID, prt.PRESET6_UUID, prt.PRESET7_UUID
           , prt.PRESET8_UUID, prt.PRESET9_UUID, prt.PRESET10_UUID, prt.PRESET11_UUID]

commands = prt.commands
def preset(shared_data):
    async def run():
        try:
             async with BleakClient(hand) as client:
                print("Staring preset")
                await client.write_gatt_char(char_specifier=presets[prt.OPEN_TRIGGER], data=bytes(commands["open"]))
                await client.write_gatt_char(char_specifier=presets[prt.CLOSE_TRIGGER], data=bytes(commands["close"]))
                await client.write_gatt_char(char_specifier=presets[prt.LEFT_TRIGGER], data=bytes(commands["left"]))
                await client.write_gatt_char(char_specifier=presets[prt.RIGHT_TRIGGER], data=bytes(commands["right"]))
                print("Done Preset")

        except Exception as e:
            print(f"Error: {e}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())


if __name__ == "__main__":

    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=preset, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[preset] Stopped by user.")