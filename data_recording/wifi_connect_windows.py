import subprocess
import time
from multiprocessing import Process, Manager

SSID = "MindRove_ARB_0a25b8"
PASSWORD = "#mindrove"

def get_current_wifi():
    """Returns the SSID of the currently connected Wi-Fi network."""
    result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "SSID" in line and "BSSID" not in line:
            return line.split(":")[1].strip()
    return None

def disconnect_wifi():
    """Disconnects from the current Wi-Fi network."""
    subprocess.run(["netsh", "wlan", "disconnect"], capture_output=True, text=True)

def connect_to_wifi():
    """Connects to the desired Wi-Fi network."""
    result = subprocess.run(["netsh", "wlan", "connect", f"name={SSID}"], capture_output=True, text=True)
    return "Connection request was completed" in result.stdout

def wifi_connect_windows(shared_data):
    """Ensures the device stays connected to the correct Wi-Fi."""
    while True:
        current_wifi = get_current_wifi()
        if current_wifi != SSID:
            print(f"Connecting to {SSID}...")
            disconnect_wifi()
            time.sleep(3)  # Allow time to disconnect
            if connect_to_wifi():
                print("Connected successfully!")
                shared_data['connected'] = 1
            else:
                print("Failed to connect.")
                shared_data['connected'] = 0
        else:
            print(f"Already connected to {SSID}")
            shared_data['connected'] = 1
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        p = Process(target=wifi_connect_windows, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[Wi-Fi] Stopped by user.")
