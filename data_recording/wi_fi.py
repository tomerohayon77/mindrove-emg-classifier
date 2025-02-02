from multiprocessing import Manager
import pywifi
from pywifi import const
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Wi-Fi credentials
ssid = "MindRove_ARB_0a25b8"  # Replace with your Wi-Fi SSID
password = "#mindrove"  # Replace with your Wi-Fi password

def wi_fi(shared_data):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]

    # Check if already connected to the desired network
    if iface.status() == const.IFACE_CONNECTED:
        current_profiles = iface.network_profiles()
        for profile in current_profiles:
            if profile.ssid == ssid:
                print(f"Already connected to Wi-Fi: {ssid}")
                shared_data['connected'] = 1
                return True

    # Disconnect from the current network
    iface.disconnect()
    time.sleep(1)
    if iface.status() != const.IFACE_DISCONNECTED:
        print("Failed to disconnect from the current network.")
        return False

        # Try to connect to the desired network
    profile = pywifi.Profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_WPA2PSK)
    profile.cipher = const.CIPHER_TYPE_CCMP
    profile.key = password

    iface.remove_all_network_profiles()
    temp_profile = iface.add_network_profile(profile)
    iface.connect(temp_profile)

    start_time = time.time()
    while time.time() - start_time < 10:  # Wait up to 10 seconds
        if iface.status() == const.IFACE_CONNECTED:
            print(f"Successfully connected to Wi-Fi: {ssid}")
            shared_data['connected'] = 1
            return True
        time.sleep(1)

    print(f"Failed to connect to Wi-Fi: {ssid}")
    return False

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=wi_fi, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[wi_fi] Stopped by user.")

