import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FIltering import apply_filters
import time

indexes = [
    np.linspace(0,4999,5000),

    np.linspace(5000, 6499, 1500),
    np.linspace(6500, 7999, 1500),
    np.linspace(8000, 9499, 1500),
    np.linspace(9500, 10099, 1500),

    np.linspace(11000, 12499, 1500),
    np.linspace(12500, 13999, 1500),
    np.linspace(14000, 15499, 1500),
    np.linspace(15500, 16999, 1500),
    np.linspace(17000, 18499, 1500),
    np.linspace(18500, 19999, 1500),
    np.linspace(20000, 21499, 1500),
    np.linspace(21500, 22999, 1500),
    np.linspace(23000, 24499, 1500),
    np.linspace(24500, 25999, 1500),
    np.linspace(26000, 27499, 1500),
    np.linspace(27500, 28999, 1500),
    np.linspace(29000, 30499, 1500),
    np.linspace(30500, 31999, 1500),
    np.linspace(32000, 33499, 1500),
    np.linspace(33500, 34999, 1500),

    np.linspace(35000, 39999, 5000),
]

if __name__ == "__main__":
    fs = 500  # Sampling frequency in Hz

    data = pd.read_csv(r'C:\Technion\Project_A\Project_A\Patient_Records\albert_olier1\albert_olier1_20250125_182149.csv')

    # Identify the columns representing EMG channels (e.g., CH1, CH2, ...)
    channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccX', 'AccY', 'AccZ', 'GyX', 'GyY', 'GyZ']

    # Replace zeros with NaN
    data[channels] = data[channels].replace(0, np.nan)

    # Interpolate missing values (linear interpolation)
    data[channels] = data[channels].interpolate(method='linear', axis=0)

    # Forward-fill remaining NaNs (if at the start of the data)
    data[channels] = data[channels].bfill()

    # Extract EMG signals
    emg_signals = data[[f'CH{i+1}' for i in range(8)]].values

    labels = data['Label'].values
    filtered_Emg = apply_filters(emg_signals, fs)

    mean_filtered_emg = np.mean(filtered_Emg, axis=1)

    for i in range(len(indexes)):
        plt.figure(figsize=(12, 6))
        x = indexes[i].astype(int)
        mean_filtered_emg_taken = mean_filtered_emg[x]

        plt.plot(x, mean_filtered_emg_taken)

        plt.title(f'Filtered EMG mean [{i}]')
        plt.xlabel('sample')
        plt.ylabel('Amplitude')
        plt.show()
        time.sleep(0.1)
