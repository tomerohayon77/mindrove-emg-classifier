import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from FIltering import apply_filters

if __name__ == "__main__":
    fs = 500  # Sampling frequency in Hz
    data = pd.read_csv(r'C:\Users\User\PycharmProjects\Project_A\Patient_Records\Roee_Savion_9.1\Roee_Savion_9.1_20250109_180436.csv')[0:7502]
    print(data.columns)

    # Identify the columns representing EMG channels (e.g., CH1, CH2, ...)
    channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccX', 'AccY', 'AccZ', 'GyX', 'GyY', 'GyZ']

    # Replace zeros with NaN
    data[channels] = data[channels].replace(0, np.nan)

    # Interpolate missing values (linear interpolation)
    data[channels] = data[channels].interpolate(method='linear', axis=0)

    # Forward-fill remaining NaNs (if at the start of the data)
    data[channels] = data[channels].bfill()

    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = data[[f'CH{i+1}' for i in range(8)]].values
    labels = data['Label'].values

    filtered_Emg = apply_filters(emg_signals, fs)
    mean_filtered_emg = np.mean(filtered_Emg, axis=1)

    # Plot the mean filtered EMG signal with different colors for each label

    for channel in range(emg_signals.shape[1]):
        plt.figure(figsize=(10, 5))
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            indices = np.where(labels == label)[0]
            plt.plot(indices, filtered_Emg[indices, channel], color=color, label=f'Label {label}')

        plt.title('Mean Filtered EMG Signal with Different Colors for Each Label')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()