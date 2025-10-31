import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extraction import extract_features
from files_for_constant_windows.Preprocessing import average_reference

from FIltering import apply_filters

# Function to plot multiple signals
def plot_signals(signals, fs, titles):
    num_channels = signals.shape[1]
    t = np.linspace(0, len(signals), len(signals))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_channels))  # Generate a list of rainbow colors

    plt.figure(figsize=(15, 10))

    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(t, signals[:, i], color=colors[i])
        plt.title(titles[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to plot Emg signal, its mean, filtered and filtered mean
def mean_plot(emg_signals, fs=500):
    filtered_Emg = apply_filters(emg_signals, fs)
    mean_emg_signal = np.mean(emg_signals, axis=1)
    mean_filtered_emg = np.mean(filtered_Emg, axis=1)
    mean_abs_filtered_emg = abs(mean_filtered_emg)
    mean_signals = np.column_stack((mean_emg_signal, mean_filtered_emg, mean_abs_filtered_emg))
    mean_titles = ['Mean EMG Signal', 'Mean Filtered EMG Signal', 'Mean Absolute Filtered EMG Signal',
                   'Mean Their Filtered EMG Signal', 'Mean Absolute Their Filtered EMG Signal']
    plot_signals(mean_signals, fs, mean_titles)

def extract_windowed_features(emg_signals, window_size, fs):
    """
    Extract features for each window of EMG signals.

    Parameters:
    emg_signals (numpy.ndarray): The EMG signals.
    window_size (int): The size of the window in samples.
    fs (int): The sampling frequency.

    Returns:
    pandas.DataFrame: DataFrame containing the extracted features for each window.
    """
    all_windows_features = []

    # Iterate over the data with the defined window size
    for start in range(0, len(emg_signals) - window_size + 1, window_size):
        window = emg_signals[start:start + window_size]
        window_features = {}

        # Calculate features for each EMG channel in the window
        for i in range(window.shape[1]):
            features = extract_features(window[:, i], fs)
            channel_features = {f'CH{i + 1}_{key}': value for key, value in features.items()}
            window_features.update(channel_features)

        all_windows_features.append(window_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_windows_features)
    return features_df


if __name__ == "__main__":
    fs = 500  # Sampling frequency in Hz
    #data = pd.read_csv(r'Record/recorded_data.csv')
    data = pd.read_csv(r'C:\Users\Lenovo\Desktop\project_yad\Patient_Records\Tomer_TEST_RUN\Tomer_TEST_RUN_20251031_150942.csv')
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
    gy_signals = data[['GyX', 'GyY', 'GyZ']].values
    acc_signals = data[['AccX', 'AccY', 'AccZ']].values



    emg_titles = [f'EMG Channel {i + 1}' for i in range(emg_signals.shape[1])]
    gy_titles = [f'Gyroscope Channel {i + 1}' for i in range(gy_signals.shape[1])]
    acc_titles = [f'Accelerometer Channel {i + 1}' for i in range(acc_signals.shape[1])]
    #vbat = data['VBAT'].values
    average_reference_titles = [f'Average Referenced EMG Channel {i + 1}' for i in range(emg_signals.shape[1])]
    filtered_Emg = apply_filters(emg_signals, fs)
    average_reference_signals= average_reference(emg_signals)
    filtered_Emg_reference = apply_filters(average_reference_signals, fs)
    """
    mean_emg_signal = np.mean(emg_signals, axis=1)
    mean_filtered_emg =np.mean(filtered_Emg)
    mean_gy_signal = np.mean(gy_signals, axis=1)
    mean_acc_signal = np.mean(acc_signals, axis=1)

    mean_signals = np.column_stack((mean_emg_signal, mean_gy_signal, mean_acc_signal))
    mean_titles = ['Mean EMG Signal', 'Mean Gyroscope Signal', 'Mean Accelerometer Signal']
    plot_signals(mean_signals, fs, mean_titles)
    """
    #Plot entire EMG signals
    plot_signals(emg_signals, fs, emg_titles)
    plot_signals(filtered_Emg, fs, emg_titles)
    plot_signals(abs(filtered_Emg), fs, emg_titles)
    plot_signals(average_reference_signals, fs, average_reference_titles)
    plot_signals(filtered_Emg_reference, fs, average_reference_titles)
    plot_signals(abs(filtered_Emg_reference), fs, average_reference_titles)
    mean_plot(emg_signals, fs=500)


    # Plot entire Gyroscope signals
    plot_signals(gy_signals, fs, gy_titles)

    # Plot entire Accelerometer signals
    plot_signals(acc_signals, fs, acc_titles)


    """

    # Define window size (0.5 seconds)
    window_size = int(0.5 * fs)

    features_df = extract_windowed_features(emg_signals, window_size, fs)
    mean_features_df = extract_windowed_features_single(mean_emg_signal, window_size, fs)

    # Calculate and plot the correlation matrix
    correlation_matrix = mean_features_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()"""