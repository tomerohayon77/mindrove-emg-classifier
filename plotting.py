import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from feature_extraction import extract_features
import seaborn as sns

# Function to plot multiple signals with rainbow colors
def plot_signals(signals, fs, titles):
    num_channels = signals.shape[1]
    t = np.linspace(0, len(signals) / fs, len(signals))
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

def extract_windowed_features_single(emg_signal, window_size, fs):
    """
    Extract features for each window of a single EMG signal.

    Parameters:
    emg_signal (numpy.ndarray): The single EMG signal.
    window_size (int): The size of the window in samples.
    fs (int): The sampling frequency.

    Returns:
    pandas.DataFrame: DataFrame containing the extracted features for each window.
    """
    all_windows_features = []

    # Iterate over the data with the defined window size
    for start in range(0, len(emg_signal) - window_size + 1, window_size):
        window = emg_signal[start:start + window_size]
        features = extract_features(window, fs)
        all_windows_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_windows_features)
    return features_df

if __name__ == "__main__":
    fs = 500  # Sampling frequency in Hz
    data = pd.read_csv(r'C:\Users\User\Desktop\Record.csv')
    print(data.columns)
    # Extract EMG, Gyroscope, and Accelerometer signals
    emg_signals = data[['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']].values
    gy_signals = data[['GyX', 'GyY', 'GyZ']].values
    acc_signals = data[['AccX', 'AccY', 'AccZ']].values

    emg_titles = [f'EMG Channel {i + 1}' for i in range(emg_signals.shape[1])]
    gy_titles = [f'Gyroscope Channel {i + 1}' for i in range(gy_signals.shape[1])]
    acc_titles = [f'Accelerometer Channel {i + 1}' for i in range(acc_signals.shape[1])]
    vbat = data['Vbat'].values

    mean_emg_signal = np.mean(emg_signals, axis=1)
    mean_gy_signal = np.mean(gy_signals, axis=1)
    mean_acc_signal = np.mean(acc_signals, axis=1)

    mean_signals = np.column_stack((mean_emg_signal, mean_gy_signal, mean_acc_signal))
    mean_titles = ['Mean EMG Signal', 'Mean Gyroscope Signal', 'Mean Accelerometer Signal']


    #Plot entire EMG signals
    plot_signals(emg_signals, fs, emg_titles)

    # Plot entire Gyroscope signals
    plot_signals(gy_signals, fs, gy_titles)

    # Plot entire Accelerometer signals
    plot_signals(acc_signals, fs, acc_titles)


    plot_signals(mean_signals, fs, mean_titles)

    # Define window size (0.5 seconds)
    window_size = int(0.5 * fs)

    features_df = extract_windowed_features(emg_signals, window_size, fs)
    mean_features_df = extract_windowed_features_single(mean_emg_signal, window_size, fs)

    # Calculate and plot the correlation matrix
    correlation_matrix = mean_features_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()