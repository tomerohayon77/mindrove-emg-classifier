import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV file (replace 'your_file.csv' with the actual file path)
df = pd.read_csv(r'C:\Users\User\PycharmProjects\Project_A\Patient_Records\Ilay_savion_9.1_2\Ilay_savion_9.1_2_20250112_111812.csv')

# Identify the columns representing EMG channels (e.g., CH1, CH2, ...)
channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']

# Replace zeros with NaN
df[channels] = df[channels].replace(0, np.nan)

# Interpolate missing values (linear interpolation)
df[channels] = df[channels].interpolate(method='linear', axis=0)

# Forward-fill remaining NaNs (if at the start of the data)
df[channels] = df[channels].bfill()


# Plot to verify
plt.figure(figsize=(12, 6))

plt.plot(df['CH1'], label='CH1')
plt.legend()
plt.title("Interpolated EMG Channels")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
