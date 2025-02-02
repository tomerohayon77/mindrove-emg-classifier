import pandas as pd

gyro_treshold = 2000

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Technion\Project_A\Project_A\Patient_Records\liad_olier_31_1_4th_try\liad_olier_31_1_4th_try_20250131_155742.csv')

# Apply the condition: set label=0 where GyZ, GyY, GyX <= 2000
df.loc[(df['GyZ'] > 2000) | (df['GyY'] > 2000) | (df['GyX'] > 2000), 'label'] = 1


# Optionally, save the modified DataFrame back to a new CSV file
df.to_csv(r'C:\Technion\Project_A\Project_A\Patient_Records\liad_olier_31_1_4th_try\liad_olier_31_1_4th_try_labeled.csv', index=False)
