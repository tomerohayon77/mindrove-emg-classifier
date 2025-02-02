import pandas as pd

# File paths
input_file = r'C:\Technion\Project_A\Project_A\Patient_Records\shira_hazrati_1_2_try1\shira_hazrati_1_2_try1_20250201_121051.csv'
output_file = r'C:\Technion\Project_A\Project_A\Patient_Records\shira_hazrati_1_2_try1\shira_hazrati_try1_labeled_2_2.csv'

# Load CSV
df = pd.read_csv(input_file)

# Ensure 'Label' column exists and default it to 0
df['Label'] = 0

# Convert Action Name to lowercase for consistent comparison
df['Action Name'] = df['Action Name'].str.lower()

# Condition: At least one Gy value is ≥ 2000
mask = (df['GyX'].abs() >= 2000) | (df['GyY'].abs() >= 2000) | (df['GyZ'].abs() >= 2000)

# Assign labels based on Action Name for rows where the condition is met
df.loc[mask & (df['Action Name'] == 'open'), 'Label'] = 1
df.loc[mask & (df['Action Name'] == 'close'), 'Label'] = 2
df.loc[mask & (df['Action Name'] == 'right'), 'Label'] = 3
df.loc[mask & (df['Action Name'] == 'left'), 'Label'] = 4

# Save the modified file
df.to_csv(output_file, index=False)

print("Labeling completed successfully.")
