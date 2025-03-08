import pandas as pd
import os

def swap_labels(input_file, output_file, label_col="Label"):  # Default column name updated
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # Load the CSV file
    df = pd.read_csv(input_file)

    # Ensure the label column exists
    if label_col not in df.columns:
        print(f"Error: Column '{label_col}' not found! Available columns: {list(df.columns)}")
        return

    # Convert label column to integers (handles cases where numbers are stored as strings)
    df[label_col] = df[label_col].astype(str).str.strip()  # Remove spaces
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')  # Convert to numbers

    # Debugging: Show unique values before swapping
    print(f"Unique values in '{label_col}' before swapping:", df[label_col].unique())

    # Swap 1s and 2s
    df[label_col] = df[label_col].replace({1: 2, 2: 1})

    # Save the modified CSV file
    df.to_csv(output_file, index=False)
    print(f"Modified file saved as: {output_file}")

# Example usage
input_file = r"C:\Users\User\PycharmProjects\Project_A\Patient_Records\liad_olier_personal_6\liad_olier_personal_6_20250307_152833.csv"
output_file = r"C:\Users\User\PycharmProjects\Project_A\Patient_Records\liad_olier_personal_6\liad_olier_personal_6_correct.csv"

swap_labels(input_file, output_file)
