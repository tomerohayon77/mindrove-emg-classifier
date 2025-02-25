import os
import pandas as pd


def combine_csv_files(input_directory, output_file):
    all_dataframes = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Reading file: {file_path}")
                df = pd.read_csv(file_path)
                all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No CSV files found in the directory.")

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


if __name__ == "__main__":
    input_directory = r'C:\Users\User\PycharmProjects\Project_A\EMG\all_the_features'
    output_file = r'C:\Users\User\PycharmProjects\Project_A\EMG\all_the_features\combined_features.csv'
    combine_csv_files(input_directory, output_file)