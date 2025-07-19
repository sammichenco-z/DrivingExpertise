import os
import pandas as pd
from pathlib import Path

# Define input and output paths
input_path = r"M:\EEG_DATA\EEG_data_0410\Output"
output_path = r"M:\EEG_DATA\EEG_data_0410\video_only"

# Define folders to process
folders = ["Exp", "Nov"]

# Define columns to keep
columns_to_keep = [
    "Record Name",
    "User",
    "Recording Time Stamp[ms]",
    "Triggle Send",
    "Triggle Receive",
    "Gaze Point X[px]",
    "Gaze Point Y[px]",
    "Fixation Point X[px]",
    "Fixation Point Y[px]",
    "Video",  # Keep only the later "Video" column
    "ExpClips.RESP"
]

# Function to process CSV files
def process_csv_files(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # Create output directory if it doesn't exist
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Read the CSV file
                try:
                    # Read the file with engine='python' to handle malformed CSVs
                    df = pd.read_csv(input_file_path, engine='python')

                    # Deduplicate columns by appending suffixes to duplicates
                    df.columns = pd.Series(df.columns).apply(lambda x: x if df.columns.tolist().count(x) == 1 else f"{x}_{df.columns.tolist().count(x)}")

                    # Ensure correct column selection
                    df = df[[col for col in columns_to_keep if col in df.columns]]

                    # Remove rows where 'Video' column is NaN
                    df = df[df["Video"].notna()]

                    # Write the processed CSV to the output folder
                    output_file_path = os.path.join(output_dir, file)
                    df.to_csv(output_file_path, index=False)
                except Exception as e:
                    print(f"Error processing file {input_file_path}: {e}")

# Process the Exp and Nov folders
for folder in folders:
    input_folder = os.path.join(input_path, folder)
    output_folder = os.path.join(output_path, folder)
    process_csv_files(input_folder, output_folder)

print("Processing complete!")