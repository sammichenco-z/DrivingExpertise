import pandas as pd
import os
from pathlib import Path

def extract_trial_data(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    columns_to_keep = [
        'Recording Time Stamp[ms]',
        'Triggle Receive',
        'Gaze Point Index',
        'Gaze Point X[px]',
        'Gaze Point Y[px]',
        'Fixation Index',
        'Fixation Duration[ms]',
        'Fixation Point X[px]',
        'Fixation Point Y[px]',
        'Video',
        'ExpClips.RESP'
    ]

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)

        try:
            # Read CSV file with specified columns
            df = pd.read_csv(input_path, usecols=columns_to_keep)

            # Keep only rows where Video column has a value
            df = df.dropna(subset=['Video'])

            # Save processed data to new file
            df.to_csv(output_path, index=False)
            print(f"Successfully processed {csv_file}")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

# Example usage
input_directory = "M:\\EEG_DATA\\EEG_data_0410\\Output\\Exp"
output_directory = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output"
extract_trial_data(input_directory, output_directory)