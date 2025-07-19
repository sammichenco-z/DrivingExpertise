import pandas as pd
import os
from pathlib import Path


def process_single_file(input_file, condition_file_path, output_dir):
    # Read the input file
    df = pd.read_csv(input_file,
                     usecols=['Recording Time Stamp[ms]', 'Video', 'ExpClips.RESP'],
                     low_memory=False)  # Added to handle mixed types warning
    # print(f"Initial df rows: {len(df)}")

    # Drop rows where Video is NaN
    df = df.dropna(subset=['Video'])
    # print(f"After dropping NaN Video rows: {len(df)}")

    # Group by Video and calculate all needed values in one operation
    result_df = df.groupby('Video').agg({
        'Recording Time Stamp[ms]': lambda x: x.max() - x.min(),
        'ExpClips.RESP': 'first'  # Take the first response for each video
    }).reset_index()
    # print(f"After groupby: {len(result_df)}")

    # Rename the time column
    result_df = result_df.rename(columns={'Recording Time Stamp[ms]': 'time'})

    # Merge with condition information
    # Changed merge to use left join and added drop_duplicates
    result_df = result_df.merge(condition_df[['Name', 'Condition']],
                                left_on='Video',
                                right_on='Name',
                                how='left'
                                ).drop_duplicates(subset=['Video'])
    # print(f"After merge: {len(result_df)}")

    # Drop the extra 'Name' column
    result_df = result_df.drop('Name', axis=1)

    # Add real/virtual suffix to condition
    result_df['Condition'] = result_df.apply(
        lambda row: f"{row['Condition']}_{'virtual' if 'bd' in str(row['Video']) else 'real'}",
        axis=1
    )

    # Process Answer and Question columns
    result_df['Answer'] = result_df['ExpClips.RESP'].str.strip() == '{SPACE}'
    result_df['Question'] = result_df['Condition'].str.contains('Hazard')

    # Drop ExpClips.RESP column as it's no longer needed
    result_df = result_df.drop('ExpClips.RESP', axis=1)

    # Select and reorder final columns
    result_df = result_df[['Video', 'time', 'Question', 'Answer', 'Condition']]
    # print(f"Final df rows: {len(result_df)}")

    # Create output filename
    output_filename = os.path.join(output_dir, os.path.basename(input_file))

    # Save to CSV
    result_df.to_csv(output_filename, index=False)

    return result_df


def process_all_files(input_dir, condition_file_path, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read condition file once
    global condition_df
    condition_df = pd.read_csv(condition_file_path)

    # Process each CSV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            print(f"\nProcessing {filename}")
            process_single_file(input_file, condition_file_path, output_dir)


# Example usage
input_directory = "M:\\EEG_DATA\\EEG_data_0410\\Output\\Exp"
condition_file_path = "M:\\EEG_DATA\\EEG_data_0410\\video_condition_list.csv"
output_directory = "M:\\EEG_DATA\\EEG_data_0410\\reaction_time_and_accuracy"

process_all_files(input_directory, condition_file_path, output_directory)