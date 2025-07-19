import pandas as pd
import os
from tqdm import tqdm

# Configuration parameters
COLUMN_MAPPING = {
    'gaze_x': 'Gaze Point X[px]',
    'gaze_y': 'Gaze Point Y[px]',
    'fixation_x': 'Fixation Point X[px]',
    'fixation_y': 'Fixation Point Y[px]',
    'video': 'Video',
    'frame_id': 'Image ID',
    'timestamp': 'Recording Time Stamp[ms]'
}
INPUT_BASE = "M:\\EEG_DATA\\EEG_data_0410\\video_only_with_frame"
OUTPUT_BASE = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\full_gaze"

fixation_only = False  # Set to True to process fixation-only data


def extract_user_id(filename):
    """Extract user ID from the filename."""
    try:
        parts = filename.split('_')
        if len(parts) >= 3 and parts[1].startswith('User'):
            return parts[1][4:]  # Extract User followed by numbers
        return filename.split('.')[0]  # Fallback to filename main part
    except:
        return 'unknown'


def process_single_group(group_name):
    """Process a single experiment group (Exp/Nov) and save valid data."""
    group_data = []

    input_folder = os.path.join(INPUT_BASE, group_name)
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input directory does not exist: {input_folder}")

    csv_files = []
    for root, _, files in os.walk(input_folder):
        csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    csv_files.sort()

    for file_path in tqdm(csv_files, desc=f"Processing {group_name} group"):
        try:
            df = pd.read_csv(file_path)
            user_id = extract_user_id(os.path.basename(file_path))

            # Filter rows if fixation_only is True
            if fixation_only:
                valid = df[
                    df[COLUMN_MAPPING['fixation_x']].notna() &
                    df[COLUMN_MAPPING['fixation_y']].notna()
                ]
                if valid.empty:
                    continue
            else:
                valid = df

            # Add user ID and group information to the data
            valid['subject_id'] = user_id
            valid['group'] = group_name

            # Keep columns of interest
            valid = valid[[
                'subject_id',
                COLUMN_MAPPING['video'],
                COLUMN_MAPPING['frame_id'],
                COLUMN_MAPPING['gaze_x'],
                COLUMN_MAPPING['gaze_y'],
                COLUMN_MAPPING['fixation_x'],
                COLUMN_MAPPING['fixation_y'],
                COLUMN_MAPPING['timestamp'],
                'group'
            ]]

            # Rename columns to standardized names
            valid.rename(columns={
                'subject_id': 'subject_id',
                COLUMN_MAPPING['video']: 'video',
                COLUMN_MAPPING['frame_id']: 'frame_id',
                COLUMN_MAPPING['gaze_x']: 'gaze_x',
                COLUMN_MAPPING['gaze_y']: 'gaze_y',
                COLUMN_MAPPING['fixation_x']: 'fixation_x',
                COLUMN_MAPPING['fixation_y']: 'fixation_y',
                COLUMN_MAPPING['timestamp']: 'timestamp_ms',
                'group': 'group'
            }, inplace=True)

            group_data.append(valid)
        except Exception as e:
            print(f"\nFailed to process file: {os.path.basename(file_path)}")
            print(f"Error: {str(e)}")
            print("=" * 50)

    # Combine all processed data for the group
    if group_data:
        return pd.concat(group_data, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    try:
        # Setup output directories
        os.makedirs(os.path.join(OUTPUT_BASE, 'Exp'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, 'Nov'), exist_ok=True)

        print("\nProcessing Exp group...")
        exp_df = process_single_group('Exp')

        print("\nProcessing Nov group...")
        nov_df = process_single_group('Nov')

        print("\nSaving processed data...")
        if not exp_df.empty:
            exp_df.to_csv(os.path.join(OUTPUT_BASE, 'Exp', 'exp_data.csv'), index=False)
            print(f"Saved Exp data: {len(exp_df):,} rows")
        else:
            print("No valid data for Exp group.")

        if not nov_df.empty:
            nov_df.to_csv(os.path.join(OUTPUT_BASE, 'Nov', 'nov_data.csv'), index=False)
            print(f"Saved Nov data: {len(nov_df):,} rows")
        else:
            print("No valid data for Nov group.")

    except Exception as e:
        print("\nA critical error occurred:")
        print(str(e))
        print("Please check the configuration paths and input file format.")


if __name__ == "__main__":
    main()