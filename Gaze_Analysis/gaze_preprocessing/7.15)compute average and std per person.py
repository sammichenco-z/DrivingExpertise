import pandas as pd
import os
from pathlib import Path
import numpy as np


def compute_statistics_with_conditions(input_dir, stats_dir, aoi_file):
    # Create stats directory if it doesn't exist
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    # Read the AOI file
    aoi_df = pd.read_csv(aoi_file)[['Name', 'Condition']]

    # To store invalid data information
    invalid_data = []

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        stats_path = os.path.join(stats_dir, f"stats_{csv_file}")

        try:
            # Read CSV file
            df = pd.read_csv(input_path)

            # Group by Video
            groups = df.groupby('Video')

            # Check if there are exactly 360 groups
            if len(groups) != 360:
                print(f"Warning: {csv_file} contains {len(groups)} video groups instead of 360")

            # Initialize lists to store results
            stats_data = []

            # Process each group
            for video_name, group in groups:
                # Remove rows with no Fixation Point X
                filtered_group = group.dropna(subset=['Fixation Point X[px]'])

                # Check if filtered group is empty
                if filtered_group.empty:
                    invalid_data.append({
                        'File': csv_file,
                        'Video': video_name
                    })
                    # Add row with nan values
                    stats_data.append({
                        'Video': video_name,
                        'X_Average': np.nan,
                        'X_Std': np.nan,
                        'Y_Average': np.nan,
                        'Y_Std': np.nan,
                        'Condition': None  # Will be updated later
                    })
                    continue

                # Compute statistics
                x_avg = filtered_group['Gaze Point X[px]'].mean()
                x_std = filtered_group['Gaze Point X[px]'].std()
                y_avg = filtered_group['Gaze Point Y[px]'].mean()
                y_std = filtered_group['Gaze Point Y[px]'].std()

                # Append results
                stats_data.append({
                    'Video': video_name,
                    'X_Average': x_avg,
                    'X_Std': x_std,
                    'Y_Average': y_avg,
                    'Y_Std': y_std,
                    'Condition': None  # Will be updated later
                })

            # Create statistics DataFrame
            stats_df = pd.DataFrame(stats_data)

            # Add conditions from AOI file
            for idx, row in stats_df.iterrows():
                video_name = row['Video']
                # Find matching condition in AOI file
                matching_aoi = aoi_df[aoi_df['Name'] == video_name]
                if not matching_aoi.empty:
                    base_condition = matching_aoi.iloc[0]['Condition']
                    # Add _virtual or _real suffix based on video name
                    if 'bd' in video_name:
                        stats_df.at[idx, 'Condition'] = f"{base_condition}_virtual"
                    else:
                        stats_df.at[idx, 'Condition'] = f"{base_condition}_real"

            # Save statistics DataFrame
            stats_df.to_csv(stats_path, index=False)

            print(f"Successfully computed statistics for {csv_file}")
            print(f"Statistics saved to {stats_path}")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    # Save invalid data to CSV
    if invalid_data:
        invalid_df = pd.DataFrame(invalid_data)
        invalid_path = os.path.join(stats_dir, "invalid_data.csv")
        invalid_df.to_csv(invalid_path, index=False)
        print(f"Invalid data information saved to {invalid_path}")


# Example usage
input_directory = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output\\Exp"
stats_directory = "M:\\EEG_DATA\\EEG_data_0410\\average_and_std"
aoi_file = "M:\\EEG_DATA\\EEG_data_0410\\video_condition_list.csv"  # Replace with your AOI file path
compute_statistics_with_conditions(input_directory, stats_directory, aoi_file)