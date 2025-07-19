import pandas as pd
import os


def clean_data(input_path):
    # Read the CSV
    original_df = pd.read_csv(input_path)

    # Keep required columns (now including Image ID)
    required_cols = ['Name', 'Label', 'xtl', 'ytl', 'xbr', 'ybr', 'Condition', 'Image ID']
    df = original_df[required_cols]

    # Store original unique video names
    original_videos = original_df['Name'].unique()

    # Group by Name and apply filtering based on Condition
    def filter_by_condition(group):
        condition = group['Condition'].iloc[0]
        if condition == 'Control':
            return group[group['Label'].str.contains('DD', na=False)]
        elif condition in ['HazardOnly', 'OccludedHazard']:
            return group[group['Label'].str.contains('Vehicle', na=False)]
        elif condition == 'OcclusionOnly':
            return group[group['Label'].str.contains('Occlusion', na=False)]
        return pd.DataFrame()

    # Apply the filtering
    filtered_df = df.groupby('Name').apply(filter_by_condition).reset_index(drop=True)

    # Find and print missing videos
    cleaned_videos = filtered_df['Name'].unique()
    missing_videos = set(original_videos) - set(cleaned_videos)

    print("\nMissing videos:")
    for video in sorted(missing_videos):
        print(video)

    # Save the cleaned dataset
    cleaned_path = os.path.join(os.path.dirname(input_path), 'cleaned_data.csv')
    filtered_df.to_csv(cleaned_path, index=False)

    return cleaned_path


if __name__ == "__main__":
    cleaned_path = clean_data('M:\\EEG_DATA\\EEG_data_0410\\scale_full_AOI.csv')