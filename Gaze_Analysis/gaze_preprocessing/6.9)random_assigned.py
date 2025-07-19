import pandas as pd
import numpy as np
import os
import re


def find_matching_virtual_videos(control_video, condition_videos, condition_type):
    # Extract components from control video name
    # Handle all four cases: veh, wal, veh_rev, wal_rev
    patterns = [
        r'(\d+)_bdFixedNull_veh_noveh_(\d+\.\d+)_\d+',
        r'(\d+)_bdFixedNull_wal_noveh_(\d+\.\d+)_\d+',
        r'(\d+)_bdFixedNull_veh_rev_noveh_(\d+\.\d+)_\d+',
        r'(\d+)_bdFixedNull_wal_rev_noveh_(\d+\.\d+)_\d+'
    ]

    match = None
    vehicle_type = None

    for pattern in patterns:
        match = re.match(pattern, control_video)
        if match:
            if 'veh_rev' in pattern:
                vehicle_type = 'veh_rev'
            elif 'wal_rev' in pattern:
                vehicle_type = 'wal_rev'
            elif 'veh' in pattern:
                vehicle_type = 'veh'
            elif 'wal' in pattern:
                vehicle_type = 'wal'
            break

    if not match:
        return None

    bd_num, speed = match.groups()

    # Define patterns for different conditions based on vehicle type
    if condition_type == 'HazardOnly':
        if '_rev' in vehicle_type:
            # Should match: 0_bdFixedNull_veh_rev_4.5_time or 0_bdFixedNull_wal_rev_4.5_time
            pattern = f'{bd_num}_bdFixedNull_{vehicle_type}_{speed}_\\d+'
        else:
            # Should match: 0_bdFixedNull_veh_4.5_time or 0_bdFixedNull_wal_4.5_time
            pattern = f'{bd_num}_bdFixedNull_{vehicle_type}_{speed}_\\d+'
    elif condition_type == 'OcclusionOnly':
        if '_rev' in vehicle_type:
            # Should match: bd0_veh_rev_noveh_4.5_time or bd0_wal_rev_noveh_4.5_time
            pattern = f'bd{bd_num}_{vehicle_type}_noveh_{speed}_\\d+'
        else:
            # Should match: bd0_veh_noveh_4.5_time or bd0_wal_noveh_4.5_time
            pattern = f'bd{bd_num}_{vehicle_type}_noveh_{speed}_\\d+'
    elif condition_type == 'OccludedHazard':
        if '_rev' in vehicle_type:
            # Should match: bd0_veh_rev_4.5_time or bd0_wal_rev_4.5_time
            pattern = f'bd{bd_num}_{vehicle_type}_{speed}_\\d+'
        else:
            # Should match: bd0_veh_4.5_time or bd0_wal_4.5_time
            pattern = f'bd{bd_num}_{vehicle_type}_{speed}_\\d+'

    # Find matching video
    matching_videos = [v for v in condition_videos if re.search(pattern, v)]

    if matching_videos:
        return matching_videos[0]
    return None


def create_video_matching(input_path):
    # Read the data
    df = pd.read_csv(input_path)

    # Separate virtual and real data
    virtual_df = df[df['Name'].str.contains('bd', case=False)]
    real_df = df[~df['Name'].str.contains('bd', case=False)]

    # Process real videos with random matching
    real_control = real_df[real_df['Condition'] == 'Control']['Name'].unique()
    real_hazard = real_df[real_df['Condition'] == 'HazardOnly']['Name'].unique()
    real_occlusion = real_df[real_df['Condition'] == 'OcclusionOnly']['Name'].unique()
    real_occludedhazard = real_df[real_df['Condition'] == 'OccludedHazard']['Name'].unique()

    np.random.seed(42)
    min_length = min(len(real_control), len(real_hazard),
                     len(real_occlusion), len(real_occludedhazard))

    real_matching = pd.DataFrame({
        'Control': np.random.choice(real_control, min_length, replace=False),
        'HazardOnly': np.random.choice(real_hazard, min_length, replace=False),
        'OcclusionOnly': np.random.choice(real_occlusion, min_length, replace=False),
        'OccludedHazard': np.random.choice(real_occludedhazard, min_length, replace=False)
    })

    # Process virtual videos with rule-based matching
    virtual_control = virtual_df[virtual_df['Condition'] == 'Control']['Name'].unique()
    virtual_hazard = virtual_df[virtual_df['Condition'] == 'HazardOnly']['Name'].unique()
    virtual_occlusion = virtual_df[virtual_df['Condition'] == 'OcclusionOnly']['Name'].unique()
    virtual_occludedhazard = virtual_df[virtual_df['Condition'] == 'OccludedHazard']['Name'].unique()

    virtual_matching_data = []
    unmatched_videos = []

    for control_video in virtual_control:
        hazard_match = find_matching_virtual_videos(control_video, virtual_hazard, 'HazardOnly')
        occlusion_match = find_matching_virtual_videos(control_video, virtual_occlusion, 'OcclusionOnly')
        occludedhazard_match = find_matching_virtual_videos(control_video, virtual_occludedhazard, 'OccludedHazard')

        if hazard_match and occlusion_match and occludedhazard_match:
            virtual_matching_data.append({
                'Control': control_video,
                'HazardOnly': hazard_match,
                'OcclusionOnly': occlusion_match,
                'OccludedHazard': occludedhazard_match
            })
        else:
            unmatched_videos.append({
                'Control': control_video,
                'HazardOnly': hazard_match,
                'OcclusionOnly': occlusion_match,
                'OccludedHazard': occludedhazard_match
            })

    virtual_matching = pd.DataFrame(virtual_matching_data)

    # Save matchings
    output_dir = os.path.dirname(input_path)
    virtual_matching.to_csv(os.path.join(output_dir, 'virtual_matching.csv'), index=False)
    real_matching.to_csv(os.path.join(output_dir, 'real_matching.csv'), index=False)

    # Print unmatched videos
    if unmatched_videos:
        print("\nUnmatched Virtual Videos:")
        for video in unmatched_videos:
            print("\nControl Video:", video['Control'])
            print("Missing matches:")
            if not video['HazardOnly']:
                print("- HazardOnly")
            if not video['OcclusionOnly']:
                print("- OcclusionOnly")
            if not video['OccludedHazard']:
                print("- OccludedHazard")

    # Print summary of matched videos
    print("\nMatched Virtual Videos Summary:")
    print(f"Total matched sets: {len(virtual_matching_data)}")
    if len(virtual_matching_data) > 0:
        # Print a few examples of matched sets
        print("\nExamples of matched sets:")
        for i in range(min(3, len(virtual_matching_data))):
            print(f"\nSet {i + 1}:")
            print(virtual_matching_data[i])

    return virtual_matching, real_matching


def create_comparison_datasets(input_path, virtual_matching, real_matching):
    # Read the data
    df = pd.read_csv(input_path)

    def create_single_comparison(control_videos, condition_videos, condition_name, data_type):
        comparison_dfs = []

        # Process each pair of videos
        for control_video, condition_video in zip(control_videos, condition_videos):
            # Get condition video data
            condition_data = df[df['Name'] == condition_video].copy()

            # Get control video info
            control_info = df[df['Name'] == control_video].iloc[0]

            # Create modified control data
            modified_control = condition_data.copy()
            modified_control['Name'] = control_info['Name']
            modified_control['Label'] = control_info['Label']
            modified_control['Condition'] = 'Control'

            # Add both to the comparison dataset
            comparison_dfs.append(modified_control)
            comparison_dfs.append(condition_data)

        # Combine all data
        comparison_df = pd.concat(comparison_dfs, ignore_index=True)

        # Save the dataset
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f'{data_type}_{condition_name.lower()}_comparison.csv')
        comparison_df.to_csv(output_path, index=False)

        return comparison_df

    # Process virtual data
    print("\nProcessing Virtual Data:")
    virtual_hazard_df = create_single_comparison(
        virtual_matching['Control'], virtual_matching['HazardOnly'], 'Hazard', 'virtual')
    virtual_occlusion_df = create_single_comparison(
        virtual_matching['Control'], virtual_matching['OcclusionOnly'], 'Occlusion', 'virtual')
    virtual_occludedhazard_df = create_single_comparison(
        virtual_matching['Control'], virtual_matching['OccludedHazard'], 'OccludedHazard', 'virtual')

    # Process real data
    print("\nProcessing Real Data:")
    real_hazard_df = create_single_comparison(
        real_matching['Control'], real_matching['HazardOnly'], 'Hazard', 'real')
    real_occlusion_df = create_single_comparison(
        real_matching['Control'], real_matching['OcclusionOnly'], 'Occlusion', 'real')
    real_occludedhazard_df = create_single_comparison(
        real_matching['Control'], real_matching['OccludedHazard'], 'OccludedHazard', 'real')

    # Print summary
    for data_type, datasets in [
        ('Virtual', [virtual_hazard_df, virtual_occlusion_df, virtual_occludedhazard_df]),
        ('Real', [real_hazard_df, real_occlusion_df, real_occludedhazard_df])
    ]:
        print(f"\n{data_type} Dataset Summary:")
        for name, df in zip(['Hazard', 'Occlusion', 'Occluded Hazard'], datasets):
            print(f"\n{name} Comparison:")
            print(f"Total videos: {len(df['Name'].unique())}")
            print("Videos per condition:")
            print(df.groupby('Condition')['Name'].nunique())
            print("Rows per condition:")
            print(df.groupby('Condition').size())


if __name__ == "__main__":
    input_path = 'M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv'

    # First create and save the matchings
    print("Creating video matchings...")
    virtual_matching, real_matching = create_video_matching(input_path)
    print("\nVideo matchings created and saved to 'virtual_matching.csv' and 'real_matching.csv'")

    # Then create the comparison datasets
    print("\nCreating comparison datasets...")
    create_comparison_datasets(input_path, virtual_matching, real_matching)