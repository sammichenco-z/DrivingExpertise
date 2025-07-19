import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_date_and_time(filename):
    """
    Extracts the date and time information from the filename.
    Assumes the filename format is "raw_User30_230807175025_0418144743.csv".
    """
    match = re.search(r"(\d{6})(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        month = date_part[2:4]
        day = date_part[4:6]
        time = time_part[:2] + time_part[2:4] + time_part[4:]
        return month, day, time
    return None, None, None


def extract_eprime_date(filename):
    """
    Extracts the date information from the Eprime filename.
    Assumes the filename format is "EEG-080501-1.xlsx".
    """
    match = re.search(r"EEG-(\d{6})-(\d+)\.xlsx", filename)
    if match:
        date_part, temp = match.groups()
        month = date_part[:2]
        day = date_part[2:4]
        order = date_part[4:]
        return month + day + order
    return None


def preprocess_gaze_files(gaze_folder):
    """
    Preprocesses gaze files and returns a dictionary of gaze date information.
    """
    gaze_files = [f for f in os.listdir(gaze_folder) if f.endswith(".csv")]
    gaze_date_info = {}
    date_order = {}  # Dictionary to store the order for each date

    for gaze_file in gaze_files:
        # print("processing ", gaze_file)
        gaze_month, gaze_day, gaze_time = extract_date_and_time(gaze_file)
        if gaze_month and gaze_day and gaze_time:
            date_key = (gaze_month, gaze_day)
            if date_key not in date_order:
                date_order[date_key] = 1
            else:
                date_order[date_key] += 1
            order = date_order[date_key]
            gaze_date_info[gaze_file] = (gaze_month, gaze_day, order)
            # print(gaze_month, gaze_day, order)

    return gaze_date_info


def preprocess_eprime_files(eprime_folder):
    """
    Preprocesses Eprime files and returns a dictionary of Eprime date information.
    """
    eprime_files = [f for f in os.listdir(eprime_folder) if f.endswith(".xlsx")]
    eprime_date_info = {}

    # Create a dictionary to store the old order for each month+day combination
    month_day_order = {}

    for eprime_file in eprime_files:
        # print("Processing eprime ", eprime_file)
        eprime_date = extract_eprime_date(eprime_file)
        if eprime_date:
            month_day = eprime_date[:4]  # Extract month+day
            old_order = int(eprime_date[4:])  # Extract old order
            # print("month_day ", month_day)
            # print("old order ", old_order)
            if month_day not in month_day_order:
                month_day_order[month_day] = []
            month_day_order[month_day].append(old_order)

    # Compute the new order for each month+day combination
    for month_day, old_orders in month_day_order.items():
        old_orders.sort()  # Sort old orders
        for i, old_order in enumerate(old_orders):
            eprime_date_info[f"EEG-{month_day}0{old_order}-1.xlsx"] = month_day + str(i + 1)

    return eprime_date_info


def rank_files_by_time_gaze(files):
    """
    Ranks gaze files based on their time (earlier files first).
    """
    return sorted(files, key=lambda x: extract_date_and_time(x)[2] if extract_date_and_time(x) else "")


def match_files(gaze_date_info, eprime_date_info):
    """
    Matches gaze files with corresponding Eprime files using preprocessed data.
    """
    gaze_dict = {}

    for gaze_file, (gaze_month, gaze_day, gaze_time) in gaze_date_info.items():
        gaze_order = str(gaze_time)
        eprime_date = gaze_month + gaze_day + gaze_order
        # print("Processing ", gaze_file, " with info ", eprime_date)
        eprime_file = next((k for k, v in eprime_date_info.items() if v == eprime_date), None)
        if eprime_file:
            gaze_dict[gaze_file] = eprime_file
            # print("INFO found ", eprime_file)
        else:
            print(f"No matching Eprime file found for {gaze_file}")

    return gaze_dict

def calculate_frames(df):
    start_time = df['Recording Time Stamp[ms]'].iloc[0]
    df['Frame'] = ((df['Recording Time Stamp[ms]'] - start_time) // 100) * 3 - 1
    remainder = (df['Recording Time Stamp[ms]'] - start_time) % 100
    df['Frame'] += np.where(remainder <= 33, 1, np.where(remainder <= 66, 2, 3))
    return df

# def calculate_frames(group):
#     group = group.copy()
#     start_time = group['Recording Time Stamp[ms]'].iloc[0]
#     group['Frame'] = group['Recording Time Stamp[ms]'].apply(lambda x: calculate_frame_number(x, start_time))
#     return group
#
# def calculate_frame_number(current_time, start_time):
#     elapsed_time = current_time - start_time
#     frame_number = (elapsed_time // 100) * 3 - 1
#     remainder = elapsed_time % 100
#
#     if remainder <= 33:
#         frame_number += 1
#     elif remainder <= 66:
#         frame_number += 2
#     else:
#         frame_number += 3
#
#     return frame_number

def merge_files(gaze_folder, eprime_folder, gaze_eprime_mapping, save_path, additional_csv_path):
    additional_df = pd.read_csv(additional_csv_path)

    # Find unique labels in the additional_df
    unique_labels = additional_df['Label'].unique()
    # Rename 'Image ID' column to 'Frame'
    additional_df.rename(columns={'Image ID': 'Frame'}, inplace=True)

    # Specify the columns to read and their data types
    columns_to_read = ['Recording Time Stamp[ms]', 'Triggle Receive', 'Gaze Point X[px]',
                       'Gaze Point Y[px]', 'Fixation Point X[px]', 'Fixation Point Y[px]']
    dtype_dict = {
        'Recording Time Stamp[ms]': 'int64',
        'Triggle Receive': 'float64',
        'Gaze Point X[px]': 'float64',
        'Gaze Point Y[px]': 'float64',
        'Fixation Point X[px]': 'float64',
        'Fixation Point Y[px]': 'float64'
    }



    for gaze_file, eprime_file in tqdm(gaze_eprime_mapping.items(), desc="Processing Files"):
        print(f"Processing {gaze_file} with {eprime_file}")
        gaze_path = os.path.join(gaze_folder, gaze_file)
        eprime_path = os.path.join(eprime_folder, eprime_file)

        # Read only the necessary columns
        gaze_df = pd.read_csv(gaze_path, usecols=columns_to_read, dtype=dtype_dict, low_memory=False)
        eprime_df = pd.read_excel(eprime_path)

        # Remove rows with empty ExpClips.DEVICE in Eprime
        eprime_df = eprime_df.dropna(subset=["ExpClips.DEVICE"])

        # Create a 'Name' column in eprime_df from the 'Video' column
        eprime_df['Name'] = eprime_df['Video'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

        # Initialize the 'Name' column in gaze_df
        gaze_df['Name'] = np.nan

        num_trials = 0
        trial_started = False
        progress_bar = tqdm(total=len(eprime_df), desc="Matching Trials...", leave=False)

        # Process gaze file
        for index, row in gaze_df.iterrows():
            triggle_value = row["Triggle Receive"]

            if pd.notna(triggle_value) and '20' in str(triggle_value):
                start_row = index
                trial_started = True
                eprime_row = eprime_df.iloc[num_trials]
                video_name = os.path.splitext(os.path.basename(eprime_row["Video"]))[0]
                gaze_df.loc[index, 'Name'] = video_name

            elif pd.notna(triggle_value) and '21' in str(triggle_value) and triggle_value != 21.0:
                if not trial_started:
                    print(f"ERROR found at index {index}")
                trial_started = False
                end_row = index
                if num_trials < len(eprime_df):
                    # Get corresponding Eprime data
                    gaze_df.loc[start_row:end_row, "Name"] = video_name
                    num_trials += 1
                    progress_bar.update(1)

        progress_bar.close()
        print("Processing AOI comparisons...")

        # Group by 'Name' and calculate frames
        gaze_df = gaze_df.groupby('Name', group_keys=True).apply(calculate_frames).reset_index(drop=True)

        # Initialize label-specific 'inAOI' columns
        for label in unique_labels:
            gaze_df[f'inAOI_{label}'] = False

        # Vectorized approach to check if each gaze point is within the AOI bounds
        for label in unique_labels:
            label_df = additional_df[additional_df['Label'] == label]
            for _, a_row in label_df.iterrows():
                video_name = a_row['Name']
                frame_number = a_row['Frame']
                x_min = min(a_row['xtl'], a_row['xbr'])
                x_max = max(a_row['xtl'], a_row['xbr'])
                y_min = min(a_row['ytl'], a_row['ybr'])
                y_max = max(a_row['ytl'], a_row['ybr'])

                in_AOI = (
                        (gaze_df['Name'] == video_name) &
                        (gaze_df['Frame'] == frame_number) &
                        (gaze_df['Gaze Point X[px]'] >= x_min) &
                        (gaze_df['Gaze Point X[px]'] <= x_max) &
                        (gaze_df['Gaze Point Y[px]'] >= y_min) &
                        (gaze_df['Gaze Point Y[px]'] <= y_max)
                )
                gaze_df.loc[in_AOI, f'inAOI_{label}'] = True
        # Merge the 'condition' column from additional_df to gaze_df
        gaze_df = gaze_df.merge(additional_df[['Name', 'Frame', 'Condition']].drop_duplicates(),
                                on=['Name', 'Frame'], how='left')
        # Columns to keep after merging
        columns_to_keep = ['Name', 'Recording Time Stamp[ms]', 'Triggle Receive', 'Gaze Point X[px]',
                           'Gaze Point Y[px]', 'Fixation Point X[px]', 'Fixation Point Y[px]', 'Frame', 'Condition']
        columns_to_keep.extend([f'inAOI_{label}' for label in unique_labels])

        # Save the merged dataframe
        merged_path = os.path.join(save_path, gaze_file.replace('.csv', '_merged.csv'))
        gaze_df[columns_to_keep].to_csv(merged_path, index=False)
        print(f"Merged data saved to {merged_path}")


# Example usage
# gaze_folder = "M:\\EEG_DATA\\EEG_data_0410\\EEG\\EXP_new"
# eprime_folder = "M:\\EEG_DATA\\EEG_data_0410\\Eprime\\EXPERT"
gaze_folder = "M:\\EEG_DATA\\EEG_data_0410\\EEG\\NOV_new"
eprime_folder = "M:\\EEG_DATA\\EEG_data_0410\\Eprime\\Novice"
save_path = "M:\\EEG_DATA\\EEG_data_0410\\full_AOI_Output"


gaze_date_info = preprocess_gaze_files(gaze_folder)
eprime_date_info = preprocess_eprime_files(eprime_folder)
# print(gaze_date_info)
# print(eprime_date_info)

single_AOI_path = "M:\\EEG_DATA\\EEG_data_0410\\new_full_AOI.csv"

gaze_eprime_mapping = match_files(gaze_date_info, eprime_date_info)
merge_files(gaze_folder, eprime_folder, gaze_eprime_mapping, save_path, single_AOI_path)

# Process the mapping
# for gaze_file, eprime_file in gaze_eprime_mapping.items():
#     print(f"Gaze file: {gaze_file} -> Eprime file: {eprime_file}")

