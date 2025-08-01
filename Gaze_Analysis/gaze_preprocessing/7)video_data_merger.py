import os
import re
import pandas as pd


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

def merge_files(gaze_folder, eprime_folder, gaze_eprime_mapping, save_path):
    for gaze_file, eprime_file in gaze_eprime_mapping.items():
        print("Processing ", gaze_file, " with ", eprime_file)
        gaze_path = os.path.join(gaze_folder, gaze_file)
        eprime_path = os.path.join(eprime_folder, eprime_file)

        # Read gaze and Eprime files
        gaze_df = pd.read_csv(gaze_path)
        eprime_df = pd.read_excel(eprime_path)

        # Remove rows with empty ExpClips.RESP in Eprime
        eprime_df = eprime_df.dropna(subset=["ExpClips.DEVICE"])

        gaze_df = gaze_df[gaze_df['Recording Time Stamp[ms]'].notna()]

        num_trials = 0
        triggle_value_list = []
        trial_started = False
        # Process gaze file
        for index, row in gaze_df.iterrows():
            triggle_value = row["Triggle Receive"]

            if pd.notna(triggle_value) and '20' in str(triggle_value):
                start_row = index
                trial_started = True
            elif pd.notna(triggle_value) and '21' in str(triggle_value) and triggle_value != 21.0:
                triggle_value_list.append(triggle_value)
                if not trial_started:
                    print("ERROR found at index ", index)
                trial_started = False
                end_row = index
                if num_trials < len(eprime_df):
                    # Get corresponding Eprime data
                    eprime_row = eprime_df.iloc[num_trials]
                    # video_value = eprime_row["Video"]
                    video_value = os.path.splitext(os.path.basename(eprime_row["Video"]))[0]  # extract the video name
                    # print("Eprime row ", video_value)
                    resp_value = eprime_row["ExpClips.RESP"]
                    # Assign values to gaze file
                    gaze_df.loc[start_row:end_row, "Video"] = video_value
                    gaze_df.loc[start_row:end_row, "ExpClips.RESP"] = resp_value

                    num_trials += 1
                else:
                    num_trials += 1
                    print(triggle_value_list)
                    print(f"Error: Eprime data not available for trial {num_trials} at row {index} with length {len(eprime_df)}")


        # if num_trials != 360:
        #     print(f"Number of trials for {gaze_file}: {num_trials}")
        print(f"Number of trials for {gaze_file}: {num_trials}")


        # Save merged dataframe
        merged_path = save_path + "\\" + gaze_file
        gaze_df.to_csv(merged_path, index=False)
        print(f"Merged data saved to {merged_path}")


# Example usage
gaze_folder = "M:\\EEG_DATA\\EEG_data_0410\\EEG\\EXP_new"
eprime_folder = "M:\\EEG_DATA\\EEG_data_0410\\Eprime\\EXPERT"
# gaze_folder = "M:\\EEG_DATA\\EEG_data_0410\\EEG\\Nov_new"
# eprime_folder = "M:\\EEG_DATA\\EEG_data_0410\\Eprime\\NOVICE"
save_path = "M:\\EEG_DATA\\EEG_data_0410\\Output"

gaze_date_info = preprocess_gaze_files(gaze_folder)
eprime_date_info = preprocess_eprime_files(eprime_folder)
# print(gaze_date_info)
# print(eprime_date_info)

gaze_eprime_mapping = match_files(gaze_date_info, eprime_date_info)
merge_files(gaze_folder, eprime_folder, gaze_eprime_mapping, save_path)

# Process the mapping
# for gaze_file, eprime_file in gaze_eprime_mapping.items():
#     print(f"Gaze file: {gaze_file} -> Eprime file: {eprime_file}")

