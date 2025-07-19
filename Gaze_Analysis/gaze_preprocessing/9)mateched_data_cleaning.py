import os
import pandas as pd


def clean_csv_file(file_path, save_path):
    df = pd.read_csv(file_path)

    # Remove rows without value in column 'xtl', or where 'Gaze Point X[px]' or 'Gaze Point Y[px]' equals -1
    df = df.dropna(subset=['xtl'])
    # df = df[(df['Gaze Point X[px]'] != -1) & (df['Gaze Point Y[px]'] != -1)]

    # Define a function to apply the cleaning rules per group
    def clean_group(group):
        # Find the first occurrence where values do not meet the conditions
        for col in ['xtl', 'xbr', 'ytl', 'ybr']:
            invalid_index = group[(group[col] < 0) |
                                  ((col in ['xtl', 'xbr']) & (group[col] > 1920)) |
                                  ((col in ['ytl', 'ybr']) & (group[col] > 1080))].index.min()
            if invalid_index is not pd.NA:  # Check if there's an invalid value
                group = group.loc[:invalid_index - 1]  # Keep rows before the invalid value
                break  # Break after the first truncation
        return group

    # Apply the cleaning rules to each group
    df = df.groupby('Name').apply(clean_group).reset_index(drop=True)

    # Save the cleaned data to the new path
    df.to_csv(save_path, index=False)


def clean_all_csv_files_in_folder(folder_path, new_base_folder_path):
    # Create the target folders in the new base folder path
    for sub_folder in ['Novice', 'Expert']:
        new_sub_folder_path = os.path.join(new_base_folder_path, sub_folder)
        os.makedirs(new_sub_folder_path, exist_ok=True)

    # Clean files in each sub-folder ('Novice' and 'Expert')
    for sub_folder in ['Novice', 'Expert']:
        sub_folder_path = os.path.join(folder_path, sub_folder)
        new_sub_folder_path = os.path.join(new_base_folder_path, sub_folder)

        # List all CSV files in the sub-folder
        for file_name in os.listdir(sub_folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(sub_folder_path, file_name)
                save_path = os.path.join(new_sub_folder_path, file_name)
                print(f"Cleaning file: {file_path}")
                clean_csv_file(file_path, save_path)
                print(f"File cleaned and saved to: {save_path}")

# Set the path for the folder containing 'Novice' and 'Expert' sub-folders
base_folder_path = 'M:\\EEG_DATA\\EEG_data_0410\\Output'  # Replace with your actual path
# Set the path for the new base folder where cleaned files will be stored
new_base_folder_path = 'M:\\EEG_DATA\\EEG_data_0410\\Cleaned_output'  # Replace with your new base path

clean_all_csv_files_in_folder(base_folder_path, new_base_folder_path)