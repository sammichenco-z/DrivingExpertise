import os
import pandas as pd

def process_fixation(fixation_group, fixation_index):
    fixation_group = fixation_group.sort_values(by='Recording Time Stamp[ms]')
    # Reset index to give an increasing index to each row
    fixation_group.reset_index(inplace=True)

    invalid_gaze_points = (fixation_group['Gaze Point X[px]'] == -1) | (fixation_group['Gaze Point Y[px]'] == -1)
    valid_groups = fixation_group[~invalid_gaze_points].copy()
    valid_groups['group'] = (valid_groups['index'].diff() > 1).cumsum()


    processed_data = pd.DataFrame()
    total_valid_fixation_time = 0
    total_time_in_aoi = 0
    number_of_in_aoi_fixations = 0
    sub_group_count = 0

    for _, group in valid_groups.groupby('group'):
        # print("Length vaild group ", len(group))
        group['subgroup'] = (group['inAOI'] != group['inAOI'].shift()).cumsum()

        for subgroup_index, subgroup in group.groupby('subgroup'):
            if len(subgroup) > 1:
                sub_group_count += 1
                # print("Length subgroup ", len(subgroup))
                valid_fixation_time = subgroup['Recording Time Stamp[ms]'].iloc[-1] - subgroup['Recording Time Stamp[ms]'].iloc[0]
                total_valid_fixation_time += valid_fixation_time

                if subgroup['inAOI'].iloc[0]:
                    total_time_in_aoi += valid_fixation_time
                    number_of_in_aoi_fixations += 1

                subgroup['subgroup_index'] = subgroup_index
                subgroup['fixation_index'] = fixation_index

                processed_data = pd.concat([processed_data, subgroup])
    return processed_data, total_valid_fixation_time, total_time_in_aoi, number_of_in_aoi_fixations, sub_group_count

def analyze_data(folder_path, processed_data_path, summary_data_path):
    data_type = 'Novice' if 'Novice' in folder_path else 'Expert' if 'Expert' in folder_path else 'Unknown'
    processed_data_subfolder_path = os.path.join(processed_data_path, data_type)
    summary_data_subfolder_path = os.path.join(summary_data_path, data_type)

    os.makedirs(processed_data_subfolder_path, exist_ok=True)
    os.makedirs(summary_data_subfolder_path, exist_ok=True)

    summary_results_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            print("Processing ", file_name)
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            all_processed_data_for_file = pd.DataFrame()

            for name, group in df.groupby('Name'):
                group = group.dropna(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'])
                total_valid_fixation_time = 0
                total_time_in_aoi = 0
                number_of_in_aoi_fixations = 0
                fixation_index = 0
                label = group['Label'].unique()

                for _, fixation_group in group.groupby(['Fixation Point X[px]', 'Fixation Point Y[px]']):
                    processed_data, valid_fixation_time, time_in_aoi, in_aoi_fixations, fixation_number = \
                        process_fixation(fixation_group, fixation_index)
                    total_valid_fixation_time += valid_fixation_time
                    total_time_in_aoi += time_in_aoi
                    number_of_in_aoi_fixations += in_aoi_fixations
                    fixation_index += 1


                    processed_data['total_subgroup_count'] = fixation_number
                    # processed_data['total_fixation_count'] = processed_data['total_fixation_count'].astype(int)
                    all_processed_data_for_file = pd.concat([all_processed_data_for_file, processed_data])

                in_aoi_percentage = (total_time_in_aoi / total_valid_fixation_time) if total_valid_fixation_time > 0 else 0

                summary_results_list.append({
                    'File': file_name,
                    'Name': name,
                    'Number_of_Fixations': fixation_index,
                    'Total_Valid_Fixation_Time': total_valid_fixation_time,
                    'Time_In_AOI': total_time_in_aoi,
                    'In_AOI_Percentage': in_aoi_percentage,
                    'Number_of_In_AOI_Fixation': number_of_in_aoi_fixations,
                    'Label': label
                })

            processed_data_file_path = os.path.join(processed_data_subfolder_path, f"{file_name[:-4]}_{name}_processed_data.csv")
            all_processed_data_for_file.to_csv(processed_data_file_path, index=False)
    # Convert the list of dictionaries to a DataFrame
    summary_results_df = pd.DataFrame(summary_results_list)
    summary_results_file_path = os.path.join(summary_data_subfolder_path, f"{data_type}_summary_results.csv")
    summary_results_df.to_csv(summary_results_file_path, index=False)

base_folder_path = 'M:\\EEG_DATA\\EEG_data_0410\\Cleaned_output'
processed_data_path = 'M:\\EEG_DATA\\EEG_data_0410\\random_assigned_AOI\\v1'  # 'M:\\EEG_DATA\\EEG_data_0410\\Final_Output\\processed_data\\new'
summary_data_path = 'M:\\EEG_DATA\\EEG_data_0410\\random_assigned_AOI\\v1'  # 'M:\\EEG_DATA\\EEG_data_0410\\Final_Output\\summary_data\\new'

for sub_folder in ['Novice', 'Expert']:
    sub_folder_path = os.path.join(base_folder_path, sub_folder)
    analyze_data(sub_folder_path, processed_data_path, summary_data_path)