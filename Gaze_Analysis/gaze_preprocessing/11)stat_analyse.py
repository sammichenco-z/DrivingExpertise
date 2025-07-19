import pandas as pd

# Define the functions for Animation and Label_Condition
def determine_animation(name):
    return 'Animated' if 'bd' in name else 'Real'

def determine_label_condition(name, label):
    if 'bd' in name:
        if 'bdFixedNull' in name and 'noveh' in name:
            return 'Control'
        elif 'bdFixedNull' in name:
            return 'HazardOnly'
        elif 'noveh' in name:
            return 'OcclusionOnly'
        else:
            return 'OccludedHazard'
    else:
        conditions_map = {
            "['Vehicle-OHR']": 'OccludedHazard',
            "['Occlusion-OR']": 'OcclusionOnly',
            "['Turn-CR']": 'Control',
            "['Vehicle-HR']": 'HazardOnly'
        }
        return conditions_map.get(label, 'Unknown')

# Load your data
df = pd.read_csv('M:\\EEG_DATA\\EEG_data_0410\\FInal_Output\\Merged_summary_results.csv')

# Pre-processing
df = df[(df['Number_of_Fixations'] != 0) & (df['Label'] != '[]')]
df['Animation'] = df['Name'].apply(determine_animation)
df['Label_Condition'] = df.apply(lambda row: determine_label_condition(row['Name'], row['Label']), axis=1)
df['Condition'] = df['Animation'] + '_' + df['Label_Condition']

# Group by and calculate directly
grouped = df.groupby(['File', 'Expertise', 'Condition']).agg(
    Total_Rows=('Number_of_Fixations', 'size'),  # Count rows in each group
    Num_In_AOI_Fixation=('Number_of_In_AOI_Fixation', lambda x: (x > 0).sum())  # Count rows > 0
).reset_index()

# Calculate watched_AOI_percentage
grouped['Watched_AOI_Percentage'] = grouped['Num_In_AOI_Fixation'] / grouped['Total_Rows']

# Pivot the DataFrame to have one row per File and conditions as columns
pivot_total_rows = grouped.pivot_table(index=['File', 'Expertise'], columns='Condition', values='Total_Rows').fillna(0)
pivot_num_in_aoi = grouped.pivot_table(index=['File', 'Expertise'], columns='Condition', values='Num_In_AOI_Fixation').fillna(0)
pivot_watched_aoi_percentage = grouped.pivot_table(index=['File', 'Expertise'], columns='Condition', values='Watched_AOI_Percentage').fillna(0)

# Save each DataFrame to a separate CSV file
output_folder = 'M:\\EEG_DATA\\EEG_data_0410\\Final_Output\\'
pivot_total_rows.to_csv(output_folder + 'total_rows_summary_data.csv')
pivot_num_in_aoi.to_csv(output_folder + 'num_in_aoi_fixation_summary_data.csv')
pivot_watched_aoi_percentage.to_csv(output_folder + 'watched_aoi_percentage_summary_data.csv')