import os
import argparse
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '{:.6e}'.format(x))


# define the channel and center
channel_list = ['AF3', 'AF4', 'CP1', 'CP2', 'CPZ', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                'FC1', 'FC2', 'FP1', 'FP2', 'FPZ', 'FT7', 'FT8', 'O1', 'O2', 'OZ', 
                'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'POZ', 'T7', 'T8']

center2time_dict = {'P1': [70, 100], 'N1': [100, 130],
                    'P150': [130, 150], 'N170': [150, 170],
                    'P200': [170, 200], 'P250': [200, 250], 'N300': [250, 350],
                    'P3a': [350, 430], 'P3b': [430, 550]}

center2frame_dict = {'P1': [2], 'N1': [3], 'P150': [4], 'N170': [4],
                     'P200': [5], 'P250': [6, 7], 'N300': [7, 8, 9, 10],
                     'P3a': [10, 11, 12], 'P3b': [13, 14, 15, 16]}


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess Excel files of EEG virtual data.')
    parser.add_argument('--data_path', default='YOUR_FOLDER_PATH/output/', type=str, required=True)
    parser.add_argument('--feature_path', default='YOUR_FOLDER_PATH/virtual_feature/', type=str, required=True)
    return parser.parse_args()


# get person-category-order-channel-center target data
def calculate_person_target(data_path):
    os.makedirs('data/', exist_ok=True)
    print('Calculate each person channel-center target data...')
    tasks = {'virtual_control': 'control', 
             'virtual_hazard only': 'hazard', 
             'virtual_occluded hazard': 'both',
             'virual_occlusion only': 'occlusion'}

    for person_id in os.listdir(data_path):
        person_path = os.path.join(data_path, person_id)
        print(f'Process Subject {person_id}.')
    
        for sub in tasks.keys():
            task_path = os.path.join(person_path, sub)
            for filename in os.listdir(task_path):
                path = os.path.join(task_path, filename)
                data = pd.read_csv(path)
                data.set_index('Time', inplace=True)
                data = data.T

                filtered_data = data[channel_list]
                filtered_data.index = pd.to_numeric(filtered_data.index, errors='coerce')
                filtered_data = filtered_data[(filtered_data.index >= 0.070) & (filtered_data.index <= 0.550)]
                filtered_data.index = (filtered_data.index * 1000).astype(int)

                result = pd.DataFrame(index=channel_list, columns=center2time_dict.keys())
                for center, (start_time, end_time) in center2time_dict.items():
                    range_data = filtered_data.loc[(filtered_data.index >= start_time) & (filtered_data.index < end_time)]
                    if not range_data.empty:
                        means = range_data.mean()
                        result[center] = means

                rank = str(int(filename.split('trial')[1].replace('.csv', '')))
                task = tasks[sub]
                result.to_csv(f'data/{person_id}-{task}-{rank}.csv')


# get video-channel-center target data
def calculate_video_target(mapping):
    os.makedirs('target/', exist_ok=True)
    print('Calculate each video channel-center target data...')

    for index, row in mapping.iterrows():
        video_data_expert = []
        video_data_novice = []
        print('Process Video', row['video'])

        for col in mapping.columns[1:]:
            person_id = col.split('_')[0] + '_0' + col.split('_')[1]
            rank = row[col]
            
            file_path = f'data/{person_id}-{rank}.csv'
            if os.path.exists(file_path):
                try:
                    person_file = pd.read_csv(file_path)
                    person_data = person_file.select_dtypes(include=[np.number]).values
                    if person_id[0] == 'e':
                        video_data_expert.append(person_data)
                    elif person_id[0] == 'n':
                        video_data_novice.append(person_data)
                except Exception as e:
                    print(f"    Error reading {file_path}: {e}")
            else:
                print(f"    File {file_path} not found.")
            
        video_name = row['video'].split('\\')[-1][:-4]
        video_mean_data_expert = np.mean(np.stack(video_data_expert), axis=0)
        mean_file_expert = pd.DataFrame(video_mean_data_expert, columns=person_file.columns[1:])
        mean_file_expert.insert(0, 'channel', person_file.iloc[:, 0])
        task = rank.split('-')[0]
        mean_file_expert.to_csv(f'target/expert-{task}-{video_name}.csv', index=False)
        video_mean_data_novice = np.mean(np.stack(video_data_novice), axis=0)
        mean_file_novice = pd.DataFrame(video_mean_data_novice, columns=person_file.columns[1:])
        mean_file_novice.insert(0, 'channel', person_file.iloc[:, 0])
        mean_file_novice.to_csv(f'target/novice-{task}-{video_name}.csv', index=False)


# get channel-center target data
def organize_y(input_folder):
    os.makedirs('y/', exist_ok=True)
    output_folder = 'y/'
    print('Organize each channel-center target data for training...')

    file_names = sorted([file_name for file_name in os.listdir(input_folder) if file_name.endswith('.csv')])
    for file_name in file_names:
        print(f'Process File {file_name}')
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            channel = row['channel']
            for col in df.columns[1:]:
                center = col
                value = row[col]
                new_file_name = f"{channel}-{center}.csv"
                new_file_path = os.path.join(output_folder, new_file_name)

                if not os.path.exists(new_file_path):
                    new_df = pd.DataFrame(columns=['video_name', 'expert', 'hazard', 'occlusion', 'value'])
                    new_df.to_csv(new_file_path, index=False)
                existing_df = pd.read_csv(new_file_path)

                video_name = file_name.split('-')[-1][:-5]
                if file_name.split('-')[0] == 'expert':
                    is_expert = 1
                elif file_name.split('-')[0] == 'novice':
                    is_expert = 0
                if file_name.split('-')[1] == 'both':
                    is_hazard, is_occlusion = 1, 1
                elif file_name.split('-')[1] == 'hazard':
                    is_hazard, is_occlusion = 1, 0
                elif file_name.split('-')[1] == 'occlusion':
                    is_hazard, is_occlusion = 0, 1
                elif file_name.split('-')[1] == 'control':
                    is_hazard, is_occlusion = 0, 0

                new_row = pd.DataFrame({'video_name': video_name, 
                                        'expert': is_expert,
                                        'hazard': is_hazard,
                                        'occlusion': is_occlusion,
                                        'value': [value]})
                new_df = pd.concat([existing_df, new_row], ignore_index=True)
                new_df.to_csv(new_file_path, index=False)


# get center input data
def organize_x(feature_path):
    os.makedirs('X/', exist_ok=True)
    print('Organize each channel-center input data for training...')

    data = pd.read_csv(f'{feature_path}/bd0_veh_4.5_60_3D_results.csv').drop('tick', axis=1)
    feature_names = ['video_name'] + [name + '-now'  for name in data.columns.tolist()] + \
                                    [name + '-past' for name in data.columns.tolist()]
    features_list = {}
    for center_id in center2frame_dict.keys():
        features_list[center_id] = pd.DataFrame(columns=feature_names)

    file_names = sorted([file_name for file_name in os.listdir(feature_path)])
    for file_name in file_names:
        print(f'Process File {file_name}')
        file_path = os.path.join(feature_path, file_name)
        data = pd.read_csv(file_path).drop('tick', axis=1)
        data = data.replace('No vehicle', 0).apply(pd.to_numeric, errors='coerce').values

        for center_id in center2frame_dict.keys():
            feature_now = np.mean(data[center2frame_dict[center_id]], axis=0).tolist()
            feature_past = np.mean(data[list(range(center2frame_dict[center_id][0]))], axis=0).tolist()
            new_row = [file_name[:-15]] + feature_now + feature_past
            features_list[center_id].loc[len(features_list[center_id])] = new_row

    for center_id in center2frame_dict.keys():
        features_list[center_id].to_csv(f'X/{center_id}.csv', index=False)


if __name__ == "__main__":
    args = get_args()
    calculate_person_target(args.data_path)
    mapping = pd.read_excel('info/video2order.xlsx')
    calculate_video_target(mapping)
    organize_y('target/')
    organize_x(args.feature_path)