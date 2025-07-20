import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess Excel files of EEG virtual data.')
    parser.add_argument('--folder_path', default='YOUR_FOLDER_PATH/Eprime/', type=str, required=True)
    return parser.parse_args()


# Map the sample order to video path
def sample2video(folder_path):
    dfs = []
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            for filename in os.listdir(sub_folder_path):
                path = os.path.join(sub_folder_path, filename)
                df = pd.read_excel(path)
                selected_df = df[['ExpTrials.Sample', 'Video']].dropna(subset=['ExpTrials.Sample']).reset_index(drop=True)
                selected_df['ExpTrials.Sample'] = selected_df['ExpTrials.Sample'].astype(int)
                if 'EXPERT' in path:
                    new_column_name = 'e_' + str(df['Subject'].iloc[0])
                elif 'NOVICE' in path:
                    new_column_name = 'n_' + str(df['Subject'].iloc[0])
                selected_df = selected_df.rename(columns={'Video': new_column_name, 'ExpTrials.Sample': 'sample'})
                dfs.append(selected_df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='sample', how='outer')

    merged_df.to_excel('info/sample2video.xlsx', index=False)
    print('Finish mapping video sample order to video path.')
    return merged_df

# Map the video path to sample order
def video2sample(folder_path):
    dfs = []
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            for filename in os.listdir(sub_folder_path):
                path = os.path.join(sub_folder_path, filename)
                df = pd.read_excel(path)
                selected_df = df[['Video', 'ExpTrials.Sample']].dropna(subset=['ExpTrials.Sample'])
                selected_df['ExpTrials.Sample'] = selected_df['ExpTrials.Sample'].astype(int)
                if 'EXPERT' in path:
                    new_column_name = 'e_' + str(df['Subject'].iloc[0])
                elif 'NOVICE' in path:
                    new_column_name = 'n_' + str(df['Subject'].iloc[0])
                selected_df = selected_df.rename(columns={'Video': 'video', 'ExpTrials.Sample': new_column_name}).sort_values(by='video').reset_index(drop=True)
                dfs.append(selected_df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='video', how='outer')

    merged_df.to_excel('info/video2sample.xlsx', index=False)
    print('Finish mapping video path to video sample order.')
    return merged_df


# Map the video path to category order
def video2order(s2v, v2s):
    for col in s2v.columns[1:]:
        def categorize_path(path):
            if 'real' in path:
                return 'real'
            elif 'noveh' in path and 'bdFixedNull' in path:
                return 'control'
            elif 'noveh' in path and 'bdFixedNull' not in path:
                return 'occlusion'
            elif 'noveh' not in path and 'bdFixedNull' in path:
                return 'hazard'
            else:
                return 'both'
        s2v[col] = s2v[col].apply(categorize_path)

    for col in s2v.columns[1:]:
        count_dict = {}
        for idx, value in s2v[col].items():
            if value == 'real':
                continue
            if value not in count_dict:
                count_dict[value] = 1
            else:
                count_dict[value] += 1
            s2v.at[idx, col] = f'{value}-{count_dict[value]}'

    s2v.set_index('sample', inplace=True)
    for col in v2s.columns[1:]:
        v2s[col] = v2s[col].apply(lambda x: s2v.loc[x, col] if x in s2v.index else x)
    v2s = v2s[~v2s['video'].str.contains('real')].reset_index(drop=True)

    # one column name 'n_901401' may be wrong, it need to be renamed to 'n_91401'
    v2s = v2s.rename(columns={'n_901401': 'n_91401'})
    v2s.to_excel('info/video2order.xlsx', index=False)
    print('Finish mapping video path to video sample order of each category.')


if __name__ == "__main__":
    args = get_args()
    os.makedirs('info/', exist_ok=True)
    print('Make folder to record EEG data info.')

    s2v = sample2video(args.folder_path)
    v2s = video2sample(args.folder_path)
    video2order(s2v, v2s)