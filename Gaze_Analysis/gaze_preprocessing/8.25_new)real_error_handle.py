import pandas as pd
import os
import numpy as np
from datetime import datetime


def standardize_video_name(video_name):
    """
    标准化视频名称
    """
    # 需要添加n前缀的特殊视频列表
    n_prefix_list = [
        'oveh_sum_501', 'oveh_sum_2001', 'oveh_sum_1201',
        'oveh_sum_1401', 'oveh_sum_2301', 'oveh_sum_1801', 'ew nalati',
        'oveh_sum_1601', 'ew ireland', 'oveh_sum_201', 'ew gobi1',
        'oveh_sum_1501', 'oveh_sum_1301', 'oveh_sum_2101'
    ]

    # 特殊转换情况
    if video_name == '4Ksc':
        return '4k sc'
    elif video_name == 'alati part3':
        return 'natila part3'
    elif video_name == 'oveh_sum_1 01':
        return 'noveh_sum 101'
    elif video_name == 'oveh_sum_401':
        return 'noveh_num_401'

    # hazard sup -> hazard supp 的转换
    if video_name in ['hazard sup 1', 'hazard sup 2', 'hazard sup 3']:
        video_name = video_name.replace('sup', 'supp')

    # 添加n前缀
    if video_name in n_prefix_list:
        video_name = 'n' + video_name

    return video_name

def is_real_video(video_name):
    """
    检查是否为real条件的视频
    """
    return 'bd' not in video_name.lower()


def save_matching_details(match_info, output_file):
    """
    保存详细的匹配信息
    """
    records = []

    # 只记录直接匹配和未匹配的AOI视频
    # 记录直接匹配的视频
    for video in match_info['direct_matches']:
        records.append({
            'condition': match_info['condition'],
            'video_name': video,
            'match_type': 'direct_match'
        })

    # 只记录相关条件的未匹配AOI视频
    for video in match_info['unmatched_aoi']:
        records.append({
            'condition': match_info['condition'],
            'video_name': video,
            'match_type': 'unmatched_aoi'
        })

    # 保存记录
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    return df


def analyze_matching(trial_data, aoi_data, condition_name):
    """
    分析视频匹配情况，只处理real视频并根据条件筛选
    """
    # 获取当前条件对应的视频类型
    if 'hazard' in condition_name.lower():
        valid_conditions = ['Control', 'HazardOnly']
    else:  # occlusion
        valid_conditions = ['Control', 'OcclusionOnly']

    # 获取AOI中的所有相关视频
    relevant_aoi_videos = set(aoi_data[aoi_data['Condition'].isin(valid_conditions)]['Name'].unique())
    print(f"\nTotal relevant AOI videos: {len(relevant_aoi_videos)}")
    print("AOI videos by condition:")
    for cond in valid_conditions:
        videos = set(aoi_data[aoi_data['Condition'] == cond]['Name'].unique())
        print(f"{cond}: {len(videos)} videos")

    # 获取trial中的所有real视频并标准化名称
    trial_videos = set(standardize_video_name(video) for video in trial_data['Video'].unique()
                       if is_real_video(video))
    print(f"Total real trial videos: {len(trial_videos)}")

    # 直接匹配的视频
    direct_matches = trial_videos.intersection(relevant_aoi_videos)
    print(f"Direct matches: {len(direct_matches)}")

    # 只保存相关条件的未匹配AOI视频
    unmatched_aoi = set()
    for video in relevant_aoi_videos - direct_matches:
        video_condition = aoi_data[aoi_data['Name'] == video]['Condition'].iloc[0]
        if video_condition in valid_conditions:
            unmatched_aoi.add(video)

    print(f"Unmatched AOI videos (relevant conditions only): {len(unmatched_aoi)}")

    return {
        'condition': condition_name,
        'valid_conditions': valid_conditions,
        'relevant_aoi_videos': relevant_aoi_videos,
        'direct_matches': direct_matches,
        'unmatched_aoi': unmatched_aoi
    }


def process_single_file(trial_file_path, aoi_folder, output_base_folder, group_name):
    """
    处理单个文件的real条件
    """
    # 只读取real条件的AOI数据
    aoi_conditions = {
        'real_control_hazard': pd.read_csv(os.path.join(aoi_folder, 'real_control_hazard_processed.csv')),
        'real_control_occlusion': pd.read_csv(os.path.join(aoi_folder, 'real_control_occlusion_processed.csv'))
    }

    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join(output_base_folder, 'real_matching_logs')
    os.makedirs(log_folder, exist_ok=True)

    # 准备日志文件
    matching_details_log = os.path.join(log_folder, f'real_matching_details_{timestamp}.csv')
    matching_stats_log = os.path.join(log_folder, f'real_matching_stats_{timestamp}.csv')

    try:
        print(f"Processing file: {os.path.basename(trial_file_path)}")
        trial_data = pd.read_csv(trial_file_path)

        matching_stats = []
        all_matching_details = []

        # 对每个real条件进行处理
        for condition_name, aoi_data in aoi_conditions.items():
            print(f"\nAnalyzing condition: {condition_name}")

            # 分析匹配情况
            match_info = analyze_matching(trial_data, aoi_data, condition_name)

            # 保存匹配详情
            details_df = save_matching_details(match_info,
                                               os.path.join(log_folder,
                                                            f'real_matching_details_{condition_name}_{timestamp}.csv'))
            all_matching_details.append(details_df)

            # 记录匹配统计
            matching_stats.append({
                'condition': condition_name,
                'total_relevant_aoi_videos': len(match_info['relevant_aoi_videos']),
                'direct_matches': len(match_info['direct_matches']),
                'unmatched_aoi': len(match_info['unmatched_aoi'])
            })

            # 处理和保存匹配的数据
            if len(match_info['direct_matches']) > 0:
                merged_data = process_matched_data(trial_data, aoi_data, match_info)
                if not merged_data.empty:
                    output_dir = os.path.join(output_base_folder, group_name, condition_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, os.path.basename(trial_file_path))
                    merged_data.to_csv(output_file, index=False)
                    print(f"Saved {len(merged_data)} rows for {condition_name}")

        # 保存总体统计信息
        pd.DataFrame(matching_stats).to_csv(matching_stats_log, index=False)
        print(f"\nSaved matching statistics to: {os.path.basename(matching_stats_log)}")

        # 保存所有匹配详情
        pd.concat(all_matching_details).to_csv(matching_details_log, index=False)
        print(f"Saved detailed matching information to: {os.path.basename(matching_details_log)}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise e


def process_matched_data(trial_data, aoi_data, matching_info):
    """
    处理匹配的数据
    """
    # 只处理real视频
    trial_data_copy = trial_data[trial_data['Video'].apply(is_real_video)].copy()

    # 重命名Frame列
    if 'Frame' in trial_data_copy.columns:
        trial_data_copy = trial_data_copy.rename(columns={'Frame': 'Image ID'})

    # 标准化视频名称
    trial_data_copy['Video'] = trial_data_copy['Video'].apply(standardize_video_name)

    # 只保留匹配的视频
    trial_data_copy = trial_data_copy[trial_data_copy['Video'].isin(matching_info['direct_matches'])]

    # 合并数据
    merged_data = pd.merge(
        trial_data_copy,
        aoi_data[['Name', 'Image ID', 'xtl', 'ytl', 'xbr', 'ybr', 'Condition']],
        left_on=['Video', 'Image ID'],
        right_on=['Name', 'Image ID'],
        how='inner'
    ).drop('Name', axis=1)

    return merged_data

def main():
    # 设置路径
    base_input_path = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output_with_frames"
    aoi_folder = "M:\\EEG_DATA\\EEG_data_0410\\gen_comparison_aoi"
    output_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\trial_with_aoi"

    # 获取第一个文件进行处理
    group_name = 'Exp'
    input_folder = os.path.join(base_input_path, group_name)
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if csv_files:
        first_file = os.path.join(input_folder, csv_files[0])
        print(f"Processing first file: {csv_files[0]}")
        process_single_file(first_file, aoi_folder, output_base_folder, group_name)
    else:
        print("No CSV files found in the input folder")


if __name__ == "__main__":
    main()