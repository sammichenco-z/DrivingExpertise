import pandas as pd
import os
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def standardize_video_name(video_name):
    """
    标准化视频名称
    """
    # 使用字典进行特殊转换，比if-elif更快
    special_cases = {
        '4Ksc': '4k sc',
        'alati part3': 'natila part3',
        'oveh_sum_1 01': 'noveh_sum 101',
        'oveh_sum_401': 'noveh_num_401',
        'hazard sup 1': 'hazard supp 1',
        'hazard sup 2': 'hazard supp 2',
        'hazard sup 3': 'hazard supp 3'
    }

    # 需要添加n前缀的特殊视频列表转换为集合，提高查找速度
    n_prefix_list = {
        'oveh_sum_501', 'oveh_sum_2001', 'oveh_sum_1201',
        'oveh_sum_1401', 'oveh_sum_2301', 'oveh_sum_1801', 'ew nalati',
        'oveh_sum_1601', 'ew ireland', 'oveh_sum_201', 'ew gobi1',
        'oveh_sum_1501', 'oveh_sum_1301', 'oveh_sum_2101'
    }

    # 特殊转换情况
    if video_name in special_cases:
        return special_cases[video_name]

    # 添加n前缀
    if video_name in n_prefix_list:
        video_name = 'n' + video_name

    return video_name


def is_real_video(video_name):
    return 'bd' not in video_name.lower()


def is_virtual_video(video_name):
    return 'bd' in video_name.lower()


def analyze_matching(trial_data, aoi_data, condition_name):
    """
    分析视频匹配情况
    """
    valid_conditions = ['Control', 'HazardOnly'] if 'hazard' in condition_name.lower() else ['Control', 'OcclusionOnly']

    # 使用向量化操作替代集合操作
    relevant_aoi_mask = aoi_data['Condition'].isin(valid_conditions)
    relevant_aoi_videos = set(aoi_data.loc[relevant_aoi_mask, 'Name'].unique())

    # 标准化视频名称
    trial_videos = set(trial_data['Video'].apply(standardize_video_name).unique())

    # 直接匹配的视频
    direct_matches = trial_videos.intersection(relevant_aoi_videos)

    # 只在有未匹配视频时计算未匹配
    unmatched_aoi = set()
    if len(direct_matches) < len(relevant_aoi_videos):
        unmatched_mask = ~aoi_data['Name'].isin(direct_matches) & relevant_aoi_mask
        unmatched_aoi = set(aoi_data.loc[unmatched_mask, 'Name'].unique())

    return {
        'condition': condition_name,
        'direct_matches': direct_matches,
        'unmatched_aoi': unmatched_aoi
    }


def process_matched_data(trial_data, aoi_data, matching_info):
    """
    处理匹配的数据
    """
    trial_data = trial_data.copy()
    if 'Frame' in trial_data.columns:
        trial_data.rename(columns={'Frame': 'Image ID'}, inplace=True)

    # 使用向量化操作
    trial_data['Video'] = trial_data['Video'].apply(standardize_video_name)
    trial_data = trial_data[trial_data['Video'].isin(matching_info['direct_matches'])]

    # 优化合并操作
    merged_data = pd.merge(
        trial_data,
        aoi_data[['Name', 'Image ID', 'xtl', 'ytl', 'xbr', 'ybr', 'Condition']],
        left_on=['Video', 'Image ID'],
        right_on=['Name', 'Image ID'],
        how='inner'
    ).drop('Name', axis=1)

    return merged_data


def process_single_file(args):
    """
    处理单个文件
    """
    file_path, input_folder, aoi_data_dict, output_base_folder, group_name = args

    try:
        csv_file = os.path.basename(file_path)
        print(f"Processing: {group_name}/{csv_file}")

        trial_data = pd.read_csv(file_path)

        # Real条件处理
        real_data = trial_data[trial_data['Video'].apply(is_real_video)].copy()
        if not real_data.empty:
            for condition, aoi_data in aoi_data_dict['real'].items():
                match_info = analyze_matching(real_data, aoi_data, condition)
                if match_info['direct_matches']:  # 只在有匹配时处理
                    merged_data = process_matched_data(real_data, aoi_data, match_info)
                    if not merged_data.empty:
                        output_dir = os.path.join(output_base_folder, group_name, condition)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, csv_file)
                        merged_data.to_csv(output_file, index=False)
                        print(f"Saved {len(merged_data)} rows for {condition}")

        # Virtual条件处理
        virtual_data = trial_data[trial_data['Video'].apply(is_virtual_video)].copy()
        if not virtual_data.empty:
            for condition, aoi_data in aoi_data_dict['virtual'].items():
                match_info = analyze_matching(virtual_data, aoi_data, condition)
                if match_info['direct_matches']:  # 只在有匹配时处理
                    merged_data = process_matched_data(virtual_data, aoi_data, match_info)
                    if not merged_data.empty:
                        output_dir = os.path.join(output_base_folder, group_name, condition)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, csv_file)
                        merged_data.to_csv(output_file, index=False)
                        print(f"Saved {len(merged_data)} rows for {condition}")

        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return False


def main():
    # 设置路径
    base_input_path = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output_with_frames"
    aoi_folder = "M:\\EEG_DATA\\EEG_data_0410\\gen_comparison_aoi"
    output_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\trial_with_aoi"

    # 预先读取所有AOI数据
    aoi_data_dict = {
        'real': {
            'real_control_hazard': pd.read_csv(os.path.join(aoi_folder, 'real_control_hazard_processed.csv')),
            'real_control_occlusion': pd.read_csv(os.path.join(aoi_folder, 'real_control_occlusion_processed.csv'))
        },
        'virtual': {
            'virtual_control_hazard': pd.read_csv(os.path.join(aoi_folder, 'virtual_control_hazard_processed.csv')),
            'virtual_control_occlusion': pd.read_csv(
                os.path.join(aoi_folder, 'virtual_control_occlusion_processed.csv'))
        }
    }

    # 处理每个组
    for group in ['Exp', 'Nov']:
        print(f"\nProcessing {group} group...")
        input_folder = os.path.join(base_input_path, group)

        # 获取所有CSV文件
        csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

        # 准备并行处理的参数
        process_args = [(f, input_folder, aoi_data_dict, output_base_folder, group) for f in csv_files]

        # 使用进程池并行处理文件
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_single_file, process_args))

        print(f"Completed processing {group} group: {sum(results)}/{len(results)} files processed successfully")


if __name__ == "__main__":
    main()