import pandas as pd
import os
import numpy as np


def standardize_condition_names(df):
    # 创建条件名称的映射字典
    condition_map = {
        'control': 'Control',
        'hazardonly': 'HazardOnly',
        'occlusiononly': 'OcclusionOnly',
        'CONTROL': 'Control',
        'HAZARDONLY': 'HazardOnly',
        'OCCLUSIONONLY': 'OcclusionOnly'
    }

    # 标准化条件名称
    df['Condition'] = df['Condition'].str.lower().map(condition_map)
    return df


def replace_control_aois(data_df, matches_df):
    # 创建新的DataFrame来存储结果
    result_df = pd.DataFrame()

    # 获取非Control条件的数据
    non_control_df = data_df[data_df['Condition'] != 'Control']
    result_df = pd.concat([result_df, non_control_df])

    # 对每个匹配对进行处理
    for _, row in matches_df.iterrows():
        control_video = row['control_video']
        comparison_video = row['comparison_video']

        # 获取comparison视频的AOI数据
        comparison_aois = data_df[data_df['Name'] == comparison_video].copy()

        # 修改AOI数据，将Name和Condition改为control视频的信息
        comparison_aois['Name'] = control_video
        comparison_aois['Condition'] = 'Control'

        # 添加到结果DataFrame
        result_df = pd.concat([result_df, comparison_aois])

    # 排序并重置索引
    result_df = result_df.sort_values(['Name', 'Image ID']).reset_index(drop=True)

    return result_df


def match_control_comparison(df, group_name):
    # 分离control组和对照组
    control_df = df[df['Condition'] == 'Control']
    comparison_df = df[df['Condition'] != 'Control']

    # 获取唯一的视频名称
    control_videos = control_df['Name'].unique()
    comparison_videos = comparison_df['Name'].unique()

    # 检查两组数量是否相等
    if len(control_videos) != len(comparison_videos):
        raise ValueError(f"Unequal number of videos in {group_name}:\n"
                         f"Control videos: {len(control_videos)}\n"
                         f"Comparison videos: {len(comparison_videos)}\n"
                         f"Control videos: {sorted(control_videos)}\n"
                         f"Comparison videos: {sorted(comparison_videos)}")

    # 随机打乱对照组顺序
    comparison_videos_shuffled = np.array(comparison_videos)
    np.random.shuffle(comparison_videos_shuffled)

    # 创建匹配对
    matches = pd.DataFrame({
        'control_video': control_videos,
        'comparison_video': comparison_videos_shuffled
    })

    return matches


def process_aoi_data(input_file, output_dir, random_seed=42):
    # 设置随机种子
    np.random.seed(random_seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取AOI数据并标准化条件名称
    df = pd.read_csv(input_file)
    df = standardize_condition_names(df)

    # 分离virtual和real数据
    virtual_df = df[df['Name'].str.contains('bd', case=False)]
    real_df = df[~df['Name'].str.contains('bd', case=False)]

    # 为virtual数据创建不同对比组
    virtual_control_hazard = virtual_df[virtual_df['Condition'].isin(['Control', 'HazardOnly'])]
    virtual_control_occlusion = virtual_df[virtual_df['Condition'].isin(['Control', 'OcclusionOnly'])]

    # 为real数据创建不同对比组
    real_control_hazard = real_df[real_df['Condition'].isin(['Control', 'HazardOnly'])]
    real_control_occlusion = real_df[real_df['Condition'].isin(['Control', 'OcclusionOnly'])]

    # 创建匹配对并保存
    data_groups = {
        'virtual_control_hazard': virtual_control_hazard,
        'virtual_control_occlusion': virtual_control_occlusion,
        'real_control_hazard': real_control_hazard,
        'real_control_occlusion': real_control_occlusion
    }

    matches_dict = {}
    processed_data_dict = {}

    # 保存分组数据和创建匹配对
    for name, data in data_groups.items():
        try:
            # 创建匹配对
            matches = match_control_comparison(data, name)
            matches_dict[name] = matches

            # 替换Control条件的AOI数据
            processed_data = replace_control_aois(data, matches)
            processed_data_dict[name] = processed_data

            # 保存处理后的数据
            output_file = os.path.join(output_dir, f"{name}_processed.csv")
            processed_data.to_csv(output_file, index=False)

            # 保存匹配信息
            matches_output_file = os.path.join(output_dir, f"{name}_matches.csv")
            matches.to_csv(matches_output_file, index=False)

            # 打印基本信息
            print(f"\n{name}:")
            print(f"Total rows in processed data: {len(processed_data)}")
            print("Conditions present:", processed_data['Condition'].unique())
            print("Unique videos:", processed_data['Name'].nunique())
            print(f"Matched pairs: {len(matches)}")

            # 打印每个视频的AOI数量
            video_aoi_counts = processed_data.groupby(['Name', 'Condition']).size()
            print("\nAOI counts per video:")
            print(video_aoi_counts)

        except ValueError as e:
            print(f"\nError in {name}:")
            print(str(e))
            return None, None, None

    return data_groups, matches_dict, processed_data_dict


# 使用实际路径
input_file = "M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv"
output_dir = "M:\\EEG_DATA\\EEG_data_0410\\gen_comparison_aoi"

# 运行处理
data_groups, matches, processed_data = process_aoi_data(input_file, output_dir, random_seed=42)

if data_groups is None:
    print("\nProcessing stopped due to unequal group sizes.")