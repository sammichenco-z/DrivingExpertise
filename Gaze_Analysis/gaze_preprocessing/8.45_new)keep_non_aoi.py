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


def process_video_groups(df, file_name, condition, output_log_folder):
    """
    处理视频组并记录异常情况，保留non_aoi的视频
    """
    print(f"\nProcessing file: {file_name}")
    print(f"Initial data shape: {df.shape}")

    # 按视频分组
    video_groups = df.groupby('Video')
    valid_groups = []  # 存储有效的数据组
    non_fixation_videos = []  # 所有采样点都是-1的视频

    # 检查每个视频组
    for video_name, group in video_groups:
        # 只过滤掉完全没有注视的视频（所有值都是-1）
        if all(group['in_aoi'] == -1):
            print(f"  - Video {video_name}: All values are -1, excluding")
            non_fixation_videos.append(video_name)
        else:
            print(f"  - Video {video_name}: Valid data, including")
            valid_groups.append(group)

    # 记录统计信息
    print(f"\nSummary for {file_name}:")
    print(f"Total videos: {len(video_groups)}")
    print(f"Non-fixation videos: {len(non_fixation_videos)}")
    print(f"Valid videos: {len(valid_groups)}")

    # 保存非注视视频记录
    if non_fixation_videos:
        file_non_fixation_path = os.path.join(output_log_folder, f"{os.path.splitext(file_name)[0]}_non_fixation.csv")
        pd.DataFrame({'Video': non_fixation_videos}).to_csv(file_non_fixation_path, index=False)

        # 更新条件级别的记录
        condition_non_fixation_path = os.path.join(output_log_folder, f"{condition}_non_fixation.csv")
        with open(condition_non_fixation_path, 'a') as f:
            f.write(f"{file_name},{len(non_fixation_videos)}\n")

    # 返回有效的数据
    if valid_groups:
        result = pd.concat(valid_groups, ignore_index=True)
        print(f"Final processed data shape: {result.shape}")
        return result
    else:
        print("Warning: No valid data after processing")
        return pd.DataFrame()


def calculate_aoi_metrics(group):
    """
    计算单个视频组的AOI指标，正确处理-1的情况并修正持续时间计算
    """
    # 将数据转换为numpy数组以提高效率
    group = group.sort_values('Recording Time Stamp[ms]')
    timestamps = group['Recording Time Stamp[ms]'].values
    aoi_values = group['in_aoi'].values

    # 初始化新列
    group['aoi_entry_count'] = -1
    group['aoi_duration'] = -1
    group['time_to_first_aoi'] = -1

    # 如果数据为空，直接返回
    if len(group) == 0:
        return group

    # 初始化计数器和时间记录
    entry_count = 0
    current_entry_start = None
    first_aoi_found = False
    start_time = timestamps[0]
    in_aoi = False
    duration_segments = []  # 存储同一次进入的多个时间段
    segment_start = None  # 每段的开始时间

    # 处理第一行
    if aoi_values[0] == 1:
        entry_count = 1
        segment_start = timestamps[0]
        group.iloc[0, group.columns.get_loc('aoi_entry_count')] = 1
        group.iloc[0, group.columns.get_loc('time_to_first_aoi')] = 0
        first_aoi_found = True
        in_aoi = True

    # 遍历数据计算指标
    for i in range(1, len(group)):
        curr_value = aoi_values[i]
        prev_value = aoi_values[i - 1]

        # 处理进入AOI的情况
        if curr_value == 1:
            # 如果之前不在AOI内且前一个值为0（不是-1），这是新的entry
            if not in_aoi and prev_value == 0:
                entry_count += 1

            # 如果之前不在AOI内（无论前一个值是0还是-1），都需要记录开始时间
            if not in_aoi:
                segment_start = timestamps[i]

                if not first_aoi_found:
                    group['time_to_first_aoi'] = timestamps[i] - start_time
                    first_aoi_found = True

            in_aoi = True
            group.iloc[i, group.columns.get_loc('aoi_entry_count')] = entry_count

            # 如果是最后一行，计算这一段的持续时间
            if i == len(group) - 1:
                duration = timestamps[i] - segment_start
                duration_segments.append(duration)

        # 处理离开AOI的情况
        elif curr_value != 1 and in_aoi:
            # 计算这一段的持续时间（使用前一个时间点）
            duration = timestamps[i - 1] - segment_start
            duration_segments.append(duration)

            # 如果是真正离开AOI（curr_value为0），而不是临时的-1
            if curr_value == 0:
                # 计算总持续时间并更新
                total_duration = sum(duration_segments)
                mask = (group['aoi_entry_count'] == entry_count)
                group.loc[mask, 'aoi_duration'] = total_duration
                duration_segments = []  # 清空segments列表

            in_aoi = False

    # 如果还有未处理的持续时间段
    if duration_segments:
        total_duration = sum(duration_segments)
        mask = (group['aoi_entry_count'] == entry_count)
        group.loc[mask, 'aoi_duration'] = total_duration

    return group

def process_data_with_metrics(input_data, output_base_folder):
    """
    处理所有条件的数据，保持原有的文件组织结构
    """
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_base_folder, f"aoi_metrics_{timestamp}")

    # 处理每个组和条件
    processed_data = {}
    for group in ['Exp', 'Nov']:
        print(f"\nProcessing group: {group}")
        group_folder = os.path.join(output_folder, group)

        for condition in ['virtual_control_hazard', 'virtual_control_occlusion',
                          'real_control_hazard', 'real_control_occlusion']:
            print(f"\nProcessing condition: {condition}")

            # 创建条件文件夹
            condition_folder = os.path.join(group_folder, condition)
            os.makedirs(condition_folder, exist_ok=True)

            # 获取该组和条件的数据
            key = f"{group}_{condition}"
            if key in input_data and not input_data[key].empty:
                data = input_data[key]

                # 按文件名（参与者）分组处理
                for participant, participant_data in data.groupby('Participant'):
                    print(f"Processing participant: {participant}")

                    # 按视频分组处理数据
                    processed_groups = []
                    for video_name, video_group in participant_data.groupby('Video'):
                        processed_group = calculate_aoi_metrics(video_group)
                        processed_groups.append(processed_group)

                    # 合并该参与者的所有处理后的数据
                    if processed_groups:
                        participant_processed = pd.concat(processed_groups, ignore_index=True)

                        # 保存到对应的文件
                        output_file = os.path.join(condition_folder, f"{participant}.csv")
                        participant_processed.to_csv(output_file, index=False)
                        print(f"  - Saved: {os.path.basename(output_file)}")

                        # 存储到processed_data字典中
                        if key not in processed_data:
                            processed_data[key] = {}
                        processed_data[key][participant] = participant_processed
            else:
                print(f"No data found for {group} - {condition}")

    print(f"\nProcessing completed.")
    print(f"Results are saved in folder: {output_folder}")
    print("The folder structure maintains the original organization:")
    print(f"  {output_folder}")
    print(f"  ├── Exp")
    print(f"  │   ├── virtual_control_hazard")
    print(f"  │   ├── virtual_control_occlusion")
    print(f"  │   ├── real_control_hazard")
    print(f"  │   └── real_control_occlusion")
    print(f"  └── Nov")
    print(f"      ├── virtual_control_hazard")
    print(f"      ├── virtual_control_occlusion")
    print(f"      ├── real_control_hazard")
    print(f"      └── real_control_occlusion")

    return processed_data, output_folder


def main():
    # 设置路径
    input_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\trial_with_aoi_processed"
    output_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\aoi_analysis_results_with_non_aoi"
    output_log_folder = os.path.join(output_base_folder, "aoi_analysis_logs")

    # 确保输出文件夹存在
    os.makedirs(output_log_folder, exist_ok=True)

    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 存储所有有效数据，按组和条件组织
    valid_data = {}
    conditions = ['virtual_control_hazard', 'virtual_control_occlusion',
                  'real_control_hazard', 'real_control_occlusion']

    # 处理每个组和条件
    for group in ['Exp', 'Nov']:
        print(f"\nProcessing {group} group...")

        for condition in conditions:
            print(f"\nProcessing condition: {condition}")

            input_dir = os.path.join(input_base_folder, group, condition)
            if not os.path.exists(input_dir):
                print(f"Directory not found: {input_dir}")
                continue

            # 创建条件特定的日志文件夹
            condition_log_dir = os.path.join(output_log_folder, group, condition)
            os.makedirs(condition_log_dir, exist_ok=True)

            # 存储该组和条件的所有有效数据
            key = f"{group}_{condition}"
            valid_data[key] = []

            # 处理该条件下的所有文件
            for file in os.listdir(input_dir):
                if file.endswith('.csv'):
                    print(f"Processing file: {file}")

                    # 读取数据
                    file_path = os.path.join(input_dir, file)
                    df = pd.read_csv(file_path)

                    # 添加participant信息（从文件名获取）
                    df['Participant'] = file.replace('.csv', '')

                    # 处理视频组并获取有效数据
                    valid_df = process_video_groups(df, file, condition, condition_log_dir)

                    if not valid_df.empty:
                        valid_data[key].append(valid_df)

    # 合并每个条件的有效数据
    final_data = {}
    for key in valid_data:
        if valid_data[key]:
            final_data[key] = pd.concat(valid_data[key], ignore_index=True)

    # 处理数据并保存结果
    processed_data, output_folder = process_data_with_metrics(final_data, output_base_folder)

    print(f"\nProcessing completed. Results saved in: {output_folder}")
    return processed_data


if __name__ == "__main__":
    processed_data = main()