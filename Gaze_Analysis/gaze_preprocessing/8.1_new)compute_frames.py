import pandas as pd
import os
import numpy as np


def calculate_frame_number(time_ms):
    """
    计算帧数
    1000ms = 30帧
    从0开始计数
    """
    # 计算完整的秒数和剩余毫秒数
    total_seconds = time_ms // 1000
    remaining_ms = time_ms % 1000

    # 计算完整秒对应的帧数
    frames_from_seconds = total_seconds * 30

    # 计算剩余毫秒对应的帧数
    frames_from_ms = int((remaining_ms / 1000) * 30)

    # 总帧数
    return frames_from_seconds + frames_from_ms


def process_trial_data(input_path, output_path):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 处理Exp和Nov文件夹
    for folder in ['Exp', 'Nov']:
        input_folder = os.path.join(input_path, folder)
        output_folder = os.path.join(output_path, folder)
        os.makedirs(output_folder, exist_ok=True)

        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            try:
                print(f"Processing {folder}/{csv_file}")

                # 读取CSV文件
                df = pd.read_csv(os.path.join(input_folder, csv_file))

                # 按Video分组处理
                processed_groups = []

                for video_name, group in df.groupby('Video'):
                    # 按时间戳排序
                    group_sorted = group.sort_values('Recording Time Stamp[ms]')

                    # 计算相对时间戳（从0开始）
                    min_timestamp = group_sorted['Recording Time Stamp[ms]'].min()
                    relative_timestamps = group_sorted['Recording Time Stamp[ms]'] - min_timestamp

                    # 计算帧数
                    group_sorted['Frame'] = relative_timestamps.apply(calculate_frame_number)

                    processed_groups.append(group_sorted)

                # 合并所有处理后的组
                processed_df = pd.concat(processed_groups)

                # 保存处理后的文件
                output_file = os.path.join(output_folder, csv_file)
                processed_df.to_csv(output_file, index=False)

                # 打印基本统计信息
                print(f"File: {csv_file}")
                print("Video frame statistics:")
                frame_stats = processed_df.groupby('Video')['Frame'].agg(['min', 'max', 'count'])
                print(frame_stats)
                print("\n")

            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue


# 设置输入和输出路径
input_path = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output"
output_path = "M:\\EEG_DATA\\EEG_data_0410\\trial_only_output_with_frames"

# 运行处理
process_trial_data(input_path, output_path)