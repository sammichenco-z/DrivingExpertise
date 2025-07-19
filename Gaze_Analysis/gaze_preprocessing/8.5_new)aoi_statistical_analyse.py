import pandas as pd
import os
import numpy as np
from datetime import datetime

def calculate_participant_statistics(file_path):
    """
    计算单个参与者在特定条件下所有视频的统计数据，分别计算Control和对比条件
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 存储Control和对比条件的统计数据
    control_stats = []
    comparison_stats = []
    total_videos = 0
    invalid_videos = 0

    print(f"\nProcessing file: {os.path.basename(file_path)}")

    # 按视频分组计算每个视频的AOI指标
    for video_name, video_data in df.groupby('Video'):
        total_videos += 1

        # 检查这个视频是否有效（是否有任何AOI注视）
        if all(video_data['in_aoi'] == -1):
            print(f"  Warning: Video {video_name} has no valid AOI data (all values are -1)")
            invalid_videos += 1
            continue

        # 计算这个视频的AOI统计数据
        stats = {
            'first_aoi_time': -1,
            'first_aoi_duration': -1,
            'total_entries': 0,
            'avg_duration': -1
        }

        # 获取首次进入AOI前的时间（第一个非-1值）
        valid_first_times = video_data[video_data['time_to_first_aoi'] != -1]['time_to_first_aoi']
        if not valid_first_times.empty:
            stats['first_aoi_time'] = valid_first_times.iloc[0]

        # 获取首个AOI内停留时间
        valid_durations = video_data[video_data['aoi_duration'] != -1]['aoi_duration']
        if not valid_durations.empty:
            stats['first_aoi_duration'] = valid_durations.iloc[0]

        # 计算进出AOI的总次数（最大entry值 + 1）
        valid_entries = video_data[video_data['aoi_entry_count'] != -1]['aoi_entry_count']
        if not valid_entries.empty:
            stats['total_entries'] = valid_entries.max() + 1

        # 计算平均停留时间
        if stats['total_entries'] > 0:
            total_duration = valid_durations.sum()
            stats['avg_duration'] = total_duration / stats['total_entries']

        # 根据Condition列的值决定将统计数据添加到哪个列表
        condition = video_data['Condition'].iloc[0]
        if stats['first_aoi_time'] != -1 or stats['first_aoi_duration'] != -1 or stats['total_entries'] > 0:
            if condition == 'Control':
                control_stats.append(stats)
            else:  # HazardOnly 或 OcclusionOnly
                comparison_stats.append(stats)

    # 计算统计结果
    result_stats = {}
    for metric in ['first_aoi_time', 'first_aoi_duration', 'total_entries', 'avg_duration']:
        # 计算Control组的统计值
        if control_stats:
            control_data = [s[metric] for s in control_stats if s[metric] != -1]
            if control_data:
                result_stats[f'{metric}_control_mean'] = np.mean(control_data)
                result_stats[f'{metric}_control_std'] = np.std(control_data)
            else:
                result_stats[f'{metric}_control_mean'] = -1
                result_stats[f'{metric}_control_std'] = -1
        else:
            result_stats[f'{metric}_control_mean'] = -1
            result_stats[f'{metric}_control_std'] = -1

        # 计算对比组的统计值
        if comparison_stats:
            comparison_data = [s[metric] for s in comparison_stats if s[metric] != -1]
            if comparison_data:
                result_stats[f'{metric}_comparison_mean'] = np.mean(comparison_data)
                result_stats[f'{metric}_comparison_std'] = np.std(comparison_data)
            else:
                result_stats[f'{metric}_comparison_mean'] = -1
                result_stats[f'{metric}_comparison_std'] = -1
        else:
            result_stats[f'{metric}_comparison_mean'] = -1
            result_stats[f'{metric}_comparison_std'] = -1

    # 添加视频计数信息
    result_stats['valid_control_videos'] = len(control_stats)
    result_stats['valid_comparison_videos'] = len(comparison_stats)
    result_stats['total_videos'] = total_videos

    return result_stats

def process_all_files(base_dir):
    """
    处理所有文件并按新格式组织结果
    """
    participant_data = {}
    video_counts = []

    for group in ['Exp', 'Nov']:
        for condition in ['virtual_control_hazard', 'virtual_control_occlusion',
                         'real_control_hazard', 'real_control_occlusion']:
            condition_dir = os.path.join(base_dir, group, condition)

            if not os.path.exists(condition_dir):
                print(f"Directory not found: {condition_dir}")
                continue

            print(f"\nProcessing {group} - {condition}")

            for file in os.listdir(condition_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(condition_dir, file)
                    participant = file.replace('.csv', '')
                    print(f"Processing participant: {participant}")

                    stats = calculate_participant_statistics(file_path)

                    video_counts.append({
                        'filename': participant,
                        'condition': condition,
                        'total_videos': stats['total_videos'],
                        'valid_control_videos': stats['valid_control_videos'],
                        'valid_comparison_videos': stats['valid_comparison_videos']
                    })

                    if participant not in participant_data:
                        participant_data[participant] = {
                            'filename': participant,
                            'expertise': 'Expert' if group == 'Exp' else 'Novice'
                        }

                    # 添加Control和对比条件的统计数据
                    for metric in ['first_aoi_time', 'first_aoi_duration', 'total_entries', 'avg_duration']:
                        for condition_type in ['control', 'comparison']:
                            participant_data[participant][f'{condition}_{metric}_{condition_type}_mean'] = \
                                stats[f'{metric}_{condition_type}_mean']
                            participant_data[participant][f'{condition}_{metric}_{condition_type}_std'] = \
                                stats[f'{metric}_{condition_type}_std']

    results_df = pd.DataFrame(list(participant_data.values()))
    video_counts_df = pd.DataFrame(video_counts)

    # 确保列的顺序正确
    columns = ['filename', 'expertise']
    conditions = ['virtual_control_hazard', 'virtual_control_occlusion',
                 'real_control_hazard', 'real_control_occlusion']
    metrics = ['first_aoi_time', 'first_aoi_duration', 'total_entries', 'avg_duration']
    condition_types = ['control', 'comparison']

    for condition in conditions:
        for metric in metrics:
            for condition_type in condition_types:
                columns.append(f'{condition}_{metric}_{condition_type}_mean')
                columns.append(f'{condition}_{metric}_{condition_type}_std')

    results_df = results_df.reindex(columns=columns)

    return results_df, video_counts_df

def main():
    # 设置输入输出路径
    input_base_dir = "M:\\EEG_DATA\\EEG_data_0410\\aoi_analysis_results\\aoi_metrics_20241122_075116"
    output_dir = "M:\\EEG_DATA\\EEG_data_0410\\aoi_statistics_results"

    input_base_dir = "M:\\EEG_DATA\\EEG_data_0410\\aoi_analysis_results_with_non_aoi\\aoi_metrics_20241122_083242"
    output_dir = "M:\\EEG_DATA\\EEG_data_0410\\aoi_statistics_results_with_non_aoi"

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理所有数据
    results_df, video_counts_df = process_all_files(input_base_dir)

    # 保存结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(output_dir, f'aoi_statistics_{timestamp}.csv'), index=False)
    video_counts_df.to_csv(os.path.join(output_dir, f'video_counts_{timestamp}.csv'), index=False)

    print("\nProcessing completed!")
    print(f"Results saved to: aoi_statistics_{timestamp}.csv")
    print(f"Video counts saved to: video_counts_{timestamp}.csv")
    print("\nColumns in the results file:")
    for col in results_df.columns:
        print(f"- {col}")

    return results_df, video_counts_df


if __name__ == "__main__":
    results_df, video_counts_df = main()