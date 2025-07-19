"""
video_sum_enhanced.py
增强版帧参考生成脚本，支持持续采样和时间控制
"""
import pandas as pd
import re
import os
from typing import Literal


def is_virtual(name):
    """判断是否为虚拟视频"""
    return 'bd' in name.lower()


def filter_frames(
        input_csv,
        output_folder,
        selection_mode: Literal['seconds', 'frames'] = 'seconds',
        selection_count: int = 5,
        start_offset: float = 0.0
):
    """
    增强版帧选择函数

    参数：
    output_folder - 输出文件夹路径
    selection_mode - 持续采样模式：
        'seconds' : 持续采样指定秒数的所有帧
        'frames'  : 持续采样指定数量的连续帧
    selection_count - 需要采集的时间长度（秒）或帧数
    start_offset - 起始偏移量（秒），相对于原始基准点
    """
    # 读取原始数据
    df = pd.read_csv(input_csv)
    all_videos = df['Name'].unique()

    # Generate filename based on parameters
    filename = f"reference_{selection_mode}_{selection_count}_{start_offset}.csv"
    output_csv = os.path.join(output_folder, filename)

    print(f"将生成文件: {filename}")

    # 视频元数据缓存（实际使用时应从视频文件读取）
    VIDEO_METADATA = {
        # 示例：'video1': {'fps': 30, 'total_frames': 300},
        # 实际应集成视频分析逻辑
    }

    processed = []
    missing_data = []

    for name in all_videos:
        group = df[df['Name'] == name]
        try:
            max_id = group['Image ID'].max()
            is_virtual_flag = is_virtual(name)

            # 原始基准点计算
            base_offset = 10 if is_virtual_flag else 15
            original_start = max(max_id - base_offset, 0)

            # 转换用户偏移量为帧数
            if selection_mode == 'seconds':
                fps = VIDEO_METADATA.get(name, {}).get('fps', 30)  # 默认30fps
                offset_frames = int(start_offset * fps)
            else:
                offset_frames = int(start_offset)

            # 计算实际起始帧
            actual_start = original_start - offset_frames
            actual_start = max(actual_start, 0)

            # 计算结束帧
            if selection_mode == 'seconds':
                duration_frames = int(selection_count * fps)
            else:
                duration_frames = selection_count

            end_frame = actual_start + duration_frames
            total_frames = group['Image ID'].max()

            # 边界保护
            end_frame = min(end_frame, total_frames)
            if end_frame <= actual_start:
                missing_data.append((name, "DurationExceeded"))
                continue

            # 生成目标帧范围
            target_frames = list(range(actual_start, end_frame + 1))

            # 获取有效帧
            valid_frames = group[group['Image ID'].isin(target_frames)]

            if not valid_frames.empty:
                valid_frames = valid_frames.copy()
                valid_frames['video_type'] = 'virtual' if is_virtual_flag else 'real'
                processed.append(valid_frames)
            else:
                missing_data.append((name, "NoFramesInRange"))

        except Exception as e:
            print(f"处理视频 {name} 时出错: {str(e)}")
            missing_data.append((name, "ProcessingError"))

    # 合并处理结果
    if processed:
        result_df = pd.concat(processed)

        # 添加采样元数据
        result_df['sampling_mode'] = selection_mode
        result_df['sampling_count'] = selection_count
        result_df['start_offset'] = start_offset

        result_df.to_csv(output_csv, index=False)
        print(f"成功生成 {len(result_df)} 条记录")
    else:
        print("错误：没有生成任何有效记录！")

    # 输出缺失信息
    if missing_data:
        error_df = pd.DataFrame(missing_data, columns=['video', 'error'])
        error_log = output_csv.replace('.csv', '_errors.csv')
        error_df.to_csv(error_log, index=False)
        print(f"\n发现 {len(missing_data)} 条错误记录，已保存至 {error_log}")


# 示例使用
if __name__ == "__main__":
    # 从基准点开始持续采集15帧的所有帧
    filter_frames(
        "M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv",
        "M:\\EEG_DATA\\Scripts\\heatmap\\",
        selection_mode='frames',
        selection_count=15,
        start_offset=0.0
    )

    # # 示例2：从基准点前推1秒开始采集0.5秒内容
    filter_frames(
        "M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv",
        "M:\\EEG_DATA\\Scripts\\heatmap\\",
        selection_mode='frames',
        selection_count=15,
        start_offset=10
    )
    #
    # 示例3：采集基准点开始的100连续帧
    filter_frames(
        "M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv",
        "M:\\EEG_DATA\\Scripts\\heatmap\\",
        selection_mode='seconds',
        selection_count=1,
        start_offset=0.5
    )
