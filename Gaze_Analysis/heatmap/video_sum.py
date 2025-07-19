"""
simplified_frame_reference.py
简化版帧对照生成脚本
"""
import pandas as pd
import re


def is_virtual(name):
    """判断是否为虚拟视频"""
    return 'bd' in name.lower()


def filter_frames(input_csv, output_csv):
    """带未处理视频统计的版本"""
    # 读取原始数据
    df = pd.read_csv(input_csv)
    all_videos = df['Name'].unique()

    # 处理数据
    processed = []
    missing_videos = []

    for name in all_videos:
        group = df[df['Name'] == name]
        try:
            max_id = group['Image ID'].max()
            offset = 10 if is_virtual(name) else 15
            target_id = max(max_id - offset, 0)  # 确保不小于0

            # 获取目标帧
            target_frame = group[group['Image ID'] == target_id]

            if not target_frame.empty:
                target_frame = target_frame.copy()
                target_frame['video_type'] = 'virtual' if is_virtual(name) else 'real'
                processed.append(target_frame)
            else:
                missing_videos.append(name)
        except:
            missing_videos.append(name)

    # 保存结果
    if processed:
        result_df = pd.concat(processed)
        result_df.to_csv(output_csv, index=False)
        print(f"生成对照文件成功！包含 {len(result_df)} 条记录")
    else:
        print("错误：没有生成任何有效记录！")

    # 输出未处理视频
    if missing_videos:
        print(f"\n未处理视频数量：{len(missing_videos)}")
        print("未处理视频列表：")
        print("\n".join(missing_videos))
    else:
        print("\n所有视频已成功处理！")


if __name__ == "__main__":
    filter_frames(
        "M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv",
        "simple_frame_reference.csv"
    )