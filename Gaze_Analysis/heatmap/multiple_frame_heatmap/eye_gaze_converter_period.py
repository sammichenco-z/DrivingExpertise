import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import hashlib

# 配置参数
COLUMN_MAPPING = {
    'gaze_x': 'Gaze Point X[px]',
    'gaze_y': 'Gaze Point Y[px]',
    'fixation_idx': 'Fixation Index',
    'video': 'Video',
    'image_id': 'Image ID',
    'timestamp': 'Recording Time Stamp[ms]'
}
REFERENCE_PATH = "M:\\EEG_DATA\\Scripts\\heatmap\\reference_full_duration.csv"
INPUT_BASE = "M:\\EEG_DATA\\EEG_data_0410\\video_only_with_frame"
OUTPUT_BASE = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final"
CONFLICT_LOG = os.path.join(OUTPUT_BASE, "timestamp_conflicts.csv")


def parse_reference_parameters(filename):
    """Parse parameters from reference filename"""
    import re
    try:
        # Extract mode (seconds or frames)
        if 'seconds' in filename:
            selection_mode = 'seconds'
            converter_mode = 'video_frame'  # Use video_frame mode for seconds
        elif 'frames' in filename:
            selection_mode = 'frames'
            converter_mode = 'video_frame'  # Use video_frame for frames too
        else:
            return 'video_frame', 2  # Default values

        # Extract count (duration/frames)
        pattern = r'reference_\w+_(\d+)_'
        count_match = re.search(pattern, filename)
        duration = int(count_match.group(1)) if count_match else 2

        return converter_mode, duration
    except Exception as e:
        print(f"Warning: Could not parse parameters: {e}")
        return 'video_frame', 2  # Default values


def load_reference():
    """加载参考数据"""
    ref = pd.read_csv(REFERENCE_PATH)
    ref['video_key'] = ref['Name'] + '_' + ref['Image ID'].astype(str)
    return ref.set_index('video_key')


def extract_user_id(filename):
    """从文件名提取用户ID（增强鲁棒性）"""
    try:
        parts = filename.split('_')
        if len(parts) >= 3 and parts[1].startswith('User'):
            return parts[1][4:]  # 提取User后的数字
        return filename.split('.')[0]  # 保底返回文件名主体
    except:
        return 'unknown'


def process_single_group(group_name, ref_df, mode, duration):
    """处理单个实验组（Exp/Nov），添加模式和时长参数"""
    group_data = []
    conflict_records = []
    seen_signatures = {}

    input_folder = os.path.join(INPUT_BASE, group_name)
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"输入目录不存在：{input_folder}")

    csv_files = []
    for root, _, files in os.walk(input_folder):
        csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    csv_files.sort()

    for file_path in tqdm(csv_files, desc=f"处理{group_name}组"):
        try:
            df = pd.read_csv(file_path)
            user_id = extract_user_id(os.path.basename(file_path))

            merged = pd.merge(
                df,
                ref_df[['Name', 'Image ID', 'Condition']],
                left_on=[COLUMN_MAPPING['video'], COLUMN_MAPPING['image_id']],
                right_on=['Name', 'Image ID'],
                how='inner',
                suffixes=('', '_ref'))

            if merged.empty:
                continue

            # valid = merged[merged[COLUMN_MAPPING['fixation_idx']].notna()]
            # if valid.empty:
            #     continue
            valid = merged

            grouped = valid.groupby([COLUMN_MAPPING['video'], COLUMN_MAPPING['image_id']])
            for (video, frame_id), group in grouped:
                video_key = (user_id, video, frame_id)
                sorted_group = group.sort_values(COLUMN_MAPPING['timestamp'])

                # 根据模式选择数据量
                if mode == 'video_frame':
                    # 按帧数限制（每秒30帧）
                    max_frame = frame_id + duration
                    selected_data = sorted_group[
                        sorted_group[COLUMN_MAPPING['image_id']] < max_frame
                        ]
                elif mode == 'eye_gaze_sampling':
                    # 按采样数限制
                    selected_data = sorted_group.iloc[:duration]
                else:
                    raise ValueError("模式必须为 'video_frame' 或 'eye_gaze_sampling'")

                if selected_data.empty:
                    continue

                # 使用第一帧数据进行时间戳检查
                first_row = selected_data.iloc[0]
                ts_values = selected_data[COLUMN_MAPPING['timestamp']].values
                current_hash = hashlib.md5(ts_values.tobytes()).hexdigest()

                if video_key in seen_signatures:
                    existing = seen_signatures[video_key]
                    if existing['hash'] != current_hash:
                        conflict_records.append({
                            'user_id': user_id,
                            'video': video,
                            'frame_id': frame_id,
                            'first_ts': existing['min_ts'],
                            'new_ts': ts_values.min(),
                            'occurrences': existing['count'] + 1
                        })
                    continue
                else:
                    seen_signatures[video_key] = {
                        'hash': current_hash,
                        'min_ts': ts_values.min(),
                        'count': 1
                    }

                # 保存所有符合条件的采样点
                for _, row in selected_data.iterrows():
                    group_data.append({
                        'subject_id': user_id,
                        'video_id': video,
                        'frame_id': row[COLUMN_MAPPING['image_id']],
                        'condition': row['Condition'],
                        'gaze_x': row[COLUMN_MAPPING['gaze_x']],
                        'gaze_y': row[COLUMN_MAPPING['gaze_y']],
                        'timestamp_ms': row[COLUMN_MAPPING['timestamp']],
                        'group': group_name
                    })

        except Exception as e:
            print(f"\n处理文件失败：{os.path.basename(file_path)}")
            print(f"错误信息：{str(e)}")
            print("=" * 50)

    return pd.DataFrame(group_data), pd.DataFrame(conflict_records)


def main():
    try:
        # Setup output directories
        os.makedirs(os.path.join(OUTPUT_BASE, 'Exp'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, 'Nov'), exist_ok=True)

        print("正在加载参考数据...")
        ref_df = load_reference()

        # Get reference filename and parse parameters
        ref_filename = os.path.basename(REFERENCE_PATH)
        detected_mode, detected_duration = parse_reference_parameters(ref_filename)

        # Use detected parameters or fall back to defaults
        mode = detected_mode
        duration = detected_duration

        print(f"\n参数检测：模式={mode}，时长={duration}")
        print(f"来源文件：{ref_filename}")

        # Continue with processing using these parameters
        print("\n开始处理expert...")
        exp_df, exp_conflicts = process_single_group('Exp', ref_df, mode, duration)

        print("\n开始处理novice...")
        nov_df, nov_conflicts = process_single_group('Nov', ref_df, mode, duration)

        all_conflicts = pd.concat([exp_conflicts, nov_conflicts])

        print("\n保存处理结果...")
        exp_df.to_csv(os.path.join(OUTPUT_BASE, 'Exp', 'exp_data.csv'), index=False)
        nov_df.to_csv(os.path.join(OUTPUT_BASE, 'Nov', 'nov_data.csv'), index=False)

        if not all_conflicts.empty:
            all_conflicts.to_csv(CONFLICT_LOG, index=False)
            print(f"发现时间戳冲突 {len(all_conflicts)} 条，已保存至 {CONFLICT_LOG}")

        print("\n处理结果统计：")
        print(f"expert数据量：{len(exp_df):,}")
        print(f"novice数据量：{len(nov_df):,}")
        print(f"总冲突记录数：{len(all_conflicts):,}")

    except Exception as e:
        print("\n发生严重错误：")
        print(str(e))
        print("请检查配置路径和输入文件格式")


if __name__ == "__main__":
    main()