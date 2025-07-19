import pandas as pd
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def check_aoi(row):
    """
    检查采样点是否在AOI区域内
    """
    # 如果注视点为NaN，返回-1
    if pd.isna(row['Fixation Point X[px]']):
        return -1

    # 检查注视点是否在AOI框内
    in_x_range = row['Gaze Point X[px]'] >= row['xtl'] and row['Gaze Point X[px]'] <= row['xbr']
    in_y_range = row['Gaze Point Y[px]'] >= row['ytl'] and row['Gaze Point Y[px]'] <= row['ybr']

    # 如果在AOI框内返回1，否则返回0
    return 1 if (in_x_range and in_y_range) else 0


def process_single_file(args):
    """
    处理单个文件
    """
    input_file, output_file = args
    try:
        # 读取数据
        df = pd.read_csv(input_file)

        # 添加AOI判断列
        df['in_aoi'] = df.apply(check_aoi, axis=1)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存处理后的数据
        df.to_csv(output_file, index=False)

        return True, input_file
    except Exception as e:
        return False, f"Error processing {input_file}: {str(e)}"


def main():
    # 设置路径
    input_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\trial_with_aoi"
    output_base_folder = "M:\\EEG_DATA\\EEG_data_0410\\trial_with_aoi_processed"

    # 收集所有需要处理的文件
    files_to_process = []
    conditions = ['virtual_control_hazard', 'virtual_control_occlusion',
                  'real_control_hazard', 'real_control_occlusion']

    for group in ['Exp', 'Nov']:
        for condition in conditions:
            input_dir = os.path.join(input_base_folder, group, condition)
            output_dir = os.path.join(output_base_folder, group, condition)

            if not os.path.exists(input_dir):
                continue

            for file in os.listdir(input_dir):
                if file.endswith('.csv'):
                    input_file = os.path.join(input_dir, file)
                    output_file = os.path.join(output_dir, file)
                    files_to_process.append((input_file, output_file))

    print(f"Found {len(files_to_process)} files to process")

    # 使用进程池并行处理文件
    successful = 0
    failed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_single_file, files_to_process),
                            total=len(files_to_process),
                            desc="Processing files"))

        for success, message in results:
            if success:
                successful += 1
            else:
                failed += 1
                errors.append(message)

    # 打印处理结果
    print("\nProcessing completed:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()