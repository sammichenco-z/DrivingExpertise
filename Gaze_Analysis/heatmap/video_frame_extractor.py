"""
video_frame_extractor_v5.py
保留小数点修复版本
"""
import os
import cv2
import pandas as pd
from tqdm import tqdm

# 配置参数
REFERENCE_CSV = "M:\\EEG_DATA\\Scripts\\heatmap\\simple_frame_reference.csv"
VIDEO_BASE = "M:\\EEG_DATA\\AOIed_Video"
OUTPUT_FOLDER = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Extracted_Frames_raw_noframe"
LOG_FILE = os.path.join(OUTPUT_FOLDER, "extraction_log.csv")


def is_virtual(name):
    """根据名称判断视频类型（保留原始名称中的小数点）"""
    return 'bd' in name.lower()


def create_video_path(name):
    """生成带子目录的视频路径（仅处理空格，保留小数点）"""
    # 只替换空格，保留其他字符（包括小数点）
    clean_name = name.replace(' ', '_')
    # 确定子目录
    sub_folder = 'Animated' if is_virtual(clean_name) else 'Real'
    # 构建完整路径
    return os.path.join(
        VIDEO_BASE,
        sub_folder,
        f"{clean_name}_AOI.avi"  # 保留原始小数点
    )


def extract_frames():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    ref_df = pd.read_csv(REFERENCE_CSV)
    log = []

    pbar = tqdm(ref_df.iterrows(), total=len(ref_df), desc="处理视频")
    for idx, row in pbar:
        name = row['Name']
        frame = row['Image ID']
        output_file = os.path.join(OUTPUT_FOLDER, f"{name}_frame{frame:04d}.jpg")


        # 跳过已存在的文件
        if os.path.exists(output_file):
            log.append({'video': name, 'status': 'skipped'})
            continue

        # 获取视频路径（保留小数点）
        video_path = create_video_path(name)
        pbar.set_postfix(file=os.path.basename(video_path))

        # 显示完整路径用于调试
        print(f"\n正在尝试访问：{video_path}")  # 调试输出

        if not os.path.exists(video_path):
            print(f"\n错误：视频文件不存在 {video_path}")
            log.append({'video': name, 'status': 'failed', 'error': 'file missing'})
            continue

        # 视频处理流程
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame >= total:
            print(f"\n错误：{name} 的帧号{frame}超出范围（最大{total-1}）")
            log.append({'video':name, 'status':'failed', 'error':'frame overflow'})
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        if ret:
            cv2.imwrite(output_file, img)
            log.append({'video':name, 'status':'success'})
        else:
            print(f"\n错误：无法读取{name}第{frame}帧")
            log.append({'video':name, 'status':'failed', 'error':'read error'})

        cap.release()

    # 保存日志并打印统计
    pd.DataFrame(log).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
    success = len([x for x in log if x['status']=='success'])
    print(f"\n处理完成！成功：{success}/{len(ref_df)}")

if __name__ == "__main__":
    extract_frames()