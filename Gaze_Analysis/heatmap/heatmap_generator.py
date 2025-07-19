import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置参数
EXP_DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Exp\\exp_data.csv"
NOV_DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Nov\\nov_data.csv"
FRAME_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Extracted_Frames_raw"
OUTPUT_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Heatmaps"
SCREEN_SIZE = (1920, 1080)  # (宽, 高)
HEATMAP_ALPHA = 0.6
MIN_DATA_POINTS = 3  # 最小有效数据点要求


def create_heatmap_using_gaussian(img, points, kernel_size, sigma, alpha=HEATMAP_ALPHA):
    """
    基于累加 + 高斯模糊方式生成热图，使得每个关注点都以固定形状进行扩散，
    从而在重合区域形成明显热点。
    """
    h, w = SCREEN_SIZE[1], SCREEN_SIZE[0]  # h: 高, w: 宽
    heatmap_data = np.zeros((h, w), dtype=np.float32)


    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            heatmap_data[y, x] += 1

    heatmap_data = cv2.GaussianBlur(heatmap_data, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if np.max(heatmap_data) > 0:
        heatmap_norm = np.uint8(255 * (heatmap_data - heatmap_data.min()) /
                                (heatmap_data.max() - heatmap_data.min()))
    else:
        heatmap_norm = np.zeros_like(heatmap_data, dtype=np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))

    result = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return result


def create_heatmap(img, points):
    """原来的基于 gaussian_kde 的热图生成方式，留作备用。"""
    try:
        if len(points) < MIN_DATA_POINTS:
            return img, 'insufficient_data'

        valid_points = points[
            (points[:, 0] >= 0) & (points[:, 0] < SCREEN_SIZE[0]) &
            (points[:, 1] >= 0) & (points[:, 1] < SCREEN_SIZE[1])
        ]
        if len(valid_points) < 2:
            return img, 'invalid_coordinates'

        x = valid_points[:, 0]
        y = valid_points[:, 1]
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        bw_x = max(50.0, x_range / 10) if x_range > 0 else 50.0
        bw_y = max(50.0, y_range / 10) if y_range > 0 else 50.0
        bandwidth = np.sqrt(bw_x * bw_y)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(valid_points.T, bw_method=bandwidth / SCREEN_SIZE[0])
        grid_x = np.linspace(0, SCREEN_SIZE[0], 192)
        grid_y = np.linspace(0, SCREEN_SIZE[1], 108)
        X, Y = np.meshgrid(grid_x, grid_y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions)
        Z = Z.reshape(X.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        heatmap = cv2.applyColorMap((Z * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, SCREEN_SIZE)

        return cv2.addWeighted(img, 1 - HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0), 'success'
    except np.linalg.LinAlgError:
        return img, 'singular_matrix'
    except Exception as e:
        print(f"\n热图生成异常：{str(e)[:100]}")
        return img, 'error'


def process_group(data_path, output_subdir, use_gaussian=True):
    """
    处理一组数据，为每个视频生成热图，使用所有匹配 video_id 的数据，但保留原有背景图像和输出格式。

    参数：
      data_path   - CSV 数据文件路径
      output_subdir - 保存热图的子目录名
      use_gaussian - 是否采用基于 GaussianBlur 的方法生成热图
    """
    try:
        df = pd.read_csv(data_path)
        valid_df = df.dropna(subset=['gaze_x', 'gaze_y'])
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        return

    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    log = []
    # 先按 video_id 分组获取所有数据点
    video_grouped = valid_df.groupby('video_id')

    with tqdm(total=len(video_grouped), desc=f"处理{output_subdir}") as pbar:
        for vid, video_group in video_grouped:
            # 获取该视频的唯一 frame_id（假设每个视频只有一帧）
            frame_ids = video_group['frame_id'].unique()
            if len(frame_ids) > 1:
                print(f"警告：视频 {vid} 包含多个 frame_id：{frame_ids}，使用第一个帧")
            fid = frame_ids[0]  # 使用第一个 frame_id

            img_name = f"{vid}_frame{fid:04d}.jpg"
            img_path = os.path.join(FRAME_DIR, img_name)
            output_path = os.path.join(output_dir, img_name)  # 保留原有输出格式

            log_entry = {
                'video': vid,
                'frame': fid,
                'data_points': len(video_group),
                'status': 'pending',
                'reason': ''
            }

            if os.path.exists(output_path):
                log_entry.update(status='skipped', reason='exists')
                log.append(log_entry)
                pbar.update(1)
                continue

            if not os.path.exists(img_path):
                log_entry.update(status='failed', reason='source_missing')
                log.append(log_entry)
                pbar.update(1)
                continue

            img = cv2.imread(img_path)
            if img is None:
                log_entry.update(status='failed', reason='image_read_error')
                log.append(log_entry)
                pbar.update(1)
                continue

            # 使用该 video_id 的所有数据点生成热图
            points = video_group[['gaze_x', 'gaze_y']].values.astype(float)
            if len(points) < MIN_DATA_POINTS:
                log_entry.update(status='filtered', reason='insufficient_data')
                log.append(log_entry)
                pbar.update(1)
                continue

            if use_gaussian:
                heatmap_img = create_heatmap_using_gaussian(img, points, kernel_size=200, sigma=25, alpha=HEATMAP_ALPHA)
                # # 动态设置heatmap效果
                # sigma = min(100, max(20, len(points) / 10))  # 点越多，扩散越大
                # alpha = min(0.8, max(0.2, 1.0 / np.log1p(len(points))))  # 点越多，透明度越低
                # heatmap_img = create_heatmap_using_gaussian(img, points, sigma=sigma, alpha=alpha)

                status = 'success'
            else:
                heatmap_img, status = create_heatmap(img, points)

            if status == 'success':
                cv2.imwrite(output_path, heatmap_img)
                log_entry.update(status='success')
            else:
                log_entry.update(status='filtered', reason=status)
            log.append(log_entry)
            pbar.update(1)

    log_df = pd.DataFrame(log)
    log_path = os.path.join(OUTPUT_DIR, f"log_{output_subdir}.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n{output_subdir}组统计：")
    print(log_df['status'].value_counts().to_string())

    if not log_df[log_df['status'] == 'filtered'].empty:
        print("\n过滤案例示例：")
        print(log_df[log_df['status'] == 'filtered'].head(3).to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 40)
    print("开始处理实验组...")
    process_group(EXP_DATA_PATH, "Exp", use_gaussian=True)

    print("=" * 40)
    print("开始处理对照组...")
    process_group(NOV_DATA_PATH, "Nov", use_gaussian=True)


if __name__ == "__main__":
    main()