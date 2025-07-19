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


def create_heatmap_using_gaussian(img, points, kernel_size=200, sigma=20, alpha=HEATMAP_ALPHA):
    """
    基于累加 + 高斯模糊方式生成热图，使得每个关注点都以固定形状进行扩散，
    从而在重合区域形成明显热点。

    参数：
      img         - 原始图像（假设分辨率与 SCREEN_SIZE 一致）
      points      - (N, 2) 数组，每行为一个 (x, y) 坐标
      kernel_size - 高斯核的大小（影响局部区域，此处未直接使用，可在调试中调整）
      sigma       - 高斯模糊的标准差，控制扩散范围（数值越大扩散越宽）
      alpha       - 热图叠加的透明度

    返回：
      叠加了热图的原始图像
    """
    h, w = SCREEN_SIZE[1], SCREEN_SIZE[0]  # h: 高, w: 宽
    heatmap_data = np.zeros((h, w), dtype=np.float32)

    # 将每个关注点的计数累加到对应位置
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            heatmap_data[y, x] += 1  # 注意：图像中 y 表示行（高度）、x 表示列（宽度）

    # 使用 GaussianBlur 对累加后的热图进行平滑
    heatmap_data = cv2.GaussianBlur(heatmap_data, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 将平滑后的热图归一化到 0-255 区间
    if np.max(heatmap_data) > 0:
        heatmap_norm = np.uint8(255 * (heatmap_data - heatmap_data.min()) /
                                (heatmap_data.max() - heatmap_data.min()))
    else:
        heatmap_norm = np.zeros_like(heatmap_data, dtype=np.uint8)

    # 通过 COLORMAP_JET 生成彩色热图
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))  # 保证和原图大小一致

    # 叠加热图和原始图像
    result = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return result


def create_heatmap(img, points):
    """
    原来的基于 gaussian_kde 的热图生成方式，留作备用。
    """
    try:
        if len(points) < MIN_DATA_POINTS:
            return img, 'insufficient_data'

        # 过滤无效坐标
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
    处理一组数据，生成每一帧的热图并保存。

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
    grouped = valid_df.groupby(['video_id', 'frame_id'])

    with tqdm(total=len(grouped), desc=f"处理{output_subdir}") as pbar:
        for (vid, fid), group in grouped:
            img_name = f"{vid}_frame{fid:04d}.jpg"
            img_path = os.path.join(FRAME_DIR, img_name)
            output_path = os.path.join(output_dir, img_name)

            log_entry = {
                'video': vid,
                'frame': fid,
                'data_points': len(group),
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

            points = group[['gaze_x', 'gaze_y']].values.astype(float)
            if len(points) < MIN_DATA_POINTS:
                log_entry.update(status='filtered', reason='insufficient_data')
                log.append(log_entry)
                pbar.update(1)
                continue

            if use_gaussian:
                # 使用改进的基于 GaussianBlur 的方法
                heatmap_img = create_heatmap_using_gaussian(img, points, kernel_size=200, sigma=20, alpha=HEATMAP_ALPHA)
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
    # 设置 use_gaussian=True 可使用高斯叠加方法生成热点热图
    process_group(EXP_DATA_PATH, "Exp", use_gaussian=True)

    print("=" * 40)
    print("开始处理对照组...")
    process_group(NOV_DATA_PATH, "Nov", use_gaussian=True)


if __name__ == "__main__":
    main()