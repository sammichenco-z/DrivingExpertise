import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置参数
EXP_DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\full_gaze\\Exp\\exp_data.csv"
NOV_DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\full_gaze\\Nov\\nov_data.csv"
FRAME_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Extracted_Frames_raw"
OUTPUT_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Heatmaps_allgaze"
SCREEN_SIZE = (1920, 1080)  # (宽, 高)
HEATMAP_ALPHA = 0.6
MIN_DATA_POINTS = 3  # 最小有效数据点要求

# 运动补偿参数
VEHICLE_SPEED_MS = 13.8889  # 车速，m/s
FPS = 30  # 视频帧率，fps
METERS_PER_FRAME = VEHICLE_SPEED_MS / FPS  # 每帧前进距离
INITIAL_DISTANCE = 45.66  # 初始距离（米）
FOCAL_LENGTH = 960  # 焦距（像素），根据AOI_Cal函数推断

# 屏幕参数
SCREEN_CENTER_X = 960
SCREEN_CENTER_Y = 540


motion_comp_factor= 1


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
    except Exception as e:
        print(f"\n热图生成异常：{str(e)[:100]}")
        return img, 'error'


def calculate_screen_shift(screen_x, screen_y, initial_distance, current_distance):
    """
    根据距离变化计算屏幕坐标的偏移量
    基于AOI_Cal函数的透视投影原理

    参数:
    - screen_x, screen_y: 屏幕坐标
    - initial_distance: 初始距离（米）
    - current_distance: 当前距离（米）

    返回:
    - dx, dy: 屏幕偏移量（像素）
    """
    # 如果 is_reverse 为 False，AOI_Cal 函数使用:
    # xx = 960 - 960 * (x1 - x2) / (y1 - y2)
    # yy = 540 - 960 * (z2 - z1) / (y1 - y2)

    # 对于固定目标，x1-x2和z2-z1保持不变
    # 只有y1-y2随着车辆接近目标而减小

    # 初始投影计算（简化版）
    relative_x = screen_x - SCREEN_CENTER_X  # 水平偏离中心的距离
    relative_y = screen_y - SCREEN_CENTER_Y  # 垂直偏离中心的距离

    # 计算距离比例变化
    distance_ratio = initial_distance / current_distance

    # 新的屏幕坐标（基于距离比例）
    new_x = SCREEN_CENTER_X + relative_x * distance_ratio
    new_y = SCREEN_CENTER_Y + relative_y * distance_ratio

    # 计算偏移量
    dx = new_x - screen_x
    dy = new_y - screen_y

    return dx, dy


def compensate_motion(video_group, reference_frame_id):
    """
    为视频组中的所有眼动数据应用运动补偿
    基于车辆沿y轴前进导致的距离变化

    参数:
    - video_group: 包含眼动数据的DataFrame
    - reference_frame_id: 参考帧ID

    返回:
    - 调整后的点坐标数组
    """
    adjusted_points = []

    # 计算参考帧的距离（假设是初始距离）
    reference_distance = INITIAL_DISTANCE - (reference_frame_id * METERS_PER_FRAME)

    for _, row in video_group.iterrows():
        x, y = row['gaze_x'], row['gaze_y']
        current_frame = row['frame_id']

        # 计算当前帧的实际距离
        current_distance = INITIAL_DISTANCE - (current_frame * METERS_PER_FRAME)

        # 确保距离不会变为负数
        if current_distance <= 0:
            print(f"警告：帧 {current_frame} 距离计算为负值，跳过")
            continue

        # 计算屏幕偏移量（从当前帧回到参考帧）
        dx, dy = calculate_screen_shift(x, y, current_distance, reference_distance)

        # 应用偏移
        x_adjusted = x + dx
        y_adjusted = y + dy

        adjusted_points.append([x_adjusted, y_adjusted])

    return np.array(adjusted_points)


def process_group(data_path, output_subdir, use_gaussian=True):
    """
    处理一组数据，为每个视频生成热图，使用所有匹配 video 的数据。
    包含对车辆沿y轴运动的补偿校正。
    现在会从可用的图像文件中确定参考帧ID。

    参数：
    - motion_comp_factor: 运动补偿系数，用于微调补偿效果(1.0表示完全补偿，<1.0减弱补偿，>1.0增强补偿)
    """
    try:
        df = pd.read_csv(data_path)
        valid_df = df.dropna(subset=['gaze_x', 'gaze_y'])
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        return

    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 打印运动补偿参数
    frames_to_target = int(INITIAL_DISTANCE / METERS_PER_FRAME)

    global motion_comp_factor


    print(f"车辆速度: {VEHICLE_SPEED_MS:.2f} m/s ({VEHICLE_SPEED_MS * 3.6:.1f} km/h)")
    print(f"每帧前进距离: {METERS_PER_FRAME:.3f} 米")
    print(f"初始目标距离: {INITIAL_DISTANCE:.2f} 米 (约 {frames_to_target} 帧)")
    print(f"运动补偿系数: {motion_comp_factor}")

    # 从FRAME_DIR获取所有可用的图像文件并解析它们的视频ID和帧ID
    available_frames = {}
    for filename in os.listdir(FRAME_DIR):
        if filename.endswith(".jpg"):
            try:
                # 尝试从文件名中解析视频ID和帧ID
                parts = filename.split("_frame")
                if len(parts) == 2:
                    vid = parts[0]
                    frame_id = int(parts[1].split(".")[0])

                    # 为每个视频ID存储可用的帧ID
                    if vid not in available_frames:
                        available_frames[vid] = []
                    available_frames[vid].append(frame_id)
            except:
                continue

    # 排序每个视频的帧列表，方便后续选择
    for vid in available_frames:
        available_frames[vid].sort()

    log = []
    # 按 video 分组获取所有数据点
    video_grouped = valid_df.groupby('video')

    with tqdm(total=len(video_grouped), desc=f"处理{output_subdir}") as pbar:
        for vid, video_group in video_grouped:
            # 检查是否有该视频的可用帧
            if vid not in available_frames or not available_frames[vid]:
                log_entry = {
                    'video': vid,
                    'status': 'failed',
                    'reason': 'no_available_frames'
                }
                log.append(log_entry)
                pbar.update(1)
                continue

            # 获取该视频所有不同的帧ID
            data_frame_ids = video_group['frame_id'].unique()
            frame_range = f"{min(data_frame_ids)}-{max(data_frame_ids)}"
            frame_count = max(data_frame_ids) - min(data_frame_ids) + 1

            # 选择参考帧ID
            # 策略1：选择最小的数据帧ID，然后查找最接近的可用帧
            # 策略2：选择可用帧中间的一帧作为参考
            # 这里使用策略1，但你可以根据需要修改

            min_data_frame = min(data_frame_ids)

            # 查找最接近的可用帧
            closest_available_frame = min(available_frames[vid],
                                          key=lambda x: abs(x - min_data_frame))

            reference_frame_id = closest_available_frame

            # 背景图文件路径
            img_name = f"{vid}_frame{reference_frame_id:04d}.jpg"
            img_path = os.path.join(FRAME_DIR, img_name)

            # 输出文件名增加周期和校正信息
            output_name = f"{vid}_frames{frame_range}_motion_comp.jpg"
            output_path = os.path.join(output_dir, output_name)

            # 计算参考帧的距离
            reference_distance = INITIAL_DISTANCE - (reference_frame_id * METERS_PER_FRAME)
            if reference_distance <= 0:
                print(f"警告：视频 {vid} 参考帧 {reference_frame_id} 距离计算为负值，跳过")
                continue

            log_entry = {
                'video': vid,
                'reference_frame': reference_frame_id,
                'reference_distance': reference_distance,
                'frame_range': frame_range,
                'frame_count': frame_count,
                'unique_frames': len(data_frame_ids),
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
                log_entry.update(status='failed0', reason=f'background_frame_missing: {img_name}')
                log.append(log_entry)
                pbar.update(1)
                continue

            img = cv2.imread(img_path)
            if img is None:
                log_entry.update(status='failed1', reason='image_read_error')
                log.append(log_entry)
                pbar.update(1)
                continue

            # 为每个数据点应用运动校正
            adjusted_points = compensate_motion(video_group, reference_frame_id)

            if len(adjusted_points) == 0:
                log_entry.update(status='filtered', reason='all_points_beyond_target')
                log.append(log_entry)
                pbar.update(1)
                continue

            if motion_comp_factor != 1.0:
                # 计算每个点相对于原始点的偏移
                original_points = video_group[['gaze_x', 'gaze_y']].values.astype(float)

                # 确保形状匹配（可能会因为跳过一些点导致不匹配）
                if len(original_points) > len(adjusted_points):
                    original_points = original_points[:len(adjusted_points)]

                offsets = adjusted_points - original_points

                # 应用补偿系数
                adjusted_points = original_points + offsets * motion_comp_factor

            # 过滤掉屏幕外的点
            valid_points = adjusted_points[
                (adjusted_points[:, 0] >= 0) & (adjusted_points[:, 0] < SCREEN_SIZE[0]) &
                (adjusted_points[:, 1] >= 0) & (adjusted_points[:, 1] < SCREEN_SIZE[1])
                ]

            # 记录校正前后的点数变化
            log_entry['points_before_filtering'] = len(adjusted_points)
            log_entry['points_after_filtering'] = len(valid_points)

            if len(valid_points) < MIN_DATA_POINTS:
                log_entry.update(status='filtered', reason='insufficient_data_after_motion_comp')
                log.append(log_entry)
                pbar.update(1)
                continue

            # 创建热图
            if use_gaussian:
                sigma = min(50, max(15, len(valid_points) / 20))
                heatmap_img = create_heatmap_using_gaussian(
                    img, valid_points, kernel_size=200, sigma=sigma, alpha=HEATMAP_ALPHA
                )
                status = 'success'
            else:
                heatmap_img, status = create_heatmap(img, valid_points)

            if status == 'success':
                # 在图片上添加信息标记
                info_text = f"Video: {vid} | Frames: {frame_range} | Ref: {reference_frame_id}"
                points_text = f"Points: {len(valid_points)}/{len(adjusted_points)}"
                dist_text = f"Ref Dist: {reference_distance:.1f}m | Speed: {VEHICLE_SPEED_MS:.1f}m/s"

                cv2.putText(
                    heatmap_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    heatmap_img, points_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    heatmap_img, dist_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                cv2.imwrite(output_path, heatmap_img)
                log_entry.update(status='success')
            else:
                log_entry.update(status='filtered', reason=status)

            log.append(log_entry)
            pbar.update(1)

    log_df = pd.DataFrame(log)
    log_path = os.path.join(OUTPUT_DIR, f"log_{output_subdir}_motion_comp.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n{output_subdir}组统计：")
    print(log_df['status'].value_counts().to_string())

    if 'points_before_filtering' in log_df.columns and 'points_after_filtering' in log_df.columns:
        total_before = log_df['points_before_filtering'].sum()
        total_after = log_df['points_after_filtering'].sum()
        retention_rate = (total_after / total_before * 100) if total_before > 0 else 0
        print(f"\n运动校正后数据点保留率: {retention_rate:.1f}% ({total_after}/{total_before})")

    if not log_df[log_df['status'] == 'filtered'].empty:
        print("\n过滤案例示例：")
        print(log_df[log_df['status'] == 'filtered'].head(3).to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n开始处理Exp...")
    process_group(EXP_DATA_PATH, "Exp", use_gaussian=True)

    print("\n开始处理Nov...")
    process_group(NOV_DATA_PATH, "Nov", use_gaussian=True)


if __name__ == "__main__":
    main()