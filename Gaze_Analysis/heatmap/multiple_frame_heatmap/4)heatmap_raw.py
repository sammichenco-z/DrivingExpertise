import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths for input and output
DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\full_gaze"
FRAME_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Extracted_Frames_raw"
OUTPUT_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Heatmaps_allgaze_nomotion"

# Configuration parameters
SCREEN_SIZE = (1920, 1080)  # (width, height)
HEATMAP_ALPHA = 0.6
MIN_DATA_POINTS = 3  # Minimum data points required for a heatmap


def create_heatmap_using_gaussian(img, points, kernel_size, sigma, alpha=HEATMAP_ALPHA):
    """
    Generate a heatmap using Gaussian blur based on gaze points.
    """
    h, w = SCREEN_SIZE[1], SCREEN_SIZE[0]  # h: height, w: width
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


def extract_video_name(image_filename):
    """
    Extract the video name from an image filename by removing the last underscore and suffix.
    Example: 'bd2_wal_rev_3.2_82_frame0061.jpg' -> 'bd2_wal_rev_3.2_82'
    """
    return "_".join(image_filename.split("_")[:-1])


def process_group(data_path, output_subdir):
    """
    Process a group of data (e.g., Exp or Nov) and generate heatmaps.
    """
    df = pd.read_csv(data_path)
    valid_df = df.dropna(subset=['gaze_x', 'gaze_y'])  # Ensure gaze points are valid

    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Map existing frames to their video names
    frame_files = [f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')]
    frame_mapping = {extract_video_name(f): f for f in frame_files}

    grouped = valid_df.groupby('video')
    log = []

    print(f"Processing {output_subdir} group...")
    for video, group in tqdm(grouped, desc=f"Processing {output_subdir}"):
        # Check if a frame exists for the video
        if video not in frame_mapping:
            print(f"[Error] Missing frame for video: {video}")
            continue

        # Use the first available frame for the video
        img_name = frame_mapping[video]
        img_path = os.path.join(FRAME_DIR, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[Error] Failed to read image: {img_path}")
            continue

        # Extract valid gaze points
        points = group[['gaze_x', 'gaze_y']].values
        if len(points) < MIN_DATA_POINTS:
            print(f"[Error] Insufficient data for video: {video}, points: {len(points)}")
            continue

        # Create heatmap
        sigma = min(50, max(15, len(points) / 20))
        heatmap_img = create_heatmap_using_gaussian(
            img, points, kernel_size=200, sigma=sigma, alpha=HEATMAP_ALPHA
        )

        # Save heatmap
        output_name = f"{video}_heatmap.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, heatmap_img)

        log.append({'video': video, 'status': 'success', 'points': len(points)})

    # Save log
    log_df = pd.DataFrame(log)
    log_path = os.path.join(OUTPUT_DIR, f"log_{output_subdir}.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n{output_subdir} summary:")
    print(log_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nProcessing Exp group...")
    process_group(os.path.join(DATA_PATH, 'Exp', 'exp_data.csv'), 'Exp')

    print("\nProcessing Nov group...")
    process_group(os.path.join(DATA_PATH, 'Nov', 'nov_data.csv'), 'Nov')


if __name__ == "__main__":
    main()