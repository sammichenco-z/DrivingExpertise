import os
import cv2
import pandas as pd
from tqdm import tqdm

# Paths for input and output
DATA_PATH = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\full_gaze"
FRAME_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\Extracted_Frames_raw"
OUTPUT_DIR = "M:\\EEG_DATA\\EEG_data_0410\\heatmap_ready_final\\GazeDots"

# Configuration parameters
SCREEN_SIZE = (1920, 1080)  # (width, height)
DOT_COLOR = (0, 0, 255)  # Red color for gaze points (BGR format)
DOT_RADIUS = 5  # Radius of the dot
DOT_THICKNESS = -1  # Filled circle


def extract_video_name(image_filename):
    """
    Extract the video name from an image filename by removing the last underscore and suffix.
    Example: 'bd2_wal_rev_3.2_82_frame0061.jpg' -> 'bd2_wal_rev_3.2_82'
    """
    return "_".join(image_filename.split("_")[:-1])


def process_group(data_path, output_subdir):
    """
    Process a group of data (e.g., Exp or Nov) and visualize gaze points as dots on the frames.
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
        if len(points) == 0:
            print(f"[Error] No valid gaze points for video: {video}")
            continue

        # Plot each gaze point as a dot on the image
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < SCREEN_SIZE[0] and 0 <= y < SCREEN_SIZE[1]:
                cv2.circle(img, (x, y), DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)

        # Save the visualized image
        output_name = f"{video}_dots.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, img)

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