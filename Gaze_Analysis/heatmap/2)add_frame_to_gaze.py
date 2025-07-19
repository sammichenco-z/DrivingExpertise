import os
import pandas as pd
from pathlib import Path

# Define input and output paths
input_path = r"M:\EEG_DATA\EEG_data_0410\video_only"
output_path = r"M:\EEG_DATA\EEG_data_0410\video_only_with_frame"

# Frame rate (30 frames per second = 30 frames per 1000ms)
FPS = 30
MS_PER_FRAME = 1000 / FPS  # Time duration per frame in milliseconds

# Function to process CSV files
def compute_image_id(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # Create output directory if it doesn't exist
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                try:
                    # Read the CSV file
                    df = pd.read_csv(input_file_path)

                    # Ensure the 'Video' and 'Recording Time Stamp[ms]' columns exist
                    if "Video" in df.columns and "Recording Time Stamp[ms]" in df.columns:
                        # Group by the 'Video' column
                        df["Image ID"] = df.groupby("Video")["Recording Time Stamp[ms]"].transform(
                            lambda x: ((x - x.min()) // MS_PER_FRAME).astype(int)
                        )

                        # Write the processed CSV to the output folder
                        output_file_path = os.path.join(output_dir, file)
                        df.to_csv(output_file_path, index=False)
                    else:
                        print(f"Missing required columns in {input_file_path}. Skipping.")
                except Exception as e:
                    print(f"Error processing file {input_file_path}: {e}")

# Process the input folder
compute_image_id(input_path, output_path)

print("Processing complete!")