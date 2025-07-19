import os
import cv2
import pandas as pd

# Set the path to your video files directory
video_dir = 'M:\\EEG_DATA\\EEG\\experiment\\real'

# Set the path to your output directory
output_dir = 'M:\\EEG_DATA\\实际实验Ver\\AOIed_Video\\Real'

# Load AOI data from the CSV file
aoi_csv_file = 'M:\\EEG_DATA\\box_info_final.csv'
aoi_df = pd.read_csv(aoi_csv_file)

# Initialize a list to track videos without assigned AOI
videos_without_aoi = []

# Loop through each subdirectory under video_dir
for root, dirs, files in os.walk(video_dir):
    # Loop through each video file in this subdirectory
    for video_file in files:
        if not video_file.endswith('.mp4'):
            continue

        # Get the name of the video file without the extension
        video_name = os.path.splitext(video_file)[0]

        # Filter AOI data for the current video
        video_aoi_df = aoi_df[aoi_df['Name'] == video_name]

        # Check if the CSV file exists for this video
        if video_name not in aoi_df['Name'].values:
            videos_without_aoi.append(video_name)
            continue
        print("PROCESSING ", video_name)

        # Open the video file
        cap = cv2.VideoCapture(os.path.join(root, video_file))

        # Define the output filename
        output_file = os.path.join(output_dir, f'{video_name}_AOI.avi')

        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping processing.")
            continue

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (1920, 1080))

        # Loop through each frame of the video
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Get the Image ID for this frame
            image_id = i

            # Filter AOI data for the current frame (matching Image ID)
            frame_aoi_df = video_aoi_df[video_aoi_df['Image ID'] == image_id]

            # Read this frame from the video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Highlight AOI regions based on the AOI data for this frame
                for _, row in frame_aoi_df.iterrows():
                    xtl, ytl, xbr, ybr = int(row['xtl']), int(row['ytl']), int(row['xbr']), int(row['ybr'])
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

                # Write this frame to the output video file
                out.write(frame)

                # Display this frame
                cv2.imshow('frame', frame)

                # Wait for 'q' key to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        # Release everything when done for this video file
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Print videos without assigned AOI
print("Videos without assigned AOI:", videos_without_aoi)
