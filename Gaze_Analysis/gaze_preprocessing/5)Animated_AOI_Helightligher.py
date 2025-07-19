import os

import cv2
import pandas as pd

# Set the path to your video files directory
video_dir = 'M:\\EEG_DATA\\实际实验Ver\\virtual'

# Set the path to your output directory
output_dir = 'M:\\EEG_DATA\\实际实验Ver\\AOIed_Video'

# Loop through each subdirectory under video_dir
for root, dirs, files in os.walk(video_dir):
    # Loop through each video file in this subdirectory
    for video_file in files:
        if not video_file.endswith('.mp4'):
            continue

        # Get the name of the video file without the extension
        video_name = os.path.splitext(video_file)[0]

        # Set the path to the CSV file for this video
        csv_file = os.path.join(root, f'{video_name}_AOI.csv')

        # Check if the CSV file exists for this video
        if not os.path.exists(csv_file):
            continue
        print("PROCESSING ", video_name)
        # Read the CSV file containing the coordinates of the bounding points
        df = pd.read_csv(csv_file)

        # Open the video file
        cap = cv2.VideoCapture(os.path.join(root, video_file))

        # Define the output filename
        output_file = os.path.join(output_dir, f'{video_name}_AOI.avi')

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (1920, 1080))

        # Loop through each frame of the video
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Get the coordinates of the bounding points for this frame
            values = df.loc[i].values[1:]
            if '  COLLISION' in values:
                continue
            if 'bdFixedNull' in video_name and 'noveh' in video_name:
                label = "DD-CR"
                left_x, left_y, right_x, right_y, height_x, height_y = map(float, values[6:])
                x1 = int(round(left_x))
                x2 = int(round((right_x)))
                if 'rev' in video_name:
                    height_diff = right_y - height_y
                    y1 = int(round(right_y - 3 * height_diff))
                    y2 = int(round(right_y + 2 * height_diff))
                else:
                    height_diff = left_y - height_y
                    y1 = int(round(left_y - 3 * height_diff))
                    y2 = int(round(left_y + 2 * height_diff))
            else:
                continue
            # elif '  No vehicle' in values:
            #     bd_x, bd_y, left_x, left_y, right_x, right_y, height_x, height_y = map(float, values[4:])
            #     if 'rev' in video_name:
            #         x1, y1 = int(round(bd_x)), int(round(bd_y))
            #         x2, y2 = int(round(1920)), int(round(0))
            #     else:
            #         x1, y1 = int(round(bd_x)), int(round(bd_y))
            #         x2, y2 = int(round(0)), int(round(0))
            # else:
            #
            #     far_x, far_y, close_x, close_y, bd_x, bd_y, left_x, left_y, right_x, right_y, height_x, height_y = \
            #         map(float, values)
            #     if 'rev' in video_name:
            #         # Round the coordinates to integers
            #         x1, y1 = int(round(far_x)), int(round(far_y))
            #         x2, y2 = int(round(close_x - 23)), int(round(close_y))
            #     else:
            #         # Round the coordinates to integers
            #         x1, y1 = int(round(far_x)), int(round(far_y))
            #         x2, y2 = int(round(close_x + 23)), int(round(close_y))

            # Read this frame from the video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Draw a rectangle around the region to highlight
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
