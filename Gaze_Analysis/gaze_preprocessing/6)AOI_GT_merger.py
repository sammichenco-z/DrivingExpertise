import os
import cv2
import pandas as pd

# Set the path to your video files directory
video_dir = 'M:\\EEG_DATA\\实际实验Ver\\virtual'
AOI_path = 'M:\\EEG_DATA\\all_box_info.csv'


def is_aoi_valid(x1, y1, x2, y2, pre_flag, max_width, max_height):
    if pre_flag:
        if 0 <= x1 <= max_width and 0 <= x2 <= max_width and 0 <= y1 <= max_height and 0 <= y2 <= max_height:
            return True
        else:
            return False
    else:
        return False


aoi = pd.read_csv(AOI_path)

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

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        flag_dd_cr = True
        flag_occlusion_or = True
        flag_vehicle = True

        # Loop through each frame of the video
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Get the coordinates of the bounding points for this frame
            values = df.loc[i].values[1:]
            if '  COLLISION' in values:
                break
            if 'bdFixedNull' in video_name and 'noveh' in video_name:
                condition = 'Control'
            elif 'bdFixedNull' in video_name:
                condition = 'HazardOnly'
            elif 'noveh' in video_name:
                condition = 'OcclusionOnly'
            else:
                condition = 'OccludedHazard'
            label_dd = "DD-CR"
            label_oc = "Occlusion-OR"
            bd_x, bd_y, left_x, left_y, right_x, right_y, height_x, height_y = map(float, values[4:])
            dd_x1 = int(round(left_x))
            dd_x2 = int(round(right_x))
            if 'rev' in video_name:
                height_diff = right_y - height_y
                dd_y1 = int(round(right_y - 3 * height_diff))
                dd_y2 = int(round(right_y + 2 * height_diff))
                bd_x1, bd_y1 = int(round(bd_x)), int(round(bd_y))
                bd_x2, bd_y2 = int(round(width)), int(round(0))  # Changed from 1920 to width
            else:
                height_diff = left_y - height_y
                dd_y1 = int(round(left_y - 3 * height_diff))
                dd_y2 = int(round(left_y + 2 * height_diff))
                bd_x1, bd_y1 = int(round(bd_x)), int(round(bd_y))
                bd_x2, bd_y2 = int(round(0)), int(round(0))

            if condition == 'OccludedHazard' or condition == 'HazardOnly':
                label = "Vehicle-OHR"
                far_x, far_y, close_x, close_y, bd_x, bd_y, left_x, left_y, right_x, right_y, height_x, height_y = \
                    map(float, values)
                if 'rev' in video_name:
                    x1, y1 = int(round(far_x)), int(round(far_y))
                    x2, y2 = int(round(close_x - 23)), int(round(close_y))
                else:
                    x1, y1 = int(round(far_x)), int(round(far_y))
                    x2, y2 = int(round(close_x + 23)), int(round(close_y))

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                flag_dd_cr = is_aoi_valid(dd_x1, dd_y1, dd_x2, dd_y2, flag_dd_cr, width, height)
                # Only check occlusion flag if it's not HazardOnly condition
                if condition != 'HazardOnly':
                    flag_occlusion_or = is_aoi_valid(bd_x1, bd_y1, bd_x2, bd_y2, flag_occlusion_or, width, height)
                # Check vehicle flag for both OccludedHazard and HazardOnly conditions
                flag_vehicle = is_aoi_valid(x1, y1, x2, y2, flag_vehicle, width, height) if (condition == 'OccludedHazard' or condition == 'HazardOnly') else False

                if flag_dd_cr:
                    new_aoi_dd = pd.DataFrame({
                        'Name': [video_name],
                        'Image ID': [i],
                        'Label': [label_dd],
                        'xtl': [dd_x1],
                        'ytl': [dd_y1],
                        'xbr': [dd_x2],
                        'ybr': [dd_y2],
                        'width': [width],
                        'height': [height],
                        'Condition': condition
                    })
                    aoi = pd.concat([aoi, new_aoi_dd], ignore_index=True)

                if flag_occlusion_or and condition != 'HazardOnly':
                    new_aoi_oc = pd.DataFrame({
                        'Name': [video_name],
                        'Image ID': [i],
                        'Label': [label_oc],
                        'xtl': [bd_x1],
                        'ytl': [bd_y1],
                        'xbr': [bd_x2],
                        'ybr': [bd_y2],
                        'width': [width],
                        'height': [height],
                        'Condition': condition
                    })
                    aoi = pd.concat([aoi, new_aoi_oc], ignore_index=True)

                if flag_vehicle and (condition == 'OccludedHazard' or condition == 'HazardOnly'):
                    new_aoi = pd.DataFrame({
                        'Name': [video_name],
                        'Image ID': [i],
                        'Label': [label],
                        'xtl': [x1],
                        'ytl': [y1],
                        'xbr': [x2],
                        'ybr': [y2],
                        'width': [width],
                        'height': [height],
                        'Condition': condition
                    })
                    aoi = pd.concat([aoi, new_aoi], ignore_index=True)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

aoi.to_csv("M:\\EEG_DATA\\new_full_AOI.csv")