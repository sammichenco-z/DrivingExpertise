import pandas as pd


def scale_coordinates(file_path, save_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Find rows where width is not 1920
    mask = df['width'] != 1920

    # Calculate width ratio for scaling
    width_ratio = 1920 / df.loc[mask, 'width']

    # Scale xtl and xbr
    df.loc[mask, 'xtl'] = df.loc[mask, 'xtl'] * width_ratio
    df.loc[mask, 'xbr'] = df.loc[mask, 'xbr'] * width_ratio
    # Set width to 1920
    df.loc[mask, 'width'] = 1920

    # Handle height scaling
    height_mask = df['height'] != 1080
    height_ratio = 1080 / df.loc[height_mask, 'height']

    # Scale ytl and ybr
    df.loc[height_mask, 'ytl'] = df.loc[height_mask, 'ytl'] * height_ratio
    df.loc[height_mask, 'ybr'] = df.loc[height_mask, 'ybr'] * height_ratio
    # Set height to 1080
    df.loc[height_mask, 'height'] = 1080

    # Save the modified dataframe
    df.to_csv(save_path, index=False)

    # Print some statistics
    print(f"Total rows processed: {len(df)}")
    print(f"Rows with width scaled: {mask.sum()}")
    print(f"Rows with height scaled: {height_mask.sum()}")

# Usage example:
file_path = "M:\\EEG_DATA\\new_full_AOI.csv"
save_path = "M:\\EEG_DATA\\scale_full_AOI.csv"
scale_coordinates(file_path, save_path)