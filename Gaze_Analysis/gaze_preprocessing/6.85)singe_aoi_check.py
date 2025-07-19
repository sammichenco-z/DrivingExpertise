import pandas as pd
import numpy as np


def analyze_input_data(input_path):
    # Read the data
    df = pd.read_csv(input_path)

    # Separate virtual and real data
    virtual_df = df[df['Name'].str.contains('bd', case=False)]
    real_df = df[~df['Name'].str.contains('bd', case=False)]

    print("\nOverall Data Analysis:")
    print(f"Total number of unique videos: {len(df['Name'].unique())}")

    print("\nVirtual Videos Analysis:")
    print(f"Total virtual videos: {len(virtual_df['Name'].unique())}")
    print("\nVideos per condition in virtual:")
    print(virtual_df.groupby('Condition')['Name'].nunique())

    print("\nReal Videos Analysis:")
    print(f"Total real videos: {len(real_df['Name'].unique())}")
    print("\nVideos per condition in real:")
    print(real_df.groupby('Condition')['Name'].nunique())

    # Print all control video names
    print("\nControl Videos:")
    print("\nVirtual Control Videos:")
    print(sorted(virtual_df[virtual_df['Condition'] == 'Control']['Name'].unique()))
    print("\nReal Control Videos:")
    print(sorted(real_df[real_df['Condition'] == 'Control']['Name'].unique()))


if __name__ == "__main__":
    analyze_input_data('M:\\EEG_DATA\\EEG_data_0410\\single_aoi.csv')