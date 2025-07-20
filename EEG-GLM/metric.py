import os
import argparse
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description='Channel and Center.')
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()


def get_channel_center_lists():
    channels = ['AF3', 'AF4', 'F3', 'F4', 'FP1', 'FPZ', 'FP2', 'F5', 'F6', 'F7', 'F8',
                'FT7', 'FT8', 'T7', 'T8', 'FC1', 'FC2', 'CP1', 'CPZ', 'CP2', 
                'O1', 'OZ', 'O2', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8']
    centers = ['P1', 'N1', 'P150', 'N170', 'P200', 'P250', 'N300', 'P3a', 'P3b']
    return channels, centers


def process_all_metrics(channels, centers, folder_path, model):
    with pd.ExcelWriter(f'metrics/all-{model}.xlsx') as writer:
        for center in centers:
            channel_data = {}
            for channel in channels:
                file_name = f'{channel}-{center}.csv'
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                channel_data[channel] = df.set_index('Content')['Value']
            result_df = pd.DataFrame(channel_data).T
            result_df.to_excel(writer, sheet_name=center)


def process_single_metric(channels, centers, folder_path, metric, model):
    data = {}
    for channel in channels:
        data[channel] = {}
        for center in centers:
            file_name = f'{channel}-{center}.csv'
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            value = df[df['Content'] == metric]['Value'].values[0]
            data[channel][center] = value

    result_df = pd.DataFrame.from_dict(data, orient='index')
    metric = metric.lower().replace(' ', '_').replace('-', '_').replace('*', '_')
    result_df.to_excel(f'metrics/{model}-{metric}.xlsx')


def main():
    args = parse_arguments()
    metric = args.metric
    model = args.model
    os.makedirs('metrics', exist_ok=True)
    channels, centers = get_channel_center_lists()
    folder_path = f'results/{model}'

    if metric == 'all':
        process_all_metrics(channels, centers, folder_path, model)
    else:
        process_single_metric(channels, centers, folder_path, metric, model)


if __name__ == "__main__":
    main()