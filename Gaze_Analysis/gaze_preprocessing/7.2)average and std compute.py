import pandas as pd
import os
import numpy as np


def process_csv_files(folder_path, output_path):
    # Define expected conditions
    expected_conditions = [
        'Control_real', 'OcclusionOnly_real', 'HazardOnly_real', 'OccludedHazard_real',
        'Control_virtual', 'OcclusionOnly_virtual', 'HazardOnly_virtual', 'OccludedHazard_virtual'
    ]

    # Lists to store results
    file_results = []
    unexpected_conditions = set()

    # Create column names for the result DataFrame
    columns = ['Filename']
    for condition in expected_conditions:
        columns.extend([
            f'{condition}_X_Average',
            f'{condition}_X_Std',
            f'{condition}_Y_Average',
            f'{condition}_Y_Std'
        ])

    # Process each CSV file
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Check for unexpected conditions
            unexpected = set(df['Condition']) - set(expected_conditions)
            if unexpected:
                unexpected_conditions.update(unexpected)
                print(f"Warning: Unexpected conditions found in {filename}: {unexpected}")

            # Initialize result dictionary for this file
            file_result = {'Filename': filename}

            # Calculate statistics for each condition
            for condition in expected_conditions:
                condition_data = df[df['Condition'] == condition]

                if len(condition_data) > 0:
                    file_result[f'{condition}_X_Average'] = condition_data['X_Average'].mean()
                    file_result[f'{condition}_X_Std'] = condition_data['X_Std'].mean()
                    file_result[f'{condition}_Y_Average'] = condition_data['Y_Average'].mean()
                    file_result[f'{condition}_Y_Std'] = condition_data['Y_Std'].mean()
                else:
                    file_result[f'{condition}_X_Average'] = np.nan
                    file_result[f'{condition}_X_Std'] = np.nan
                    file_result[f'{condition}_Y_Average'] = np.nan
                    file_result[f'{condition}_Y_Std'] = np.nan

            file_results.append(file_result)

    # Create DataFrame with all file results
    results_df = pd.DataFrame(file_results)

    # Calculate overall statistics and add them as new rows
    overall_mean = {'Filename': 'Average'}
    overall_std = {'Filename': 'Std'}

    # Calculate statistics for each metric column (excluding Filename)
    for col in results_df.columns:
        if col != 'Filename':
            overall_mean[col] = results_df[col].mean()
            overall_std[col] = results_df[col].std()

    # Append overall statistics to results DataFrame
    results_df = pd.concat([results_df,
                            pd.DataFrame([overall_mean]),
                            pd.DataFrame([overall_std])],
                           ignore_index=True)

    # Save results
    output_file = os.path.join(output_path, 'condition_analysis_results.csv')
    results_df.to_csv(output_file, index=False)

    # Print summary
    if unexpected_conditions:
        print("\nWarning: The following unexpected conditions were found:")
        for cond in unexpected_conditions:
            print(f"- {cond}")

    print(f"\nResults saved to {output_file}")

    return results_df


# Example usage:
folder_path = "M:\\EEG_DATA\\EEG_data_0410\\average_and_std\\Nov"
output_path = "M:\\EEG_DATA\\EEG_data_0410\\average_and_std"
results_df = process_csv_files(folder_path, output_path)