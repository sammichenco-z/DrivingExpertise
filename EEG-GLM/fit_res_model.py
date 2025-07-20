import os
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import statsmodels.api as sm
from scipy.stats import ttest_1samp, f


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Type, Channel and Center.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--channel', type=str, required=True)
    parser.add_argument('--center', type=str, required=True)
    parser.add_argument('--type', type=str, required=True,
                        choices=['full', 'occlusion_reduced', 'hazard_occlusion_reduced'])
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_merge_data(args):
    """Load and merge raw data from CSV files"""
    channel = args.channel
    center = args.center
    
    # Load raw data
    X_raw = pd.read_csv(f'X/{center}.csv')
    y_raw = pd.read_csv(f'y/{channel}-{center}.csv')
    
    # Process data based on analysis type
    if args.type == 'full':
        y_raw.insert(0, 'expert*occlusion', y_raw['expert'] * y_raw['occlusion'])
        y_raw.insert(0, 'expert*hazard', y_raw['expert'] * y_raw['hazard'])
    elif args.type == 'occlusion_reduced':
        y_raw.insert(0, 'expert*hazard', y_raw['expert'] * y_raw['hazard'])
    
    # Merge datasets and filter
    merged_data_raw = pd.merge(y_raw, X_raw, on='video_name', how='left')
    merged_data_select = merged_data_raw[~((merged_data_raw['hazard'] == 1) & (merged_data_raw['occlusion'] == 1))]
    merged_data_select = merged_data_select.sort_values(by='video_name').reset_index(drop=True)
    X_select = merged_data_select[X_raw.columns]
    y_select = merged_data_select[y_raw.columns]
    
    return X_select, y_select, merged_data_raw.columns


def normalize_data(X_select, y_select):
    """Normalize numerical features using StandardScaler"""
    # Normalize X features
    scaler_X = StandardScaler()
    num_cols = X_select.select_dtypes(include='number').columns
    str_cols = X_select.select_dtypes(exclude='number').columns
    X_ignore_0 = X_select[num_cols].replace(0, np.nan)
    standard_data = scaler_X.fit_transform(X_ignore_0)
    standard_data = pd.DataFrame(standard_data, columns=num_cols).fillna(0)
    X_standard = pd.concat([X_select[str_cols], standard_data], axis=1)
    
    # Normalize y values
    scaler_y = StandardScaler()
    y_value = y_select[['value']]
    value_data = scaler_y.fit_transform(y_value)
    y_standard = y_select.copy()
    y_standard['value'] = value_data
    
    return X_standard, y_standard


def prepare_final_data(X_standard, y_standard, args):
    """Prepare data for full and reduced models based on analysis type"""
    # Merge and process data
    merged_data = pd.merge(y_standard, X_standard, on='video_name', how='left')
    merged_data = merged_data.sort_values(by='video_name').reset_index(drop=True).drop('video_name', axis=1)
    
    # Create full and reduced datasets
    if args.type == 'full':
        full_data = merged_data
        reduced_data = merged_data.drop(columns=['expert'])
    elif args.type == 'occlusion_reduced':
        full_data = merged_data.drop(columns=['occlusion'])
        reduced_data = merged_data.drop(columns=['expert', 'occlusion'])
    elif args.type == 'hazard_occlusion_reduced':
        full_data = merged_data.drop(columns=['occlusion', 'hazard'])
        reduced_data = merged_data.drop(columns=['expert', 'occlusion', 'hazard'])
    
    return full_data, reduced_data


def get_predictions(data):
    """Perform cross-validation and return predictions and model list"""
    X = data.drop('value', axis=1)
    y = data['value']
    predictions = []
    model_list = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train).fit()
        y_test_pred = model.predict(X_test)
        predictions.extend(y_test_pred.values.tolist())
        model_list.append(model)

    return predictions, model_list


def calculate_t_test(full_predictions, reduced_predictions):
    """Calculate t-test statistics for residual predictions"""
    residual_predictions = np.array(full_predictions) - np.array(reduced_predictions)
    t_statistic, p_value_t = ttest_1samp(residual_predictions, 0)
    degrees_of_freedom = len(residual_predictions) - 1
    effect_size = np.mean(residual_predictions) / np.std(residual_predictions, ddof=1)
    
    return t_statistic, degrees_of_freedom, p_value_t, effect_size, residual_predictions


def calculate_f_test(full_model_list, reduced_model_list, full_data):
    """Calculate F-test statistics for model comparison"""
    f_statistic_list, p_value_f_list = [], []
    
    for i in range(5):
        p = 1  # Number of predictors removed
        k = full_data.shape[1] - 2  # Number of predictors in full model
        n = len(full_data)  # Sample size
        rss_full = full_model_list[i].ssr  # Residual sum of squares for full model
        rss_reduced = reduced_model_list[i].ssr  # Residual sum of squares for reduced model
        
        # Calculate F-statistic
        f_statistic = ((rss_reduced - rss_full) / p) / (rss_full / (n - k - 1))
        p_value_f = 1 - f.cdf(f_statistic, dfn=p, dfd=n - k - 1)
        
        f_statistic_list.append(f_statistic)
        p_value_f_list.append(p_value_f)
    
    # Calculate average F-statistic and p-value
    f_statistic = sum(f_statistic_list) / len(f_statistic_list)
    p_value_f = sum(p_value_f_list) / len(p_value_f_list)
    
    return f_statistic, p_value_f, p, k, n, rss_full, rss_reduced


def prepare_results(t_statistic, degrees_of_freedom, p_value_t, effect_size, 
                    f_statistic, p_value_f, p, k, n, rss_full, rss_reduced):
    """Prepare results data for saving"""
    test_results = []
    test_results.append(('T statistic', t_statistic))
    test_results.append(('Degrees of freedom', degrees_of_freedom))
    test_results.append(('T test P value', p_value_t))
    test_results.append(('Effect size', effect_size))
    test_results.append(('Number of predictors removed (p)', p))
    test_results.append(('Total predictors in full model (k)', k))
    test_results.append(('Sample size (n)', n))
    test_results.append(('Residual sum of squares (full model) (RSS_full)', rss_full))
    test_results.append(('Residual sum of squares (reduced model) (RSS_reduced)', rss_reduced))
    test_results.append(('F statistic', f_statistic))
    test_results.append(('F test P value', p_value_f))
    
    return test_results


def save_results(test_results, args):
    """Save results to CSV file"""
    result_df = pd.DataFrame(test_results, columns=['Content', 'Value'])
    os.makedirs(f'results/{args.type}_residual', exist_ok=True)
    result_df.to_csv(f'results/{args.type}_residual/{args.channel}-{args.center}.csv', index=False)


def main():
    """Main function to coordinate data processing and analysis"""
    # Parse command line arguments
    args = get_args()
    print(f'Fit residual model for {args.type}-{args.channel}-{args.center}.')
    
    # Load and merge data
    X_select, y_select, _ = load_and_merge_data(args)
    
    # Normalize data
    X_standard, y_standard = normalize_data(X_select, y_select)
    
    # Prepare datasets for full and reduced models
    full_data, reduced_data = prepare_final_data(X_standard, y_standard, args)
    
    # Get predictions and models
    full_predictions, full_model_list = get_predictions(full_data)
    reduced_predictions, reduced_model_list = get_predictions(reduced_data)
    
    # Calculate t-test statistics
    t_statistic, degrees_of_freedom, p_value_t, effect_size, _ = calculate_t_test(full_predictions, reduced_predictions)
    
    # Calculate F-test statistics
    f_statistic, p_value_f, p, k, n, rss_full, rss_reduced = calculate_f_test(full_model_list, reduced_model_list, full_data)
    
    # Prepare results
    test_results = prepare_results(t_statistic, degrees_of_freedom, p_value_t, effect_size, 
                                  f_statistic, p_value_f, p, k, n, rss_full, rss_reduced)
    
    # Save results
    save_results(test_results, args)


if __name__ == "__main__":
    main()