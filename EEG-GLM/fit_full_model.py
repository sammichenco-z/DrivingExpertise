import os
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr


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
    """Load and merge data from CSV files"""
    channel = args.channel
    center = args.center
    
    # Load raw data
    X_raw = pd.read_csv(f'X/{center}.csv')
    y_raw = pd.read_csv(f'y/{channel}-{center}.csv')
    
    # Process data based on the type argument
    if args.type == 'full':
        y_raw.insert(0, 'expert*occlusion', y_raw['expert'] * y_raw['occlusion'])
        y_raw.insert(0, 'expert*hazard', y_raw['expert'] * y_raw['hazard'])
    elif args.type == 'occlusion_reduced':
        y_raw.insert(0, 'expert*hazard', y_raw['expert'] * y_raw['hazard'])
    
    # Merge and filter data
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


def prepare_final_data(X_standard, y_standard, args, original_columns):
    """Prepare final dataset for modeling by dropping specified columns"""
    # Merge and reduce features
    merged_data = pd.merge(y_standard, X_standard, on='video_name', how='left')
    merged_data = merged_data.sort_values(by='video_name').reset_index(drop=True).drop('video_name', axis=1)
    
    # Drop specific columns based on the type argument
    if args.type == 'full':
        data = merged_data
    elif args.type == 'occlusion_reduced':
        data = merged_data.drop(columns=['occlusion'])
    elif args.type == 'hazard_occlusion_reduced':
        data = merged_data.drop(columns=['occlusion', 'hazard'])
    
    X = data.drop('value', axis=1)
    y = data['value']
    
    return X, y


def perform_cross_validation(X, y):
    """Perform 5-fold cross-validation using Ordinary Least Squares (OLS) regression"""
    train_mse_scores = []
    test_mse_scores = []
    r2_scores = []
    aic_scores = []
    bic_scores = []
    model_p_values = []
    model_effect_sizes = []
    targets = []
    predictions = []
    feature_coeffs = {col: [] for col in X.columns}
    feature_p_values = {col: [] for col in X.columns}
    feature_effect_sizes = {col: [] for col in X.columns}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        
        model = sm.OLS(y_train, X_train).fit()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        targets.extend(y_test.values.tolist())
        predictions.extend(y_test_pred.values.tolist())
        
        # Calculate evaluation metrics
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        test_mse = np.mean((y_test - y_test_pred) ** 2)
        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)
        
        r2 = model.rsquared
        r2_scores.append(r2)
        
        aic = model.aic
        bic = model.bic
        aic_scores.append(aic)
        bic_scores.append(bic)
        
        model_p_value = model.f_pvalue
        model_p_values.append(model_p_value)
        
        model_effect_size = model.fvalue
        model_effect_sizes.append(model_effect_size)
        
        # Collect feature coefficients and p-values
        for col in X.columns:
            feature_coeffs[col].append(model.params[col])
            feature_p_values[col].append(model.pvalues[col])
            feature_effect_sizes[col].append(model.params[col])
    
    return (train_mse_scores, test_mse_scores, r2_scores, aic_scores, bic_scores, 
            model_p_values, model_effect_sizes, targets, predictions, 
            feature_coeffs, feature_p_values, feature_effect_sizes)


def calculate_statistics(train_mse_scores, test_mse_scores, r2_scores, aic_scores, bic_scores, 
                         model_p_values, model_effect_sizes, targets, predictions, 
                         feature_coeffs, feature_p_values, feature_effect_sizes, X):
    """Calculate final statistics and aggregate results"""
    average_train_mse = np.mean(train_mse_scores)
    average_test_mse = np.mean(test_mse_scores)
    average_r2 = np.mean(r2_scores)
    average_aic = np.mean(aic_scores)
    average_bic = np.mean(bic_scores)
    average_model_p_value = np.mean(model_p_values)
    average_model_effect_size = np.mean(model_effect_sizes)
    spearman_corr, spearman_p = spearmanr(predictions, targets)
    pearson_corr, pearson_p = pearsonr(predictions, targets)
    feature_coeff_means = {col: np.mean(coeffs) for col, coeffs in feature_coeffs.items()}
    feature_coeff_vars = {col: np.var(coeffs) for col, coeffs in feature_coeffs.items()}
    feature_p_value_means = {col: np.mean(p_values) for col, p_values in feature_p_values.items()}
    feature_effect_size_means = {col: np.mean(effect_sizes) for col, effect_sizes in feature_effect_sizes.items()}
    
    results = []
    results.append(('Average Train MSE', average_train_mse))
    results.append(('Average Test MSE', average_test_mse))
    results.append(('Average R-squared', average_r2))
    results.append(('Average AIC', average_aic))
    results.append(('Average BIC', average_bic))
    results.append(('Average Model Significance P-value', average_model_p_value))
    results.append(('Average Model Significance Effect Size', average_model_effect_size))
    results.append(('Spearman Correlation Coefficient', spearman_corr))
    results.append(('Spearman Correlation P-value', spearman_p))
    results.append(('Pearson Correlation Coefficient', pearson_corr))
    results.append(('Pearson Correlation P-value', pearson_p))
    
    for col in X.columns:
        results.append((f'{col} Coefficient Mean', feature_coeff_means[col]))
        results.append((f'{col} Coefficient Variance', feature_coeff_vars[col]))
        results.append((f'{col} P-value Mean', feature_p_value_means[col]))
        results.append((f'{col} Effect Size Mean', feature_effect_size_means[col]))
    
    return results


def save_results(results, args):
    """Save results to a CSV file"""
    result_df = pd.DataFrame(results, columns=['Content', 'Value'])
    os.makedirs(f'results/{args.type}_full', exist_ok=True)
    result_df.to_csv(f'results/{args.type}_full/{args.channel}-{args.center}.csv', index=False)


def main():
    """Main function to coordinate the execution of all modules"""
    # Parse command line arguments
    args = get_args()
    print(f'Fit full model for {args.type}-{args.channel}-{args.center}.')
    
    # Load and merge data
    X_select, y_select, original_columns = load_and_merge_data(args)
    
    # Normalize data
    X_standard, y_standard = normalize_data(X_select, y_select)
    
    # Prepare final dataset
    X, y = prepare_final_data(X_standard, y_standard, args, original_columns)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(X, y)
    
    # Calculate statistics
    results = calculate_statistics(*cv_results, X)
    
    # Save results
    save_results(results, args)


if __name__ == "__main__":
    main()