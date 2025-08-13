import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_mock_data(scenario="default"):
    np.random.seed(42)
    size = 100
    if scenario == "small_noisy_data":
        size = 20 # Smaller dataset
    
    data = pd.DataFrame({
        'feature1': np.random.rand(size) * 100,
        'feature2': np.random.randint(0, 5, size),
        'target': np.random.rand(size) * 10
    })

    if scenario == "non_numeric_for_scaling":
        data['feature1'] = data['feature1'].astype(str) # Introduce non-numeric
    elif scenario == "missing_values":
        data.loc[np.random.choice(data.index, int(0.1*size)), 'feature1'] = np.nan # 10% missing
    elif scenario == "imbalanced_target":
        data['target'] = np.random.choice([0, 1], size=size, p=[0.9, 0.1]) # Simulate imbalance
    elif scenario == "small_noisy_data":
        # Introduce more noise for 'feature1'
        data['feature1'] = data['feature1'] + np.random.randn(size) * 50
        data['target'] = (data['feature1'] + data['feature2'] + np.random.randn(size)*20 > 100).astype(int)
    
    return data

def preprocess_data(data: pd.DataFrame, params: dict, task_context: dict) -> pd.DataFrame:
    logging.info(f"Preprocessing data for task '{task_context.get('task_id', 'N/A')}' with params: {params}")

    processed_data = data.copy()
    
    # Simulate missing value handling
    if params.get('handle_missing'):
        logging.info("  Handling missing values...")
        if processed_data.isnull().sum().sum() > 0:
            processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
        else:
            logging.info("  No missing values found.")
    else:
        # If not handled, check for critical missing values that might cause downstream errors
        if processed_data.isnull().sum().sum() > 0 and task_context.get('simulate_preprocess_error', False) and task_context.get('preprocess_error_type') == "unhandled_missing":
             raise ValueError("PreprocessingError: Unhandled missing values detected.")

    # Simulate scaling
    if params.get('scale_data'):
        logging.info("  Scaling data...")
        numeric_cols = processed_data.select_dtypes(include=np.number).columns
        non_numeric_cols = processed_data.select_dtypes(exclude=np.number).columns
        
        if len(non_numeric_cols) > 0 and task_context.get('simulate_preprocess_error', False) and task_context.get('preprocess_error_type') == "non_numeric_for_scaling":
            logging.error(f"PreprocessingError: Non-numeric columns {list(non_numeric_cols)} found for scaling. Raising error.")
            raise TypeError(f"PreprocessingError: Non-numeric columns {list(non_numeric_cols)} detected, cannot scale.")
        
        for col in numeric_cols:
            if processed_data[col].std() > 0:
                processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
            else:
                processed_data[col] = 0 # Avoid division by zero for constant columns
    
    # Simulate adaptive outlier handling
    if params.get('adaptive_outlier_handling'):
        logging.info("  Applying adaptive outlier handling...")
        # Simple percentile based outlier removal for demonstration
        for col in processed_data.select_dtypes(include=np.number).columns:
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Replace outliers with median or cap them
            processed_data[col] = np.where(processed_data[col] < lower_bound, processed_data[col].median(), processed_data[col])
            processed_data[col] = np.where(processed_data[col] > upper_bound, processed_data[col].median(), processed_data[col])
        logging.info("  Outlier handling applied.")

    logging.info("Preprocessing complete.")
    return processed_data

if __name__ == '__main__':
    # Simple test run
    print("--- Preprocess Test Run ---")
    
    # Scenario 1: Default successful run
    print("\nScenario 1: Default successful run")
    data_default = generate_mock_data()
    params_default = {"scale_data": True, "handle_missing": True}
    processed_data_default = preprocess_data(data_default, params_default, {"task_id": "test_default"})
    print(f"Processed data shape: {processed_data_default.shape}")
    print(f"Processed data head:\n{processed_data_default.head()}")

    # Scenario 2: Simulate non-numeric error
    print("\nScenario 2: Simulate non-numeric for scaling error")
    data_non_numeric = generate_mock_data(scenario="non_numeric_for_scaling")
    params_non_numeric = {"scale_data": True, "handle_missing": False}
    try:
        preprocess_data(data_non_numeric, params_non_numeric, {"task_id": "test_non_numeric", "simulate_preprocess_error": True, "preprocess_error_type": "non_numeric_for_scaling"})
    except TypeError as e:
        print(f"Caught expected error: {e}")

    # Scenario 3: Simulate missing values error
    print("\nScenario 3: Simulate unhandled missing values error")
    data_missing = generate_mock_data(scenario="missing_values")
    params_missing = {"scale_data": False, "handle_missing": False} # Not handling missing
    try:
        preprocess_data(data_missing, params_missing, {"task_id": "test_missing", "simulate_preprocess_error": True, "preprocess_error_type": "unhandled_missing"})
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Scenario 4: Adaptive outlier handling
    print("\nScenario 4: Adaptive outlier handling")
    data_outlier = generate_mock_data()
    # Manually inject some outliers
    data_outlier.loc[0, 'feature1'] = 1000.0
    data_outlier.loc[1, 'feature1'] = -500.0
    params_outlier = {"scale_data": True, "handle_missing": True, "adaptive_outlier_handling": True}
    processed_data_outlier = preprocess_data(data_outlier, params_outlier, {"task_id": "test_outlier"})
    print(f"Processed data with outlier handling head:\n{processed_data_outlier.head()}")
    
