import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def preprocess_data(df, config, advanced_imputation=False):
    print("Starting data preprocessing...")
    target_column = config['data']['target_column']
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(exclude=np.number).columns

    # Imputation strategies
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    if advanced_imputation:
        print("Applying advanced imputation (e.g., iterative imputer - conceptual)")
        # In a real scenario, you'd use sklearn.impute.IterativeImputer or similar
        # For simplicity, we'll just demonstrate a different conceptual path.
        numerical_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = ColumnTransformer(
        [('num_imputer', numerical_imputer, numerical_cols),
         ('scaler', StandardScaler(), numerical_cols)],
        remainder='passthrough'
    )
    
    # Handle potential empty categorical columns
    if len(categorical_cols) > 0:
        categorical_transformer = ColumnTransformer(
            [('cat_imputer', categorical_imputer, categorical_cols),
             ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
    else:
        preprocessor = numerical_transformer # Only numerical transformer if no categorical columns

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Convert back to DataFrame (for easier debugging, though not strictly necessary for PyTorch)
    # This part can be tricky due to one-hot encoding changing column names
    # For now, just return numpy array
    print("Data preprocessing complete.")
    return X_processed, y, preprocessor # Return preprocessor for potential future use (e.g., new data)

def get_preprocessed_data(config, advanced_imputation=False):
    data_path = os.path.join(config['paths']['data_dir'], config['data']['filename'])
    df = load_data(data_path)
    X, y, _ = preprocess_data(df, config, advanced_imputation=advanced_imputation)
    X = pd.DataFrame(X) # Convert back to DataFrame for consistency with train/eval expectations
    return train_test_split(X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'])

if __name__ == '__main__':
    # This block is for simple testing of preprocess.py in isolation
    print("Running preprocess.py in test mode...")
    
    dummy_data_dir = '../data'
    dummy_data_path = os.path.join(dummy_data_dir, 'dummy_data.csv')
    os.makedirs(dummy_data_dir, exist_ok=True)

    # Create dummy CSV data with some missing values and categorical data
    dummy_df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.randint(0, 10, 100).astype(float),
        'feature_3': ['A', 'B', 'C'] * 30 + ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'target': np.random.randint(0, 2, 100)
    })
    dummy_df.loc[10:20, 'feature_1'] = np.nan
    dummy_df.loc[30, 'feature_3'] = np.nan # Introduce a missing categorical value
    dummy_df.to_csv(dummy_data_path, index=False)

    dummy_config = {
        'data': {
            'filename': 'dummy_data.csv',
            'target_column': 'target',
            'test_size': 0.2,
            'random_state': 42
        },
        'paths': {
            'data_dir': dummy_data_dir
        }
    }
    
    try:
        X_train, X_test, y_train, y_test = get_preprocessed_data(dummy_config)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        # Test advanced imputation path
        print("\nTesting advanced imputation path...")
        X_train_adv, X_test_adv, y_train_adv, y_test_adv = get_preprocessed_data(dummy_config, advanced_imputation=True)
        print(f"X_train_adv shape: {X_train_adv.shape}, y_train_adv shape: {y_train_adv.shape}")

    except Exception as e:
        print(f"Error during preprocess.py test: {e}")
    finally:
        # Clean up dummy data
        if os.path.exists(dummy_data_path):
            os.remove(dummy_data_path)
    print("preprocess.py test finished.")
