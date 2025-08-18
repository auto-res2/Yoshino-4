import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple

def generate_synthetic_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Generates synthetic classification data with controlled issues."""
    np.random.seed(42)
    num_samples = config['num_samples']
    num_features = config['num_features']
    missing_value_ratio = config['missing_value_ratio']
    imbalance_ratio = config['imbalance_ratio']

    # Generate features
    X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])

    # Introduce some categorical features
    num_categorical = min(3, num_features // 3)
    for i in range(num_categorical):
        X[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], num_samples)

    # Generate a simple target based on some features
    y = ((X[f'feature_0'] + X[f'feature_1']) > 1.0).astype(int)

    # Introduce class imbalance
    if imbalance_ratio < 0.5:
        minority_class_indices = np.where(y == 1)[0]
        majority_class_indices = np.where(y == 0)[0]
        
        num_minority_samples = int(num_samples * imbalance_ratio)
        if len(minority_class_indices) > num_minority_samples:
            # Downsample minority class if it's currently too large
            downsample_indices = np.random.choice(minority_class_indices, len(minority_class_indices) - num_minority_samples, replace=False)
            y.iloc[downsample_indices] = 0 # Change some minority to majority
        elif len(minority_class_indices) < num_minority_samples:
            # Upsample minority class (by flipping some majority)
            flip_count = num_minority_samples - len(minority_class_indices)
            if len(majority_class_indices) >= flip_count:
                flip_indices = np.random.choice(majority_class_indices, flip_count, replace=False)
                y.iloc[flip_indices] = 1

    # Introduce missing values
    for col in X.columns:
        if np.random.rand() < 0.7: # Only some columns get NaNs
            nan_indices = np.random.choice(X.index, int(num_samples * missing_value_ratio), replace=False)
            X.loc[nan_indices, col] = np.nan
    
    # Introduce unscaled features (if config asks for it)
    if config.get('unscaled_features', False):
        for col in X.select_dtypes(include=np.number).columns:
            X[col] = X[col] * np.random.uniform(1, 100) + np.random.uniform(0, 50)

    X['target'] = y
    return X

def get_data_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes data and extracts key characteristics relevant for CFAE."""
    characteristics = {}
    
    # Check for missing values
    missing_cols = df.isnull().sum()
    characteristics['has_missing_values'] = any(missing_cols > 0)
    characteristics['missing_cols_count'] = len(missing_cols[missing_cols > 0])
    
    # Check class imbalance
    if 'target' in df.columns:
        target_counts = df['target'].value_counts(normalize=True)
        characteristics['target_class_distribution'] = target_counts.to_dict()
        if len(target_counts) > 1:
            minority_ratio = target_counts.min()
            characteristics['is_imbalanced'] = minority_ratio < 0.3 # Threshold for imbalance
        else:
            characteristics['is_imbalanced'] = False
    
    # Check for unscaled numerical features (simple check)
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('target', errors='ignore')
    if not numeric_cols.empty:
        max_vals = df[numeric_cols].max()
        min_vals = df[numeric_cols].min()
        characteristics['has_unscaled_features'] = any((max_vals - min_vals) > 100) # Arbitrary large range
    else:
        characteristics['has_unscaled_features'] = False
    
    return characteristics

def apply_preprocessing(df: pd.DataFrame, adaptations: Dict[str, Any]) -> Tuple[Any, pd.Series, Any]:
    """
    Splits data into features (X) and target (y), then applies preprocessing steps.
    Adaptations can modify the preprocessing logic.
    """
    X = df.drop('target', axis=1)
    y = df['target']

    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include='object').columns

    numerical_transformer_steps = []
    
    # Imputation
    if adaptations.get('impute_missing', False):
        print("Preprocessing: Applying missing value imputation.")
        numerical_transformer_steps.append(('imputer', SimpleImputer(strategy='mean')))
    else:
        # If no explicit imputation, fill with a placeholder to avoid pipeline errors on NaNs
        # Or remove this if the data is guaranteed to not have NaNs when no imputation is applied.
        # For robustness in demo, keep a default imputer.
        numerical_transformer_steps.append(('imputer_default', SimpleImputer(strategy='constant', fill_value=0)))

    # Scaling
    if adaptations.get('scale_features', False):
        print("Preprocessing: Applying feature scaling.")
        numerical_transformer_steps.append(('scaler', StandardScaler()))

    numerical_transformer = Pipeline(steps=numerical_transformer_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute categorical NaNs
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    
    print("Fitting and transforming data...")
    X_processed = preprocessor.fit_transform(X)
    
    print("Data preprocessing complete.")
    return X_processed, y, preprocessor

if __name__ == '__main__':
    # Simple test run
    dummy_config = {
        'num_samples': 100,
        'num_features': 5,
        'missing_value_ratio': 0.1,
        'imbalance_ratio': 0.2,
        'unscaled_features': True
    }
    
    print("--- Preprocessing Test Run ---")
    data = generate_synthetic_data(dummy_config)
    print("Generated data sample:")
    print(data.head())
    
    chars = get_data_characteristics(data)
    print("\nData Characteristics:")
    print(chars)
    
    adaptations_test = {
        'impute_missing': True,
        'scale_features': True
    }
    
    X_proc, y_proc, preprocessor_obj = apply_preprocessing(data, adaptations_test)
    print(f"\nProcessed X shape: {X_proc.shape}")
    print(f"Processed y shape: {y_proc.shape}")
    print("--- Preprocessing Test Complete ---")