import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Union

def generate_synthetic_data(n_samples: int = 500, n_features: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset for AMMEF 2.0 experiments."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_clusters_per_class=1,
        random_state=random_state
    )
    return X, y

def preprocess_data(X: np.ndarray, y: np.ndarray, scale_features: bool = True) -> Tuple[Union[np.ndarray, Any], Union[np.ndarray, Any], Union[np.ndarray, Any], Union[np.ndarray, Any]]:
    """Preprocess data by splitting and optionally scaling features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def introduce_adversarial_labels(y: np.ndarray, flip_ratio: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Introduce adversarial label flips for EMAF-RLG experiment."""
    np.random.seed(random_state)
    y_adv = y.copy()
    n_flip = int(len(y) * flip_ratio)
    flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
    y_adv[flip_indices] = 1 - y_adv[flip_indices]
    return y_adv

if __name__ == '__main__':
    print("--- Preprocessing Test Run ---")
    X, y = generate_synthetic_data(n_samples=100, n_features=10)
    print(f"Generated data shapes: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Split data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    y_adv = introduce_adversarial_labels(y, flip_ratio=0.1)
    print(f"Adversarial labels created with {np.sum(y != y_adv)} flips")
    print("--- Preprocessing Test Complete ---")
