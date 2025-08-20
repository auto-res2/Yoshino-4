import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Union

def generate_streaming_data(total_steps: int = 300, shift_step: int = 150, feature_dim: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic streaming data with domain shift and adversarial perturbations."""
    X_stream = []
    y_stream = []
    
    np.random.seed(42)  # For reproducibility
    
    for t in range(total_steps):
        if t < shift_step:
            x = np.random.randn(feature_dim)
            label = int(np.sum(x) > 0)  # simple decision boundary
        else:
            x = np.random.randn(feature_dim) + 2
            label = int(np.sum(x) > 10)  
        
        if (t % 50 == 0) and (t > 0):
            x += np.sign(x) * 3
            
        X_stream.append(x)
        y_stream.append(label)
    
    return np.array(X_stream), np.array(y_stream)

def generate_batch_data(n_samples: int = 500, n_features: int = 10, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic batch classification dataset."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=max(2, n_features//2),
        n_redundant=max(0, n_features//4),
        n_clusters_per_class=1,
        n_classes=2,
        random_state=random_state
    )
    return X, y

def preprocess_streaming_data(X_stream: np.ndarray, y_stream: np.ndarray, 
                             normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess streaming data for online learning."""
    if normalize:
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_stream)
        return X_normalized, y_stream
    return X_stream, y_stream

def preprocess_batch_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                         scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess batch data by splitting and optionally scaling features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def introduce_adversarial_perturbations(X: np.ndarray, perturbation_strength: float = 0.1, 
                                       perturbation_ratio: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Introduce adversarial perturbations to input features."""
    np.random.seed(random_state)
    X_adv = X.copy()
    n_perturb = int(len(X) * perturbation_ratio)
    perturb_indices = np.random.choice(len(X), size=n_perturb, replace=False)
    
    for idx in perturb_indices:
        perturbation = np.random.randn(X.shape[1]) * perturbation_strength
        X_adv[idx] += perturbation
    
    return X_adv

def introduce_label_noise(y: np.ndarray, flip_ratio: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Introduce label noise by flipping a fraction of labels."""
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_flip = int(len(y) * flip_ratio)
    flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return y_noisy

def create_domain_shift_data(X_base: np.ndarray, y_base: np.ndarray, 
                            shift_type: str = 'mean_shift', shift_magnitude: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create data with domain shift for robustness testing."""
    X_shifted = X_base.copy()
    y_shifted = y_base.copy()
    
    if shift_type == 'mean_shift':
        X_shifted += shift_magnitude
    elif shift_type == 'covariance_shift':
        X_shifted = X_shifted * shift_magnitude
    elif shift_type == 'label_shift':
        y_shifted = (np.sum(X_shifted, axis=1) > shift_magnitude).astype(int)
    
    return X_shifted, y_shifted

def simulate_concept_drift(X_stream: np.ndarray, y_stream: np.ndarray, 
                          drift_points: list, drift_types: list) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate concept drift at specified time points."""
    X_drift = X_stream.copy()
    y_drift = y_stream.copy()
    
    for i, (drift_point, drift_type) in enumerate(zip(drift_points, drift_types)):
        if drift_point < len(X_stream):
            if drift_type == 'sudden':
                X_drift[drift_point:] += 1.0
            elif drift_type == 'gradual':
                for t in range(drift_point, len(X_stream)):
                    progress = (t - drift_point) / (len(X_stream) - drift_point)
                    X_drift[t] += progress * 1.0
    
    return X_drift, y_drift

def get_data_statistics(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Calculate basic statistics of the dataset."""
    stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': np.bincount(y),
        'feature_means': np.mean(X, axis=0),
        'feature_stds': np.std(X, axis=0),
        'feature_ranges': np.ptp(X, axis=0)
    }
    return stats

if __name__ == '__main__':
    print("--- Adaptive Preprocessing Test ---")
    
    X_stream, y_stream = generate_streaming_data(total_steps=100, shift_step=50, feature_dim=10)
    print(f"Generated streaming data: X_stream={X_stream.shape}, y_stream={y_stream.shape}")
    
    X_batch, y_batch = generate_batch_data(n_samples=200, n_features=10)
    print(f"Generated batch data: X_batch={X_batch.shape}, y_batch={y_batch.shape}")
    
    X_train, X_test, y_train, y_test = preprocess_batch_data(X_batch, y_batch)
    print(f"Preprocessed batch data: X_train={X_train.shape}, X_test={X_test.shape}")
    
    X_adv = introduce_adversarial_perturbations(X_batch, perturbation_strength=0.1)
    print(f"Adversarial perturbations added: {np.mean(np.abs(X_adv - X_batch)):.4f} mean difference")
    
    y_noisy = introduce_label_noise(y_batch, flip_ratio=0.1)
    print(f"Label noise added: {np.sum(y_batch != y_noisy)} labels flipped")
    
    stats = get_data_statistics(X_batch, y_batch)
    print(f"Data statistics: {stats['n_samples']} samples, {stats['n_features']} features, {stats['n_classes']} classes")
    
    print("--- Preprocessing Module Test Complete ---")
