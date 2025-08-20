import numpy as np
import random

def generate_streaming_data(n_steps=500, n_features=2):
    """Generate synthetic streaming data with domain shift for ARDSON experiments."""
    data = []
    labels = []
    for t in range(n_steps):
        mean = np.array([np.sin(0.01 * t) * 5, np.cos(0.01 * t) * 5])
        cov = np.eye(n_features) * (1 + 0.5 * np.sin(0.02 * t))
        point = np.random.multivariate_normal(mean, cov)
        data.append(point)
        labels.append(0 if t < n_steps/2 else 1)
    return np.array(data), np.array(labels)
