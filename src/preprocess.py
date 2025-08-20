"""
AutoMind++ Framework - Data Preprocessing Module
Implements multi-modal data synthesis and preprocessing for the experimental framework.
"""

import io
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def generate_multimodal_data(num_samples=50, seed=42):
    """
    Generate synthetic multi-modal data for Experiment 1.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing texts, plot_images, stat_summaries, and noise_flags
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    texts = []
    plot_images = []
    stat_summaries = []
    noise_flags = []

    for i in range(num_samples):
        text = f"This is sample {i}. It has a {'positive' if i % 2 == 0 else 'negative'} outlook."
        noise_text = random.random() < 0.2
        if noise_text:
            text = text.replace('positive', '???').replace('negative', '???')
        texts.append(text)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        data = np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        ax.plot(data, label='Signal')
        ax.legend()
        ax.set_title(f'Sample {i} Signal')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img = torch.rand(3, 224, 224)  # simulate a preprocessed plot image
        noise_plot = random.random() < 0.2
        if noise_plot:
            img += torch.rand_like(img) * 0.5  # add extra noise
        plot_images.append(img)

        data_for_df = np.random.randn(100) + i * 0.01  # slight drift
        df = pd.DataFrame({'value': data_for_df})
        summary = {
            'mean': df['value'].mean(),
            'std': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max()
        }
        noise_stats = random.random() < 0.2
        if noise_stats:
            summary['std'] *= 3
        stat_summaries.append(summary)

        noise_flags.append({
            'text_noise': noise_text,
            'plot_noise': noise_plot,
            'stats_noise': noise_stats
        })

    return {
        'texts': texts,
        'plot_images': plot_images,
        'stat_summaries': stat_summaries,
        'noise_flags': noise_flags
    }


def generate_evolving_classification_data(shift=0.0, n_samples=300, seed=42):
    """
    Generate evolving classification data for meta-adaptation experiments.
    
    Args:
        shift: Distribution shift parameter
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Tuple of (X, y) tensors
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2) + shift
    y = (X[:, 0] > 0).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def generate_sensitive_data(n_samples=500, input_dim=10, seed=42):
    """
    Generate synthetic sensitive dataset for privacy-preserving experiments.
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimensionality
        seed: Random seed
        
    Returns:
        Tuple of (X, y) tensors
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = (np.sum(X, axis=1) > 0).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)


def preprocess_text_embeddings(texts, embedding_dim=128):
    """
    Simulate text embedding preprocessing.
    
    Args:
        texts: List of text strings
        embedding_dim: Embedding dimension
        
    Returns:
        Tensor of text embeddings
    """
    def encode_text(text):
        base = len(text) / 100.0
        embedding = torch.ones(embedding_dim) * base + torch.randn(embedding_dim) * 0.05
        return embedding

    return torch.stack([encode_text(t) for t in texts])


def preprocess_statistical_features(stat_summaries):
    """
    Preprocess statistical summary features.
    
    Args:
        stat_summaries: List of statistical summary dictionaries
        
    Returns:
        Standardized tensor of statistical features
    """
    stat_vectors = []
    for summary in stat_summaries:
        vec = torch.tensor([summary['mean'], summary['std'], summary['min'], summary['max']], 
                          dtype=torch.float32)
        stat_vectors.append(vec)
    stat_vectors = torch.stack(stat_vectors)

    scaler = StandardScaler()
    stat_np = stat_vectors.numpy()
    stat_np = scaler.fit_transform(stat_np)
    return torch.tensor(stat_np, dtype=torch.float32), scaler


def detect_anomalies(features, contamination=0.1, seed=42):
    """
    Detect anomalies in feature data using Isolation Forest.
    
    Args:
        features: Feature tensor
        contamination: Expected proportion of anomalies
        seed: Random seed
        
    Returns:
        Array of anomaly predictions (-1 for anomaly, 1 for normal)
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=seed)
    return iso_forest.fit_predict(features.numpy())
