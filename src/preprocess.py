import numpy as np
import torch
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

def generate_streaming_data():
    """Generate synthetic binary classification dataset with label noise for streaming experiment."""
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, 
                               n_redundant=5, flip_y=0.1, random_state=42)
    
    stream_windows = np.array_split(np.arange(len(y)), 20)
    return X, y, stream_windows

def prepare_multimodal_data():
    """Prepare data for multi-modal experiment."""
    text_samples = ["I loved this movie!", "I hated this film..."]
    text_labels = [1, 0]  # 1: positive, 0: negative
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    image_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    structured_dataset = TensorDataset(torch.FloatTensor(X_iris_scaled), torch.LongTensor(y_iris))
    structured_loader = DataLoader(structured_dataset, batch_size=16, shuffle=True)
    
    return text_samples, text_labels, image_loader, structured_loader

def simulate_concept_drift(X_batch, y_batch, drift_index, current_index):
    """Simulate concept drift by transforming data at specified index."""
    if current_index >= drift_index:
        rotation_matrix = np.linalg.qr(np.random.randn(20, 20))[0]
        X_batch = np.dot(X_batch, rotation_matrix)
        y_batch = 1 - y_batch
    return X_batch, y_batch
