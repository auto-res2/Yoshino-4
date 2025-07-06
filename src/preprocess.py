import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def generate_true_sample(n_samples=1):
    """Generate samples from 2D Gaussian Mixture: Two modes at (-3,-3) and (3,3) with equal probability."""
    mode = np.random.choice([0, 1], size=n_samples)
    means = np.array([[-3.0, -3.0], [3.0, 3.0]])
    samples = means[mode]
    return torch.tensor(samples, dtype=torch.float32)

def add_noise(data, noise_std):
    """Add Gaussian noise to image data (assuming data is a torch tensor)"""
    noise = torch.randn_like(data) * noise_std
    return data + noise

def get_mnist_dataloader(batch_size=16, shuffle=True):
    """Get MNIST dataloader for Experiment 3"""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def true_likelihood(x, mean, variance):
    """Computes the likelihood for a 1D Gaussian."""
    return (1.0/np.sqrt(2*np.pi*variance)) * np.exp(-((x - mean)**2)/(2*variance))
