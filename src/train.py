import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(X_train, y_train, config):
    print("Starting model training...")
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    num_epochs = config['training']['epochs']
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model_path = os.path.join(config['paths']['models_dir'], 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

if __name__ == '__main__':
    # This block is for simple testing of train.py in isolation
    # In a real scenario, main.py will call this function.
    print("Running train.py in test mode...")
    # Create dummy data
    X_train_dummy = pd.DataFrame(np.random.rand(100, 10))
    y_train_dummy = pd.Series(np.random.randint(0, 2, 100))
    
    dummy_config = {
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 5
        },
        'paths': {
            'models_dir': '../models'
        }
    }
    os.makedirs(dummy_config['paths']['models_dir'], exist_ok=True)
    train_model(X_train_dummy, y_train_dummy, dummy_config)
    print("train.py test finished.")
