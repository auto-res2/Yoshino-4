import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from src.train import SimpleNN # Import the model definition

def evaluate_model(X_test, y_test, config):
    print("Starting model evaluation...")
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    model = SimpleNN(input_dim, num_classes)
    model_path = os.path.join(config['paths']['models_dir'], 'trained_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    print(f"Evaluation Metrics: {metrics}")
    return metrics

if __name__ == '__main__':
    # This block is for simple testing of evaluate.py in isolation
    print("Running evaluate.py in test mode...")
    # Need to have a dummy model trained for this to work
    from src.train import train_model
    X_test_dummy = pd.DataFrame(np.random.rand(50, 10))
    y_test_dummy = pd.Series(np.random.randint(0, 2, 50))

    dummy_config = {
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 1
        },
        'evaluation': {
            'batch_size': 16
        },
        'paths': {
            'models_dir': '../models'
        }
    }
    os.makedirs(dummy_config['paths']['models_dir'], exist_ok=True)
    # Train a dummy model first
    _ = train_model(pd.DataFrame(np.random.rand(100, 10)), pd.Series(np.random.randint(0, 2, 100)), dummy_config)
    
    metrics = evaluate_model(X_test_dummy, y_test_dummy, dummy_config)
    print("evaluate.py test finished.")
