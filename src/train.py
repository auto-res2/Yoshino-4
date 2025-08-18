import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.0, apply_regularization: bool = False):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1) # Binary classification output
        self.sigmoid = nn.Sigmoid()
        self.apply_regularization = apply_regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_model(X: np.ndarray, y: np.ndarray, model_config: Dict[str, Any]) -> nn.Module:
    """Trains the SimpleMLP model."""
    print("Starting model training...")

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Add a dimension for binary output

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor.numpy())

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False)

    input_size = X_train.shape[1]
    model = SimpleMLP(input_size, model_config['hidden_size'], model_config['dropout_rate'], model_config['apply_regularization'])
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Loss function and optimizer
    # Apply class weights if specified by SCEL for imbalanced data
    criterion = nn.BCELoss()
    if model_config.get('apply_class_weights', False):
        print("Training: Applying class weights to BCELoss.")
        # Calculate class weights based on actual training data imbalance
        neg_count = (y_train == 0).sum().item()
        pos_count = (y_train == 1).sum().item()
        if pos_count == 0 or neg_count == 0:
            print("Warning: One class has 0 samples, cannot apply class weights effectively.")
        else:
            total_samples = neg_count + pos_count
            weight_for_neg = total_samples / (2.0 * neg_count)
            weight_for_pos = total_samples / (2.0 * pos_count)
            pos_weight_tensor = torch.tensor([weight_for_pos], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            # Adjust model output for BCEWithLogitsLoss (remove sigmoid from model if it was there)
            model.sigmoid = nn.Identity() # No sigmoid if using BCEWithLogitsLoss

    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    # Training loop
    for epoch in range(model_config['num_epochs']):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L2 regularization if applicable (simple weight decay)
            if model_config['apply_regularization']:
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += model_config.get('regularization_lambda', 0.001) * l2_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{model_config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Model training complete.")
    return model

if __name__ == '__main__':
    # Simple test run
    dummy_X = np.random.rand(100, 5)
    dummy_y = np.random.randint(0, 2, 100)
    
    dummy_model_config = {
        'learning_rate': 0.01,
        'num_epochs': 5,
        'batch_size': 16,
        'hidden_size': 32,
        'dropout_rate': 0.1,
        'apply_regularization': False,
        'apply_class_weights': False
    }

    print("--- Training Test Run ---")
    trained_model = train_model(dummy_X, dummy_y, dummy_model_config)
    print("--- Training Test Complete ---")