import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import generate_synthetic_regression_data

def train_linear_regression(static=True):
    """
    Train a simple linear regression model using PyTorch.
    If static is True, use a "static" (pre-curated) knowledge base hyperparameter.
    If False, simulate improved hyperparameters based on an updated expert knowledge base.
    Returns the loss history and final loss.
    """
    X, y = generate_synthetic_regression_data(n=100)
    
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    
    if static:
        lr = 0.1
    else:
        lr = 0.05
        
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    n_epochs = 50
    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"{'Static' if static else 'Dynamic'} KB - Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    
    return loss_history, loss.item()
