import torch
import torch.nn as nn
import torch.optim as optim
try:
    from .preprocess import generate_true_sample
except ImportError:
    from preprocess import generate_true_sample

class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ScoreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ScoreNet1D(nn.Module):
    def __init__(self):
        super(ScoreNet1D, self).__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)

def train_score_net_2d(device='cpu'):
    """Train 2D score network for Experiment 1"""
    input_dim, hidden_dim, output_dim = 2, 32, 2
    score_net = ScoreNet(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
    
    for epoch in range(200):
        x_true = generate_true_sample(n_samples=32).to(device)
        optimizer.zero_grad()
        output = score_net(x_true)
        loss = torch.mean(output**2)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Training Epoch {epoch}: Loss = {loss.item():.5f}")
    print("Training for Experiment 1 completed.")
    return score_net

def train_score_net_1d(device='cpu'):
    """Train 1D score network for Experiment 2"""
    net_1d = ScoreNet1D().to(device)
    optimizer = optim.Adam(net_1d.parameters(), lr=1e-3)
    
    for epoch in range(300):
        x_train = torch.randn((32, 1), device=device)
        target = -x_train
        optimizer.zero_grad()
        score_pred = net_1d(x_train)
        loss = torch.mean((score_pred - target)**2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"1D Network Training Epoch {epoch}: Loss = {loss.item():.5f}")
    print("Training for Experiment 2 completed.")
    return net_1d

def train_score_net_img(data_loader, device='cpu'):
    """Train image score network for Experiment 3"""
    score_net_img = ScoreNet(28*28, 128, 28*28).to(device)
    optimizer = optim.Adam(score_net_img.parameters(), lr=1e-3)
    
    for epoch in range(5):
        for images, _ in data_loader:
            images = images.to(device)
            inputs = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = score_net_img(inputs)
            loss = torch.mean((outputs - inputs)**2)
            loss.backward()
            optimizer.step()
        print(f"Image Network Training Epoch {epoch} completed.")
    return score_net_img
