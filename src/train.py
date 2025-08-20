import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any

class AdaptiveAgent(nn.Module):
    """Lightweight classifier with Monte Carlo dropout and LoRA-style adaptation."""
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=2, dropout_prob=0.3):
        super(AdaptiveAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lora_A = nn.Parameter(torch.randn(input_dim, 4) * 0.01)  
        self.lora_B = nn.Parameter(torch.randn(4, input_dim) * 0.01)

    def forward(self, x):
        adaptation = torch.matmul(x, torch.matmul(self.lora_A, self.lora_B))
        x_mod = x + adaptation
        h = torch.relu(self.fc1(x_mod))
        h_drop = self.dropout(h)
        out = self.fc2(h_drop)
        return out

    def predict_with_uncertainty(self, x, mc_samples=5):
        """Predict with uncertainty estimation using Monte Carlo dropout."""
        self.train()  # keep dropout active for MC sampling
        preds = []
        for _ in range(mc_samples):
            logits = self.forward(x)
            preds.append(torch.softmax(logits, dim=-1).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        mean_pred = torch.mean(preds, dim=0)
        uncertainty = torch.var(preds, dim=0).mean().item()  
        return mean_pred, uncertainty

class BaselineNN(nn.Module):
    """Simple baseline neural network for comparison."""
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=2):
        super(BaselineNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_adaptive_agent(agent: AdaptiveAgent, X_train: np.ndarray, y_train: np.ndarray, 
                        epochs: int = 50, lr: float = 0.01) -> AdaptiveAgent:
    """Train an adaptive agent on streaming data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    
    optimizer = optim.SGD(agent.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    agent.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = agent(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Agent Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return agent

def train_baseline_model(model: BaselineNN, X_train: np.ndarray, y_train: np.ndarray, 
                        epochs: int = 50, lr: float = 0.01) -> BaselineNN:
    """Train baseline model for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Baseline Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def create_agent_swarm(num_agents: int = 3, input_dim: int = 10) -> List[AdaptiveAgent]:
    """Create a swarm of adaptive agents with different initializations."""
    agents = []
    for i in range(num_agents):
        torch.manual_seed(42 + i)
        agent = AdaptiveAgent(input_dim=input_dim)
        agents.append(agent)
    return agents

def online_training_step(agent: AdaptiveAgent, x: torch.Tensor, y: torch.Tensor, 
                        optimizer: torch.optim.Optimizer) -> float:
    """Perform one online training step for streaming data."""
    agent.train()
    optimizer.zero_grad()
    logits = agent(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == '__main__':
    print("--- Adaptive Agent Training Test ---")
    
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    
    agent = AdaptiveAgent(input_dim=10)
    trained_agent = train_adaptive_agent(agent, X_dummy, y_dummy, epochs=10)
    print("Adaptive agent training test complete.")
    
    swarm = create_agent_swarm(num_agents=3, input_dim=10)
    print(f"Created agent swarm with {len(swarm)} agents.")
    
    baseline = BaselineNN(input_dim=10)
    trained_baseline = train_baseline_model(baseline, X_dummy, y_dummy, epochs=10)
    print("Baseline model training test complete.")
    
    print("--- Training Module Test Complete ---")
