import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class SimpleClassifier(nn.Module):
    """Simple neural network for binary classification."""
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        return self.fc(x)

class KnowledgeCard:
    """Knowledge card for Bayesian bandit experiment."""
    def __init__(self, name, base_accuracy):
        self.name = name
        self.base_accuracy = base_accuracy

    def evaluate(self, drift_factor):
        """Simulate performance drop/improvement given drift factor."""
        performance = self.base_accuracy * (1 - drift_factor) + np.random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, performance))

def update_model(model, optimizer, X_batch, y_batch, criterion):
    """Update model with a batch of data."""
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_batch))
    loss = criterion(outputs, torch.LongTensor(y_batch))
    loss.backward()
    optimizer.step()
    return loss.item()

def train_streaming_models(X, y, stream_windows):
    """Train baseline and meta-adaptive models on streaming data."""
    baseline_model = SimpleClassifier(input_dim=20)
    meta_model = SimpleClassifier(input_dim=20)
    
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=0.001)
    optimizer_meta = optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    baseline_accuracies = []
    meta_accuracies = []
    
    for i, indices in enumerate(stream_windows):
        X_batch = X[indices]
        y_batch = y[indices].copy()
        
        if i == 10:
            rotation_matrix = np.linalg.qr(np.random.randn(20, 20))[0]
            X_batch = np.dot(X_batch, rotation_matrix)
            y_batch = 1 - y_batch
        
        loss_baseline = update_model(baseline_model, optimizer_baseline, X_batch, y_batch, criterion)
        loss_meta = update_model(meta_model, optimizer_meta, X_batch, y_batch, criterion)
        
        baseline_model.eval()
        meta_model.eval()
        with torch.no_grad():
            outputs_base = baseline_model(torch.FloatTensor(X_batch))
            preds_base = torch.argmax(outputs_base, dim=1).numpy()
            accuracy_baseline = np.mean(preds_base == y_batch)
            
            outputs_meta = meta_model(torch.FloatTensor(X_batch))
            preds_meta = torch.argmax(outputs_meta, dim=1).numpy()
            accuracy_meta = np.mean(preds_meta == y_batch)
        
        baseline_accuracies.append(accuracy_baseline)
        meta_accuracies.append(accuracy_meta)
        
        if i >= 10 and accuracy_meta < 0.7:
            for param_group in optimizer_meta.param_groups:
                param_group['lr'] *= 1.05
        
        print(f'Window {i:2d}: Baseline Acc = {accuracy_baseline:.3f}, MetaAcc = {accuracy_meta:.3f}')
    
    return baseline_accuracies, meta_accuracies

def thompson_sampling(bandit_params):
    """Thompson sampling for Bayesian bandit selection."""
    sampled_values = {}
    for card_name, (a, b) in bandit_params.items():
        sampled_values[card_name] = np.random.beta(a, b)
    return max(sampled_values.keys(), key=lambda k: sampled_values[k])

def train_knowledge_cards():
    """Train knowledge cards using Bayesian bandit approach."""
    cards = [KnowledgeCard('Time-Series', 0.85),
             KnowledgeCard('Text', 0.80),
             KnowledgeCard('Anomaly', 0.90)]
    
    bandit_params = {card.name: [1, 1] for card in cards}
    
    num_iterations = 50
    selection_history = {card.name: [] for card in cards}
    performance_history = []
    drift_factor = 0.0
    
    for it in range(num_iterations):
        drift_factor = min(0.5, it / num_iterations)
        
        selected_card_name = thompson_sampling(bandit_params)
        selected_card = next(card for card in cards if card.name == selected_card_name)
        performance = selected_card.evaluate(drift_factor)
        performance_history.append(performance)
        
        reward = 1 if performance > 0.75 else 0
        if reward == 1:
            bandit_params[selected_card_name][0] += 1
        else:
            bandit_params[selected_card_name][1] += 1
        
        for name in selection_history.keys():
            selection_history[name].append(1 if name == selected_card_name else 0)
        
        print(f'Iteration {it:2d}: Selected {selected_card_name} with performance {performance:.3f} (Reward: {reward})')
    
    return selection_history, performance_history
