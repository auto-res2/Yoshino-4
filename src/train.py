import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from typing import List, Tuple, Any

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for binary classification."""
    def __init__(self, input_size: int, hidden_size: int = 64, dropout_rate: float = 0.2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_simple_mlp(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, lr: float = 0.001) -> SimpleMLP:
    """Train a simple MLP model for binary classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on device: {device}")
    
    model = SimpleMLP(X_train.shape[1])
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def train_ensemble_models(X_train: np.ndarray, y_train: np.ndarray) -> List[Any]:
    """Train ensemble of models for EMAF-RLG experiment."""
    print("Training ensemble models...")
    
    models = [
        LogisticRegression(max_iter=500, random_state=42),
        DecisionTreeClassifier(random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    for model in models:
        model.fit(X_train, y_train)
    
    print("Ensemble training complete.")
    return models

def get_ensemble_predictions(models: List[Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get ensemble predictions and confidence scores."""
    predictions = []
    confidences = []
    
    for model in models:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]
            pred = (proba > 0.5).astype(int)
        else:
            pred = model.predict(X)
            proba = model.decision_function(X)
        
        predictions.append(pred)
        confidences.append(proba)
    
    ensemble_confidence = np.mean(confidences, axis=0)
    ensemble_prediction = (ensemble_confidence > 0.5).astype(int)
    
    return ensemble_prediction, ensemble_confidence

def train_anomaly_detector(ensemble_features: np.ndarray, contamination: float = 0.1) -> IsolationForest:
    """Train anomaly detector for label guard."""
    detector = IsolationForest(contamination=contamination, random_state=42)
    detector.fit(ensemble_features)
    return detector

if __name__ == '__main__':
    print("--- Training Test Run ---")
    
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    
    mlp_model = train_simple_mlp(X_dummy, y_dummy, epochs=10)
    print("MLP training test complete.")
    
    ensemble_models = train_ensemble_models(X_dummy, y_dummy)
    print("Ensemble training test complete.")
    
    print("--- Training Test Complete ---")
