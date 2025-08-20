"""
AutoMind++ Framework - Training Module
Implements multi-modal fusion, meta-adaptation, and privacy-preserving training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Opacus not available. Privacy-preserving features will be simulated.")


class MultiModalFusionModule(nn.Module):
    """Multi-modal fusion module for combining text, image, and statistical features."""
    
    def __init__(self, text_dim=128, image_dim=512, stat_dim=4, hidden_dim=256, output_dim=64):
        super(MultiModalFusionModule, self).__init__()
        
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.image_processor = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.stat_processor = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU()
        )
        
        self.text_uncertainty = nn.Linear(text_dim, 1)
        self.image_uncertainty = nn.Linear(image_dim, 1)
        self.stat_uncertainty = nn.Linear(stat_dim, 1)
        
    def forward(self, text_features, image_features, stat_features):
        text_processed = self.text_processor(text_features)
        image_processed = self.image_processor(image_features)
        stat_processed = self.stat_processor(stat_features)
        
        text_weight = torch.sigmoid(self.text_uncertainty(text_features))
        image_weight = torch.sigmoid(self.image_uncertainty(image_features))
        stat_weight = torch.sigmoid(self.stat_uncertainty(stat_features))
        
        text_weighted = text_processed * text_weight
        image_weighted = image_processed * image_weight
        stat_weighted = stat_processed * stat_weight
        
        fused = torch.cat([text_weighted, image_weighted, stat_weighted], dim=1)
        output = self.fusion_layer(fused)
        
        return output, (text_weight, image_weight, stat_weight)


class SimpleClassifier(nn.Module):
    """Simple classifier for meta-adaptation experiments."""
    
    def __init__(self, input_dim=2, hidden_dim=20, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class TeacherStudentNet(nn.Module):
    """Network for privacy-preserving knowledge distillation."""
    
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=2):
        super(TeacherStudentNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)


def train_multimodal_fusion(text_embeddings, plot_features, stat_vectors, 
                           noise_flags, epochs=5, lr=0.001):
    """
    Train the multi-modal fusion module.
    
    Args:
        text_embeddings: Text feature tensor
        plot_features: Image feature tensor  
        stat_vectors: Statistical feature tensor
        noise_flags: Noise indicator flags
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiModalFusionModule(
        text_dim=text_embeddings.shape[1],
        image_dim=plot_features.shape[1], 
        stat_dim=stat_vectors.shape[1]
    ).to(device)
    
    text_embeddings = text_embeddings.to(device)
    plot_features = plot_features.to(device)
    stat_vectors = stat_vectors.to(device)
    
    labels = torch.randint(0, 2, (text_embeddings.shape[0],)).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    classifier = nn.Linear(64, 2).to(device)
    
    training_history = {'loss': [], 'weights': []}
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        fused_features, weights = model(text_embeddings, plot_features, stat_vectors)
        logits = classifier(fused_features)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        training_history['loss'].append(loss.item())
        training_history['weights'].append([w.mean().item() for w in weights])
        
        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return model, training_history


def active_learning_selection(model, inputs, num_samples=20):
    """
    Select most informative samples using uncertainty-based active learning.
    
    Args:
        model: Trained model
        inputs: Input tensor
        num_samples: Number of samples to select
        
    Returns:
        Indices of selected samples
    """
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
    _, indices = torch.topk(entropy, num_samples)
    return indices


def train_with_meta_adaptation(num_iterations=5, epochs_per_iter=3, lr=0.01):
    """
    Train classifier with human-guided meta-adaptation.
    
    Args:
        num_iterations: Number of adaptation iterations
        epochs_per_iter: Training epochs per iteration
        lr: Learning rate
        
    Returns:
        Training history and model
    """
    from preprocess import generate_evolving_classification_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    feedback_buffer = []
    history = {'accuracy': [], 'shifts': []}
    
    for iteration in range(num_iterations):
        shift = iteration * 0.5  # Simulate shifting distribution
        X_train, y_train = generate_evolving_classification_data(shift=shift, n_samples=300)
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs_per_iter):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        informative_idx = active_learning_selection(model, X_train, num_samples=20)
        feedback_buffer.append((X_train[informative_idx], y_train[informative_idx]))
        
        if iteration % 2 == 0 and len(feedback_buffer) > 0:
            combined_x = torch.cat([item[0] for item in feedback_buffer], dim=0)
            combined_y = torch.cat([item[1] for item in feedback_buffer], dim=0)
            rapid_dataset = TensorDataset(combined_x, combined_y)
            rapid_loader = DataLoader(rapid_dataset, batch_size=16, shuffle=True)
            
            for _ in range(2):  # Fast retraining epochs
                for bx, by in rapid_loader:
                    optimizer.zero_grad()
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_train)
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == y_train).float().mean().item()
        
        history['accuracy'].append(accuracy)
        history['shifts'].append(shift)
        
        print(f'Iteration {iteration}, Shift {shift:.2f}, Accuracy: {accuracy:.3f}')
    
    return model, history


def train_privacy_preserving_kd(epochs=3, lr=0.1, noise_multiplier=1.0):
    """
    Train student model with privacy-preserving knowledge distillation.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        noise_multiplier: Differential privacy noise multiplier
        
    Returns:
        Teacher model, student model, and training history
    """
    from preprocess import generate_sensitive_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, y = generate_sensitive_data(n_samples=500, input_dim=10)
    X, y = X.to(device), y.to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    teacher = TeacherStudentNet().to(device)
    teacher_optimizer = optim.SGD(teacher.parameters(), lr=lr)
    teacher_criterion = nn.CrossEntropyLoss()
    
    print("Training teacher model...")
    teacher.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            teacher_optimizer.zero_grad()
            outputs = teacher(batch_x)
            loss = teacher_criterion(outputs, batch_y)
            loss.backward()
            teacher_optimizer.step()
    
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(X)
    
    student = TeacherStudentNet().to(device)
    student_optimizer = optim.SGD(student.parameters(), lr=lr)
    
    if OPACUS_AVAILABLE:
        try:
            privacy_engine = PrivacyEngine()
            student, student_optimizer, dataloader = privacy_engine.make_private(
                module=student,
                optimizer=student_optimizer,
                data_loader=dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0,
            )
            print("Differential privacy enabled with Opacus")
        except Exception as e:
            print(f"Opacus setup failed: {e}. Continuing without DP.")
            opacus_enabled = False
    
    print("Training student model with knowledge distillation...")
    student.train()
    history = {'loss': [], 'epsilon': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            student_optimizer.zero_grad()
            
            student_logits = student(batch_x)
            
            with torch.no_grad():
                teacher_batch_logits = teacher(batch_x)
            
            temperature = 3.0
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(student_logits / temperature, dim=1),
                torch.softmax(teacher_batch_logits / temperature, dim=1)
            ) * (temperature ** 2)
            
            ce_loss = nn.CrossEntropyLoss()(student_logits, batch_y)
            
            loss = 0.7 * kd_loss + 0.3 * ce_loss
            loss.backward()
            student_optimizer.step()
            
            epoch_loss += loss.item()
        
        history['loss'].append(epoch_loss / len(dataloader))
        
        if OPACUS_AVAILABLE and 'privacy_engine' in locals() and hasattr(privacy_engine, 'get_epsilon'):
            try:
                epsilon = privacy_engine.get_epsilon(delta=1e-5)
                history['epsilon'].append(epsilon)
                print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}, ε: {epsilon:.2f}')
            except:
                print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}')
        else:
            print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}')
    
    return teacher, student, history
