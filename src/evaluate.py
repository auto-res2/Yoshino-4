import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, Any, Tuple
from datetime import datetime

# Import SimpleMLP from train.py assuming it's in the same directory or accessible
from train import SimpleMLP

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, images_dir: str) -> Dict[str, Any]:
    """Evaluates the trained model and saves plots."""
    print("Starting model evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    # Use a DataLoader for batch processing even for evaluation
    eval_dataset = TensorDataset(X_tensor, y_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            # Determine if model output needs sigmoid (if trained with BCEWithLogitsLoss)
            # Check if original sigmoid was replaced with Identity (for BCEWithLogitsLoss)
            if isinstance(model.sigmoid, nn.Identity): # If sigmoid was bypassed during training
                predicted_probs = torch.sigmoid(outputs)
            else:
                predicted_probs = outputs # Already has sigmoid applied by model

            predicted = (predicted_probs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_pred = np.array(all_preds).flatten()
    y_true = np.array(all_labels).flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    os.makedirs(images_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_filename = os.path.join(images_dir, f'confusion_matrix_{timestamp}.pdf')
    plt.savefig(cm_filename, format='pdf')
    plt.close()
    print(f"Confusion Matrix saved to {cm_filename}")
    
    # Save Classification Report as text (or visualize as table)
    report_filename = os.path.join(images_dir, f'classification_report_{timestamp}.txt')
    with open(report_filename, 'w') as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))
    print(f"Classification Report saved to {report_filename}")

    print("Model evaluation complete.")
    return metrics

if __name__ == '__main__':
    # Simple test run
    # Dummy model and data
    input_size = 5
    dummy_model = SimpleMLP(input_size, 32) # Using imported SimpleMLP
    dummy_X = np.random.rand(100, input_size)
    dummy_y = np.random.randint(0, 2, 100)
    
    # Create dummy images directory
    test_images_dir = "test_images_eval"
    os.makedirs(test_images_dir, exist_ok=True)

    print("--- Evaluation Test Run ---")
    metrics = evaluate_model(dummy_model, dummy_X, dummy_y, test_images_dir)
    print("Metrics from test run:", metrics)
    print("--- Evaluation Test Complete ---")