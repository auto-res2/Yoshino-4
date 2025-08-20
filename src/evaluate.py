import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List, Tuple
import os

def evaluate_pytorch_model(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a PyTorch model and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1_score': f1_score(y_test, predictions, zero_division=0)
    }
    
    return metrics

def evaluate_ensemble_models(models: List[Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate ensemble of sklearn models."""
    predictions_list = []
    
    for model in models:
        pred = model.predict(X_test)
        predictions_list.append(pred)
    
    ensemble_pred = np.round(np.mean(predictions_list, axis=0)).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred, zero_division=0),
        'recall': recall_score(y_test, ensemble_pred, zero_division=0),
        'f1_score': f1_score(y_test, ensemble_pred, zero_division=0)
    }
    
    return metrics

def detect_anomalies(detector: IsolationForest, ensemble_features: np.ndarray) -> np.ndarray:
    """Detect anomalies using trained isolation forest."""
    anomaly_scores = detector.predict(ensemble_features)
    return anomaly_scores  # -1 for anomalies, 1 for normal

def apply_label_guard(predictions: np.ndarray, anomaly_scores: np.ndarray, 
                     original_labels: np.ndarray) -> np.ndarray:
    """Apply label guard by reverting anomalous predictions to original labels."""
    guarded_predictions = predictions.copy()
    anomaly_indices = np.where(anomaly_scores == -1)[0]
    guarded_predictions[anomaly_indices] = original_labels[anomaly_indices]
    return guarded_predictions

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         filename: str, title: str = "Confusion Matrix"):
    """Save confusion matrix as PDF."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    classes = ['Class 0', 'Class 1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def save_metrics_comparison(metrics_before: Dict[str, float], metrics_after: Dict[str, float], 
                           filename: str, title: str = "Metrics Comparison"):
    """Save metrics comparison plot as PDF."""
    metrics_names = list(metrics_before.keys())
    before_values = list(metrics_before.values())
    after_values = list(metrics_after.values())
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, before_values, width, label='Before', alpha=0.8)
    plt.bar(x + width/2, after_values, width, label='After', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {filename}")

if __name__ == '__main__':
    print("--- Evaluation Test Run ---")
    
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    
    y_pred_dummy = np.random.randint(0, 2, 100)
    
    metrics = {
        'accuracy': accuracy_score(y_dummy, y_pred_dummy),
        'precision': precision_score(y_dummy, y_pred_dummy, zero_division=0),
        'recall': recall_score(y_dummy, y_pred_dummy, zero_division=0),
        'f1_score': f1_score(y_dummy, y_pred_dummy, zero_division=0)
    }
    
    print("Test metrics:", metrics)
    
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    save_confusion_matrix(y_dummy, y_pred_dummy, f"{test_dir}/test_cm.pdf")
    
    print("--- Evaluation Test Complete ---")
