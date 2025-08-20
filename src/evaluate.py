import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, List, Tuple
import os

def evaluate_adaptive_agent(agent: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate an adaptive agent and return performance metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    agent.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = agent(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0, average='weighted'),
        'recall': recall_score(y_test, predictions, zero_division=0, average='weighted'),
        'f1_score': f1_score(y_test, predictions, zero_division=0, average='weighted')
    }
    
    return metrics

def evaluate_agent_swarm(agents: List[torch.nn.Module], meta_controller, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the entire agent swarm with meta-controller aggregation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_predictions = []
    all_uncertainties = []
    
    for agent in agents:
        agent.to(device)
        agent.eval()
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            pred_prob, uncertainty = agent.predict_with_uncertainty(X_tensor, mc_samples=5)
            all_predictions.append(pred_prob)
            all_uncertainties.append(uncertainty)
    
    aggregated_preds = []
    for i in range(len(X_test)):
        sample_preds = [pred[i] for pred in all_predictions]
        sample_uncertainties = [unc for unc in all_uncertainties]  # Use same uncertainty for all samples
        agg_pred = meta_controller.aggregate(sample_preds, sample_uncertainties, dynamic_weighting=True)
        pred_class = torch.argmax(agg_pred).item()
        aggregated_preds.append(pred_class)
    
    aggregated_preds = np.array(aggregated_preds)
    
    metrics = {
        'accuracy': accuracy_score(y_test, aggregated_preds),
        'precision': precision_score(y_test, aggregated_preds, zero_division=0, average='weighted'),
        'recall': recall_score(y_test, aggregated_preds, zero_division=0, average='weighted'),
        'f1_score': f1_score(y_test, aggregated_preds, zero_division=0, average='weighted')
    }
    
    return metrics

def evaluate_streaming_performance(accuracy_log: List[int], window_size: int = 50) -> Dict[str, float]:
    """Evaluate streaming performance with sliding window metrics."""
    accuracy_array = np.array(accuracy_log)
    
    metrics = {
        'overall_accuracy': np.mean(accuracy_array),
        'final_window_accuracy': np.mean(accuracy_array[-window_size:]) if len(accuracy_array) >= window_size else np.mean(accuracy_array),
        'accuracy_std': np.std(accuracy_array),
        'accuracy_trend': np.polyfit(range(len(accuracy_array)), accuracy_array, 1)[0]  # Linear trend slope
    }
    
    return metrics

def calculate_uncertainty_metrics(uncertainty_log: List[float]) -> Dict[str, float]:
    """Calculate uncertainty-related metrics."""
    uncertainty_array = np.array(uncertainty_log)
    
    metrics = {
        'mean_uncertainty': np.mean(uncertainty_array),
        'uncertainty_std': np.std(uncertainty_array),
        'max_uncertainty': np.max(uncertainty_array),
        'min_uncertainty': np.min(uncertainty_array)
    }
    
    return metrics

def save_performance_comparison(baseline_acc: List[int], adaptive_acc: List[int], 
                               filename: str, title: str = "Performance Comparison"):
    """Save performance comparison plot as PDF."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline_acc, label='Baseline', alpha=0.7, linewidth=2)
    plt.plot(adaptive_acc, label='Adaptive', alpha=0.7, linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    window = 20
    baseline_smooth = np.convolve(baseline_acc, np.ones(window)/window, mode='valid')
    adaptive_smooth = np.convolve(adaptive_acc, np.ones(window)/window, mode='valid')
    
    plt.plot(range(window-1, len(baseline_acc)), baseline_smooth, label='Baseline (smoothed)', linewidth=2)
    plt.plot(range(window-1, len(adaptive_acc)), adaptive_smooth, label='Adaptive (smoothed)', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Smoothed Accuracy')
    plt.title('Smoothed Accuracy Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Performance comparison saved to {filename}")

def save_uncertainty_analysis(uncertainty_log: List[float], accuracy_log: List[int],
                             filename: str, title: str = "Uncertainty Analysis"):
    """Save uncertainty analysis plot as PDF."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(uncertainty_log, color='red', alpha=0.7, linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(uncertainty_log, accuracy_log, alpha=0.6, s=20)
    plt.xlabel('Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Uncertainty vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(uncertainty_log, accuracy_log)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Uncertainty analysis saved to {filename}")

def evaluate_prediction_single(logits: torch.Tensor, label: int) -> int:
    """Evaluate single prediction accuracy."""
    pred = torch.argmax(logits).item()
    return int(pred == label)

if __name__ == '__main__':
    print("--- Adaptive Evaluation Test ---")
    
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    
    dummy_accuracy_log = [np.random.randint(0, 2) for _ in range(100)]
    streaming_metrics = evaluate_streaming_performance(dummy_accuracy_log)
    print("Streaming metrics:", streaming_metrics)
    
    dummy_uncertainty_log = [np.random.rand() * 0.5 for _ in range(100)]
    uncertainty_metrics = calculate_uncertainty_metrics(dummy_uncertainty_log)
    print("Uncertainty metrics:", uncertainty_metrics)
    
    print("--- Evaluation Module Test Complete ---")
