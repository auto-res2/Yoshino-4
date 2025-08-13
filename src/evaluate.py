import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model_output: dict, true_labels: pd.Series, params: dict, task_context: dict) -> dict:
    logging.info(f"Evaluating model for task '{task_context.get('task_id', 'N/A')}' with params: {params}")

    predictions = model_output.get("predictions_test")
    true_labels_list = true_labels.tolist() 

    if not predictions or not true_labels_list:
        raise ValueError("EvaluationError: Missing predictions or true labels for evaluation.")
    
    if len(predictions) != len(true_labels_list):
        raise ValueError("EvaluationError: Length of predictions and true labels do not match.")

    metrics = {}
    try:
        # Assuming binary classification for simplicity for now
        metrics['accuracy'] = accuracy_score(true_labels_list, predictions)
        metrics['precision'] = precision_score(true_labels_list, predictions, zero_division=0)
        metrics['recall'] = recall_score(true_labels_list, predictions, zero_division=0)
        metrics['f1_score'] = f1_score(true_labels_list, predictions, zero_division=0)
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  F1-Score: {metrics['f1_score']:.4f}")

    except Exception as e:
        logging.error(f"EvaluationError: Error calculating metrics: {e}")
        raise ValueError(f"EvaluationError: {e}")
    
    # Determine if performance is suboptimal
    suboptimal_threshold = params.get('suboptimal_threshold', 0.75) # F1-score threshold
    metrics['is_suboptimal'] = metrics['f1_score'] < suboptimal_threshold
    if metrics['is_suboptimal']:
        logging.warning(f"  Performance is suboptimal (F1-score: {metrics['f1_score']:.4f} < {suboptimal_threshold:.2f}).")
    else:
        logging.info(f"  Performance is satisfactory (F1-score: {metrics['f1_score']:.4f} >= {suboptimal_threshold:.2f}).")
    
    # Save a mock plot
    output_dir = os.path.join(".research", "iteration1", "images")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"evaluation_metrics_task_{task_context.get('task_id', 'N_A')}.pdf")
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(metrics.keys())[:4], y=[metrics[k] for k in list(metrics.keys())[:4]])
    plt.title(f"Evaluation Metrics for Task {task_context.get('task_id', 'N/A')}")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plot_path, format='pdf')
    plt.close()
    logging.info(f"  Evaluation plot saved to {plot_path}")

    return {"metrics": metrics, "plot_path": plot_path}

if __name__ == '__main__':
    print("--- Evaluate Test Run ---")

    # Generate mock predictions and true labels
    def generate_mock_results(score=0.8, suboptimal=False):
        np.random.seed(42)
        true_labels = (np.random.rand(100) > 0.5).astype(int).tolist()
        if suboptimal:
            # Create predictions that lead to low F1
            predictions = [(label if np.random.rand() < 0.6 else 1 - label) for label in true_labels] # ~60% accuracy
        else:
            predictions = [(label if np.random.rand() < score else 1 - label) for label in true_labels]
        return {"predictions_test": predictions}, pd.Series(true_labels)

    # Scenario 1: Good performance
    print("\nScenario 1: Good performance")
    model_output_good, true_labels_good = generate_mock_results(score=0.9)
    params_good = {"suboptimal_threshold": 0.75}
    task_context_good = {"task_id": "test_eval_good"}
    try:
        eval_results_good = evaluate_model(model_output_good, true_labels_good, params_good, task_context_good)
        print(f"Good performance metrics: {eval_results_good['metrics']}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # Scenario 2: Suboptimal performance
    print("\nScenario 2: Suboptimal performance")
    model_output_suboptimal, true_labels_suboptimal = generate_mock_results(suboptimal=True)
    params_suboptimal = {"suboptimal_threshold": 0.75}
    task_context_suboptimal = {"task_id": "test_eval_suboptimal"}
    try:
        eval_results_suboptimal = evaluate_model(model_output_suboptimal, true_labels_suboptimal, params_suboptimal, task_context_suboptimal)
        print(f"Suboptimal performance metrics: {eval_results_suboptimal['metrics']}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # Scenario 3: Mismatched lengths
    print("\nScenario 3: Mismatched lengths")
    model_output_mismatch = {"predictions_test": [0, 1, 0]}
    true_labels_mismatch = pd.Series([0, 1, 0, 1])
    params_mismatch = {}
    task_context_mismatch = {"task_id": "test_eval_mismatch"}
    try:
        evaluate_model(model_output_mismatch, true_labels_mismatch, params_mismatch, task_context_mismatch)
    except ValueError as e:
        print(f"Caught expected error: {e}")
