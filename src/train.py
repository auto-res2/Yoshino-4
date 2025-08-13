import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data: pd.DataFrame, params: dict, task_context: dict):
    logging.info(f"Training model for task '{task_context.get('task_id', 'N/A')}' with params: {params}")

    if 'target' not in data.columns:
        raise ValueError("TrainingError: 'target' column not found in data.")

    X = data.drop('target', axis=1)
    y = data['target']

    # Ensure target is binary for Logistic Regression mock
    if y.nunique() > 2:
        logging.warning("Target has more than 2 unique values. Binarizing for mock Logistic Regression.")
        y = (y > y.median()).astype(int)
    elif y.nunique() == 1:
        raise ValueError("TrainingError: Target variable has only one unique value, cannot train a meaningful model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    epochs = params.get('epochs', 10)
    learning_rate = params.get('learning_rate', 0.001)
    add_regularization = params.get('add_regularization', False)

    # Simulate model training (Logistic Regression as a placeholder)
    # The 'max_iter' and 'C' (inverse of regularization strength) will simulate epochs and regularization
    # Large C means less regularization.
    C_val = 1.0 / learning_rate # Inversely related to LR conceptually for simple sim
    if add_regularization:
        C_val = 0.1 # Stronger regularization

    try:
        model = LogisticRegression(max_iter=epochs * 10, solver='liblinear', C=C_val, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        logging.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"  Test Accuracy: {test_accuracy:.4f}")

        # Simulate overfitting based on config or conditions
        if task_context.get('simulate_train_overfitting', False):
            # Artificially inflate train_accuracy and suppress test_accuracy
            train_accuracy = 0.95
            test_accuracy = 0.60
            logging.warning("  Simulating overfitting as per task context.")
        elif train_accuracy > 0.9 and test_accuracy < 0.7: # Heuristic for real overfitting
             logging.warning("  Potential overfitting detected: High train accuracy, low test accuracy.")
             task_context['overfitting_detected'] = True
        
        model_output = {
            "model": model, # In a real scenario, this would be a serialized model or path
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "predictions_test": y_pred_test.tolist(),
            "true_labels_test": y_test.tolist()
        }
        
        logging.info("Model training complete.")
        return model_output

    except Exception as e:
        logging.error(f"TrainingError: An error occurred during model training: {e}")
        raise ValueError(f"TrainingError: {e}")

if __name__ == '__main__':
    # Simple test run
    print("--- Train Test Run ---")
    
    # Generate mock data for training
    def generate_binary_mock_data(size=100):
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(size),
            'feature2': np.random.rand(size),
            'target': (np.random.rand(size) > 0.5).astype(int)
        })
        return data

    # Scenario 1: Default successful training
    print("\nScenario 1: Default successful training")
    data_default = generate_binary_mock_data()
    params_default = {"epochs": 10, "learning_rate": 0.001, "add_regularization": False}
    task_context_default = {"task_id": "test_train_default"}
    try:
        model_output_default = train_model(data_default, params_default, task_context_default)
        print(f"Model output keys: {model_output_default.keys()}")
        print(f"Train accuracy: {model_output_default['train_accuracy']:.4f}")
        print(f"Test accuracy: {model_output_default['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # Scenario 2: Simulate overfitting
    print("\nScenario 2: Simulate overfitting")
    data_overfit = generate_binary_mock_data(size=50) # Smaller dataset prone to overfitting
    params_overfit = {"epochs": 50, "learning_rate": 0.01} # Higher epochs/LR
    task_context_overfit = {"task_id": "test_train_overfit", "simulate_train_overfitting": True}
    try:
        model_output_overfit = train_model(data_overfit, params_overfit, task_context_overfit)
        print(f"Simulated Overfit Train accuracy: {model_output_overfit['train_accuracy']:.4f}")
        print(f"Simulated Overfit Test accuracy: {model_output_overfit['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # Scenario 3: Real overfitting (heuristic detection)
    print("\nScenario 3: Real overfitting (heuristic detection)")
    # Create data that naturally overfits a bit
    data_real_overfit = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20),
        'target': (np.random.rand(20) > 0.5).astype(int)
    })
    params_real_overfit = {"epochs": 100, "learning_rate": 0.0001, "add_regularization": False} # Many epochs, low LR
    task_context_real_overfit = {"task_id": "test_train_real_overfit"}
    try:
        model_output_real_overfit = train_model(data_real_overfit, params_real_overfit, task_context_real_overfit)
        print(f"Real Overfit Train accuracy: {model_output_real_overfit['train_accuracy']:.4f}")
        print(f"Real Overfit Test accuracy: {model_output_real_overfit['test_accuracy']:.4f}")
        print(f"Overfitting detected in context: {task_context_real_overfit.get('overfitting_detected', False)}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # Scenario 4: Target with only one unique value
    print("\nScenario 4: Target with only one unique value")
    data_single_target = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.zeros(100)
    })
    params_single_target = {"epochs": 10, "learning_rate": 0.001}
    task_context_single_target = {"task_id": "test_train_single_target"}
    try:
        train_model(data_single_target, params_single_target, task_context_single_target)
    except ValueError as e:
        print(f"Caught expected error: {e}")
