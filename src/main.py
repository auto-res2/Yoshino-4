import os
import json
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.preprocess import get_preprocessed_data
from src.train import train_model
from src.evaluate import evaluate_model

# --- SCEL Module Components (Conceptual Implementations) ---

class CFAE: # Causal Failure Analysis Engine
    def analyze(self, error_message: str, context: Dict[str, Any]) -> Dict[str, str]:
        print(f"[CFAE] Analyzing failure: {error_message}")
        print(f"[CFAE] Context: {context}")
        
        # Simulate LLM-based analysis for common issues
        if "not found in data" in error_message or "column missing" in error_message:
            root_cause = "Missing or incorrect target/feature column in input data."
            proposed_action = "Verify column names in config and data; potentially adjust data loading or feature selection logic."
            adaptation = {'type': 'data_config_check', 'details': 'Review config.yaml and data headers.'}
        elif "NaN values" in error_message or "input contains NaN" in error_message:
            root_cause = "Inadequate handling of missing values during preprocessing."
            proposed_action = "Implement more robust imputation strategies (e.g., advanced_imputation=True)."
            adaptation = {'type': 'preprocessing_strategy', 'param': 'advanced_imputation', 'value': True}
        elif "singular matrix" in error_message or "data has 0 variance" in error_message:
            root_cause = "Numerical instability or constant features after scaling."
            proposed_action = "Review feature scaling; consider removing low-variance features."
            adaptation = {'type': 'preprocessing_strategy', 'details': 'Review feature selection/scaling.'} # For this demo, just conceptual
        elif "Model not found" in error_message:
            root_cause = "Model training failed or model file was not saved/loaded correctly."
            proposed_action = "Ensure training completes successfully and model path is correct."
            adaptation = {'type': 'retry', 'step': 'train'}
        elif context.get('performance') and context['performance']['accuracy'] < 0.6: # Example of suboptimal outcome
            root_cause = "Model performance is suboptimal, potentially due to overfitting, underfitting, or imbalanced data."
            proposed_action = "Consider increasing epochs, adjusting learning rate, or applying data balancing techniques."
            adaptation = {'type': 'training_strategy', 'param': 'epochs', 'value': context['config']['training']['epochs'] + 5}
        else:
            root_cause = "Unidentified or novel error."
            proposed_action = "Consult documentation or human expert; log full context for future analysis."
            adaptation = {'type': 'log_and_alert'}

        analysis_result = {
            'root_cause': root_cause,
            'proposed_action': proposed_action,
            'adaptation': adaptation
        }
        print(f"[CFAE] Analysis complete. Result: {analysis_result}")
        return analysis_result

class DEKG: # Dynamic Experiential Knowledge Graph (simple list-based for demo)
    def __init__(self):
        self.cases: List[Dict[str, Any]] = []

    def store_case(self, case_details: Dict[str, Any]):
        print(f"[DEKG] Storing new case: {case_details['problem_summary']}")
        self.cases.append(case_details)
        # In a real system, this would update a graph database or persistent store
        # For this demo, just print a confirmation
        print(f"[DEKG] Current knowledge graph size: {len(self.cases)}")

    def retrieve_case(self, problem_description: str) -> Dict[str, Any] or None:
        print(f"[DEKG] Retrieving relevant cases for: '{problem_description}'")
        # Simple keyword-based CBR for demonstration. In a real system, use semantic similarity (e.g., Sentence Transformers).
        best_match = None
        highest_score = 0
        
        keywords = problem_description.lower().split()

        for case in self.cases:
            case_summary = case['problem_summary'].lower()
            score = sum(1 for keyword in keywords if keyword in case_summary)
            if score > highest_score:
                highest_score = score
                best_match = case
        
        if best_match:
            print(f"[DEKG] Found relevant case: {best_match['problem_summary']}")
        else:
            print("[DEKG] No direct relevant case found.")
        return best_match

class SCELModule:
    def __init__(self):
        self.cfae = CFAE()
        self.dekg = DEKG()

    def self_correct(self, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = self.cfae.analyze(error_message, context)
        # Store the failure and analysis as a new case
        case_details = {
            'timestamp': time.time(),
            'problem_summary': error_message,
            'context_snapshot': context,
            'root_cause': analysis['root_cause'],
            'proposed_action': analysis['proposed_action'],
            'adaptation_applied': analysis['adaptation']
        }
        self.dekg.store_case(case_details)
        return analysis['adaptation']

    def proactive_adapt(self, task_description: str) -> Dict[str, Any] or None:
        print(f"[SCEL] Proactively checking DEKG for task: '{task_description}'")
        relevant_case = self.dekg.retrieve_case(task_description)
        if relevant_case and 'adaptation_applied' in relevant_case:
            print(f"[SCEL] Applying proactive adaptation from past experience: {relevant_case['adaptation_applied']}")
            return relevant_case['adaptation_applied']
        return None

# --- Experiment Orchestration ---

def run_experiment(config_path: str, trial: int = 1, scel_module: SCELModule = None, initial_config_override: Dict = None):
    print(f"\n--- Starting Experiment Trial {trial} ---")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply any initial overrides (e.g., from SCEL proactive adaptation)
    if initial_config_override:
        print(f"[Main] Applying initial config overrides: {initial_config_override}")
        if 'data' in initial_config_override and 'advanced_imputation' in initial_config_override['data']:
            config['data']['advanced_imputation'] = initial_config_override['data']['advanced_imputation']
        if 'training' in initial_config_override and 'epochs' in initial_config_override['training']:
             config['training']['epochs'] = initial_config_override['training']['epochs']
        # Add more override logic here as needed for other adaptation types

    X_train, X_test, y_train, y_test = None, None, None, None
    metrics = {}
    
    # Flags for SCEL-driven retries/adaptations
    advanced_imputation_applied = config['data'].get('advanced_imputation', False)
    
    current_trial_status = 'Success'
    failure_details = None

    try:
        print("Step 1: Data Preprocessing...")
        X_train, X_test, y_train, y_test = get_preprocessed_data(config, advanced_imputation=advanced_imputation_applied)
        print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

        print("Step 2: Model Training...")
        model = train_model(X_train, y_train, config)
        
        print("Step 3: Model Evaluation...")
        metrics = evaluate_model(X_test, y_test, config)
        
        if metrics['accuracy'] < config['evaluation']['min_acceptable_accuracy']:
            current_trial_status = 'Suboptimal Performance'
            failure_details = {
                'error_type': 'Suboptimal Performance',
                'message': f"Accuracy {metrics['accuracy']:.2f} is below threshold {config['evaluation']['min_acceptable_accuracy']:.2f}.",
                'context': {'performance': metrics, 'config': config.copy()}
            }
            raise ValueError(failure_details['message']) # Raise an error to trigger SCEL

    except Exception as e:
        print(f"[Experiment Trial {trial}] An error occurred: {e}")
        current_trial_status = 'Failed'
        if failure_details is None: # If not already set by suboptimal performance check
            failure_details = {
                'error_type': 'Execution Error',
                'message': str(e),
                'context': {'stage': 'unknown', 'config': config.copy()}
            }
            if 'Data Preprocessing' in str(e): # Simple heuristic to identify stage
                failure_details['context']['stage'] = 'preprocessing'
            elif 'Model Training' in str(e):
                failure_details['context']['stage'] = 'training'
            elif 'Model Evaluation' in str(e):
                failure_details['context']['stage'] = 'evaluation'
            
        if scel_module:
            print("[Main] Engaging SCEL Module for self-correction.")
            adaptation = scel_module.self_correct(failure_details['message'], failure_details['context'])
            
            if adaptation:
                print(f"[Main] SCEL proposed adaptation: {adaptation}")
                return 'needs_retry', adaptation # Indicate need for retry with new config
            else:
                print("[Main] SCEL did not propose a specific adaptation. Terminating trial.")
                return 'terminated', None # No adaptation, terminate
        else:
            print("[Main] SCEL Module not enabled. Terminating trial.")
            return 'terminated', None

    print(f"--- Experiment Trial {trial} Finished: {current_trial_status} ---")

    # Save results and plot if successful or if a plot is needed for analysis
    if current_trial_status == 'Success':
        results_dir = os.path.join('.research', 'iteration1', 'images')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(results_dir, f'metrics_trial_{trial}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {os.path.join(results_dir, f'metrics_trial_{trial}.json')}")

        # Create a dummy plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title(f'Model Evaluation Metrics - Trial {trial}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plot_path = os.path.join(results_dir, f'evaluation_metrics_trial_{trial}.pdf')
        plt.savefig(plot_path, format='pdf')
        plt.close()
        print(f"Metrics plot saved to {plot_path}")
        
    return 'success', None

def main():
    print("Starting AutoMind SCEL Experiment Orchestration...")
    
    # Ensure necessary directories exist
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('config', exist_ok=True)

    config_path = 'config/config.yaml'
    
    # Create a dummy data file if it doesn't exist for initial run
    dummy_data_path = os.path.join('data', 'sample_data.csv')
    if not os.path.exists(dummy_data_path):
        print(f"Creating dummy data at {dummy_data_path}")
        pd.DataFrame({
            'feature_1': np.random.rand(200),
            'feature_2': np.random.randint(0, 5, 200).astype(float),
            'feature_3': ['X', 'Y', 'Z', 'X', 'Y'] * 40,
            'target': np.random.randint(0, 2, 200)
        }).to_csv(dummy_data_path, index=False)
        # Introduce some missing values for testing SCEL
        df_temp = pd.read_csv(dummy_data_path)
        df_temp.loc[50:60, 'feature_1'] = np.nan
        df_temp.loc[70, 'feature_3'] = np.nan
        df_temp.to_csv(dummy_data_path, index=False)

    scel_module = SCELModule()
    
    max_trials = 3 # Allow for initial run + 2 self-correction attempts
    current_trial = 1
    config_overrides = None # This will hold dictionary of overrides
    
    while current_trial <= max_trials:
        print(f"\n{'='*50}\nRunning Full Experiment Cycle - Trial {current_trial}\n{'='*50}")
        
        task_description_for_proactive_check = "tabular classification with potential missing values and categorical features."
        proactive_adaptation = scel_module.proactive_adapt(task_description_for_proactive_check)
        
        # Merge proactive adaptations into config_overrides
        if proactive_adaptation:
            if config_overrides is None:
                config_overrides = {}
            if proactive_adaptation['type'] == 'preprocessing_strategy' and proactive_adaptation['param'] == 'advanced_imputation':
                if 'data' not in config_overrides: config_overrides['data'] = {}
                config_overrides['data']['advanced_imputation'] = proactive_adaptation['value']
            elif proactive_adaptation['type'] == 'training_strategy' and proactive_adaptation['param'] == 'epochs':
                 if 'training' not in config_overrides: config_overrides['training'] = {}
                 config_overrides['training']['epochs'] = proactive_adaptation['value']
            print(f"[Main] Proactively adjusting config based on DEKG: {config_overrides}")

        status, next_adaptation = run_experiment(config_path, current_trial, scel_module, initial_config_override=config_overrides)
        
        if status == 'success':
            print("Experiment succeeded after self-correction or on first attempt.")
            break
        elif status == 'needs_retry':
            print("Experiment failed, attempting self-correction and retry.")
            # Update config overrides for next trial based on SCEL's proposed adaptation
            if next_adaptation:
                if config_overrides is None:
                    config_overrides = {}
                if next_adaptation['type'] == 'preprocessing_strategy' and next_adaptation['param'] == 'advanced_imputation':
                    if 'data' not in config_overrides: config_overrides['data'] = {}
                    config_overrides['data']['advanced_imputation'] = next_adaptation['value']
                elif next_adaptation['type'] == 'training_strategy' and next_adaptation['param'] == 'epochs':
                    if 'training' not in config_overrides: config_overrides['training'] = {}
                    config_overrides['training']['epochs'] = next_adaptation['value']
            
            # For demo purposes, let's artificially introduce an error scenario after the first trial
            # to ensure SCEL has something to correct on subsequent runs if the first one was too perfect.
            if current_trial == 1 and status == 'needs_retry': # If 1st trial failed and needs retry
                print("[Main] Forcing a 'missing value' scenario for demo purposes for the next trial.")
                # Modify the config for next trial to simulate a problem that SCEL will solve
                with open(config_path, 'r') as f:
                    temp_config = yaml.safe_load(f)
                temp_config['data']['filename'] = 'sample_data_with_nans.csv' # Assume this file has issues
                with open(config_path, 'w') as f:
                    yaml.safe_dump(temp_config, f)
                # Create the 'problematic' file with guaranteed NaNs for the next run
                pd.DataFrame({
                    'feature_1': np.random.rand(200),
                    ''feature_2': np.random.randint(0, 5, 200).astype(float),
                    'feature_3': ['X', 'Y', 'Z', 'X', 'Y'] * 40,
                    'target': np.random.randint(0, 2, 200)
                }).to_csv(os.path.join('data', 'sample_data_with_nans.csv'), index=False)
                df_problem = pd.read_csv(os.path.join('data', 'sample_data_with_nans.csv'))
                df_problem.loc[0:100, 'feature_1'] = np.nan # Lots of NaNs
                df_problem.to_csv(os.path.join('data', 'sample_data_with_nans.csv'), index=False)

            current_trial += 1
        else: # 'terminated'
            print("Experiment terminated due to unresolvable errors or max retries.")
            break

    if current_trial > max_trials:
        print(f"Max trials ({max_trials}) reached. Experiment could not converge to a successful state.")
    print("AutoMind SCEL Experiment Orchestration finished.")

if __name__ == '__main__':
    main()
