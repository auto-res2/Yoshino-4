import os
import yaml
import json
import networkx as nx # Not directly used in simplified demo, but kept as per original thought for DEKG
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Import components - assuming they are in the same 'src' directory
from preprocess import generate_synthetic_data, get_data_characteristics, apply_preprocessing
from train import train_model, SimpleMLP
from evaluate import evaluate_model

class SCELAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dekg_path = config['scel_config']['dekg_path']
        self.images_dir = config['scel_config']['images_dir']
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.dekg_path), exist_ok=True) # Ensure data/ directory exists
        self.dekg = self._load_dekg()
        print(f"SCELAgent initialized. DEKG loaded with {len(self.dekg)} cases.")

    def _load_dekg(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.dekg_path):
            try:
                with open(self.dekg_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {self.dekg_path}. Starting with empty DEKG.")
                return []
        return []

    def _save_dekg(self):
        with open(self.dekg_path, 'w') as f:
            json.dump(self.dekg, f, indent=2)
        print(f"DEKG saved to {self.dekg_path}")

    def causal_failure_analysis_engine(self, data_characteristics: Dict[str, Any], initial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates CFAE: analyzes data characteristics and initial performance to
        propose potential root causes and adaptive strategies.
        This is a simplified rule-based system instead of an LLM for the demo.
        """
        print("\n--- CFAE: Performing Root Cause Analysis ---")
        proposed_adaptations = {
            'impute_missing': False,
            'scale_features': False,
            'apply_regularization': False,
            'apply_class_weights': False
        }
        root_causes = []
        problem_description_elements = [] # Use a list to build description

        if data_characteristics.get('has_missing_values'):
            root_causes.append('Missing values in raw data.')
            problem_description_elements.append('data_has_missing_values')
            proposed_adaptations['impute_missing'] = True
            print("- Detected: Missing values. Proposed: Imputation.")

        if data_characteristics.get('has_unscaled_features'):
            root_causes.append('Unscaled numerical features.')
            problem_description_elements.append('data_has_unscaled_features')
            proposed_adaptations['scale_features'] = True
            print("- Detected: Unscaled features. Proposed: Scaling.")

        if data_characteristics.get('is_imbalanced'):
            root_causes.append('Class imbalance in target variable.')
            problem_description_elements.append('data_is_imbalanced')
            proposed_adaptations['apply_class_weights'] = True
            print("- Detected: Class imbalance. Proposed: Class weighting.")
            
        # Analyze performance metrics for potential overfitting/underfitting or specific issues
        # If high accuracy but low F1-score, it often indicates class imbalance or poor minority class performance
        if initial_metrics['accuracy'] > 0.75 and initial_metrics['f1_score'] < 0.65: 
            if not proposed_adaptations['apply_class_weights']: # If not already handled by data char
                root_causes.append('Poor minority class performance despite high accuracy (suggests imbalance).')
                problem_description_elements.append('low_f1_high_accuracy')
                proposed_adaptations['apply_class_weights'] = True
                print("- Detected: High accuracy, low F1. Re-emphasizing: Class weighting.")
        
        # If F1-score is generally very low, could be underfitting or highly noisy data
        if initial_metrics['f1_score'] < 0.4: 
             root_causes.append('Model underfitting or high bias (very low F1-score).')
             problem_description_elements.append('model_underfitting')
             print("- Detected: Very low F1. Consider more complex model or better features (beyond current scope).")

        # Simple rule for potential overfitting (if model trained without regularization and performance is high but F1 still not great)
        if not self.config['model_config']['apply_regularization'] and initial_metrics['accuracy'] > 0.9 and initial_metrics['f1_score'] < 0.7: 
            root_causes.append('Potential overfitting as accuracy is high but F1 is not matching.')
            problem_description_elements.append('potential_overfitting')
            proposed_adaptations['apply_regularization'] = True
            print("- Detected: Potential overfitting. Proposed: Regularization.")

        if not root_causes:
            root_causes.append('No obvious issues detected based on initial analysis.')
            problem_description_elements.append('no_obvious_problem')
            print("- No critical issues detected by CFAE initially.")

        return {
            'root_causes': root_causes,
            'problem_description': " ".join(sorted(list(set(problem_description_elements)))), # Standardize description
            'proposed_adaptations': proposed_adaptations
        }

    def case_based_reasoning(self, problem_description: str) -> Optional[Dict[str, Any]]:
        """
        Simulates CBR: Retrieves the most relevant past case from DEKG.
        Uses simple string matching (Jaccard similarity) for demo purposes.
        In a real system, this would use sentence embeddings and vector similarity.
        """
        print(f"\n--- CBR: Searching DEKG for similar cases to '{problem_description}' ---")
        best_match = None
        max_similarity = 0.0

        if not self.dekg:
            print("DEKG is empty. No cases to retrieve.")
            return None

        # Simplified similarity: Jaccard similarity of keywords.
        problem_keywords = set(problem_description.lower().split())

        for case in self.dekg:
            case_keywords = set(case['problem_description'].lower().split())
            intersection = len(problem_keywords.intersection(case_keywords))
            union = len(problem_keywords.union(case_keywords))
            
            if union > 0:
                jaccard_similarity = intersection / union
                if jaccard_similarity > max_similarity:
                    max_similarity = jaccard_similarity
                    best_match = case
        
        if best_match and max_similarity > 0.5: # Threshold for considering a match
            print(f"Found relevant case with similarity {max_similarity:.2f}: {best_match['problem_description']}")
            return best_match
        
        print("No sufficiently similar case found in DEKG.")
        return None

    def update_dekg(self, data_characteristics: Dict[str, Any], initial_metrics: Dict[str, Any],
                    problem_description: str, proposed_adaptations: Dict[str, Any],
                    adapted_metrics: Dict[str, Any]):
        """
        Updates the Dynamic Experiential Knowledge Graph (DEKG) with new learning.
        A 'case' includes the problem, root cause, proposed solution, and outcome.
        """
        print("\n--- DEKG: Updating Knowledge Graph ---")
        
        # Check if the adaptation was successful based on F1-score improvement
        success = adapted_metrics['f1_score'] > initial_metrics['f1_score'] # Simple success metric
        outcome = "SUCCESS" if success else "FAILURE"
        
        # Create a new case to add to the DEKG
        new_case = {
            'timestamp': datetime.now().isoformat(),
            'problem_description': problem_description,
            'data_characteristics_at_failure': data_characteristics,
            'initial_metrics': initial_metrics,
            'proposed_adaptations': proposed_adaptations, # These are the adaptations proposed by CFAE
            'adapted_metrics': adapted_metrics,
            'outcome': outcome,
            'notes': 'Simulated case from SCEL experiment.'
        }
        
        self.dekg.append(new_case)
        self._save_dekg()
        print(f"DEKG updated with a new case (outcome: {outcome}). Total cases: {len(self.dekg)}")

def run_experiment(agent: SCELAgent, config: Dict[str, Any]):
    print("\n--- Starting SCEL Experiment Run ---")
    data_config = config['data_config']
    model_config = config['model_config']
    
    # --- Step 1: Generate initial data and get characteristics --- 
    print("\n[Step 1/7] Generating initial synthetic data...")
    raw_data = generate_synthetic_data(data_config)
    data_characteristics = get_data_characteristics(raw_data)
    print("Initial Data Characteristics:", json.dumps(data_characteristics, indent=2))

    # --- Step 2: Initial Run (without SCEL proactive adaptation) ---
    print("\n[Step 2/7] Running initial pipeline (no SCEL adaptation)...")
    initial_preprocessing_adaptations = {
        'impute_missing': False, 
        'scale_features': False
    }
    initial_model_config = model_config.copy() 
    initial_model_config['apply_regularization'] = False
    initial_model_config['apply_class_weights'] = False
    
    X_initial_processed, y_initial_processed, preprocessor_initial_obj = apply_preprocessing(raw_data, initial_preprocessing_adaptations)
    
    # Ensure X is dense numpy array for training
    X_initial_numpy = X_initial_processed.toarray() if hasattr(X_initial_processed, 'toarray') else X_initial_processed
    y_initial_numpy = y_initial_processed.to_numpy() if hasattr(y_initial_processed, 'to_numpy') else y_initial_processed

    trained_initial_model = train_model(X_initial_numpy, y_initial_numpy, initial_model_config)
    initial_metrics = evaluate_model(trained_initial_model, X_initial_numpy, y_initial_numpy, agent.images_dir)
    print("Initial Pipeline Metrics:", json.dumps(initial_metrics, indent=2))
    
    # --- Step 3: CFAE - Root Cause Analysis ---
    print("\n[Step 3/7] SCEL: Invoking Causal Failure Analysis Engine (CFAE)...")
    cfa_result = agent.causal_failure_analysis_engine(data_characteristics, initial_metrics)
    print("CFAE Analysis Result:", json.dumps(cfa_result, indent=2))
    
    problem_description = cfa_result['problem_description']
    proposed_adaptations_from_cfa = cfa_result['proposed_adaptations']

    # --- Step 4: CBR - Retrieve Past Cases ---
    print("\n[Step 4/7] SCEL: Invoking Case-Based Reasoning (CBR)...")
    relevant_case = agent.case_based_reasoning(problem_description)
    
    # Initialize adapted parameters with CFAE suggestions
    adapted_preprocessing_params = proposed_adaptations_from_cfa.copy()
    adapted_model_params = proposed_adaptations_from_cfa.copy()

    if relevant_case:
        print("CBR: Found relevant past case. Merging learned adaptations.")
        # In a real system, this merging logic could be complex (e.g., weighting, conflict resolution)
        # For demo, if CBR found a case with specific adaptation, ensure it's applied.
        # This simple merge ensures anything CFAE proposes or CBR confirms is applied.
        for key, value in relevant_case['proposed_adaptations'].items():
            if key in adapted_preprocessing_params:
                adapted_preprocessing_params[key] = adapted_preprocessing_params[key] or value
            if key in adapted_model_params:
                adapted_model_params[key] = adapted_model_params[key] or value
    else:
        print("CBR: No sufficiently relevant case found. Proceeding with CFAE proposed adaptations.")

    # --- Step 5: Proactive Adaptive Planning & Execution ---
    print("\n[Step 5/7] SCEL: Applying proactive adaptations and re-running pipeline...")
    
    # Apply adaptations to the model config for the adapted run
    adapted_model_config = model_config.copy()
    adapted_model_config['apply_regularization'] = adapted_model_params['apply_regularization']
    adapted_model_config['apply_class_weights'] = adapted_model_params['apply_class_weights']

    # Preprocessing with adaptations
    X_adapted_processed, y_adapted_processed, preprocessor_adapted_obj = apply_preprocessing(raw_data, adapted_preprocessing_params)

    X_adapted_numpy = X_adapted_processed.toarray() if hasattr(X_adapted_processed, 'toarray') else X_adapted_processed
    y_adapted_numpy = y_adapted_processed.to_numpy() if hasattr(y_adapted_processed, 'to_numpy') else y_adapted_processed

    # Train the adapted model
    trained_adapted_model = train_model(X_adapted_numpy, y_adapted_numpy, adapted_model_config)

    # Evaluate the adapted model
    adapted_metrics = evaluate_model(trained_adapted_model, X_adapted_numpy, y_adapted_numpy, agent.images_dir)
    print("Adapted Pipeline Metrics:", json.dumps(adapted_metrics, indent=2))

    # --- Step 6: Meta-Policy Refinement & DEKG Update ---
    print("\n[Step 6/7] SCEL: Updating Dynamic Experiential Knowledge Graph (DEKG)...")
    agent.update_dekg(data_characteristics, initial_metrics, problem_description, proposed_adaptations_from_cfa, adapted_metrics)

    # --- Step 7: Summary and Comparison ---
    print("\n--- Experiment Summary ---")
    print(f"Initial Run F1-Score: {initial_metrics['f1_score']:.4f}")
    print(f"Adapted Run F1-Score: {adapted_metrics['f1_score']:.4f}")

    if adapted_metrics['f1_score'] > initial_metrics['f1_score']:
        print("Conclusion: The SCEL-driven adaptations improved performance!")
    elif adapted_metrics['f1_score'] == initial_metrics['f1_score']:
        print("Conclusion: The SCEL-driven adaptations resulted in similar performance.")
    else:
        print("Conclusion: The SCEL-driven adaptations did not improve performance or performance decreased. Further analysis needed.")
    print("--- SCEL Experiment Run Complete ---")

if __name__ == '__main__':
    # Load configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/config.yaml not found. Please create it under the 'config' directory.")
        exit(1)
    
    agent = SCELAgent(config)
    run_experiment(agent, config)