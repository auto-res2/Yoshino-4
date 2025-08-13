import os
import json
import logging
import yaml
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas for pd.Series

# Ensure .research/iteration1/images directory exists
os.makedirs(os.path.join(".research", "iteration1", "images"), exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import modules from src
import src.preprocess as preprocess
import src.train as train
import src.evaluate as evaluate

# --- Enums for Causal Failure Analysis ---
class ProblemType(Enum):
    PREPROCESSING_ERROR = "Preprocessing Error"
    TRAINING_ERROR = "Training Error"
    EVALUATION_ERROR = "Evaluation Error"
    SUBOPTIMAL_PERFORMANCE = "Suboptimal Performance"

class RootCause(Enum):
    DATA_QUALITY_ISSUE = "Data Quality Issue"
    MISSING_VALUES = "Missing Values"
    NON_NUMERIC_DATA = "Non-Numeric Data"
    OUTLIERS = "Outliers"
    DATA_IMBALANCE = "Data Imbalance"
    MODEL_OVERFITTING = "Model Overfitting"
    MODEL_UNDERFITTING = "Model Underfitting"
    INVALID_HYPERPARAMETERS = "Invalid Hyperparameters"
    NO_TARGET_VARIABLE = "No Target Variable"
    UNKNOWN_ERROR = "Unknown Error"

class ProposedSolution(Enum):
    APPLY_MISSING_VALUE_HANDLING = "Apply Missing Value Handling"
    CONVERT_DATA_TYPES = "Convert Data Types"
    APPLY_FEATURE_SCALING = "Apply Feature Scaling"
    APPLY_OUTLIER_HANDLING = "Apply Outlier Handling"
    APPLY_REGULARIZATION = "Apply Regularization"
    REDUCE_COMPLEXITY = "Reduce Model Complexity"
    INCREASE_COMPLEXITY = "Increase Model Complexity"
    ADJUST_HYPERPARAMETERS = "Adjust Hyperparameters"
    IMPROVE_DATA_QUALITY = "Improve Data Quality"
    CHECK_TARGET_VARIABLE = "Check Target Variable Definition"
    GENERIC_TROUBLESHOOT = "Generic Troubleshooting"

# --- SCEL Agent Implementation ---
class SCELAgent:
    def __init__(self, config: dict):
        self.config = config
        self.dekg_path = config['dekg_path']
        self.embedding_model_name = config['embedding_model']
        self.cbr_top_k = config['cbr_top_k']
        
        self.dekg = self._load_dekg()
        self.sentence_model = SentenceTransformer(self.embedding_model_name)
        logging.info(f"Initialized SCELAgent with DEKG from {self.dekg_path} and embedding model {self.embedding_model_name}")

    def _load_dekg(self) -> list:
        """Loads DEKG from file or initializes an empty one."""
        if os.path.exists(self.dekg_path):
            try:
                with open(self.dekg_path, 'r') as f:
                    dekg_data = json.load(f)
                    logging.info(f"Loaded {len(dekg_data)} cases from DEKG.")
                    return dekg_data
            except Exception as e:
                logging.warning(f"Could not load DEKG from {self.dekg_path}: {e}. Initializing empty DEKG.")
                return []
        logging.info("DEKG file not found. Initializing empty DEKG.")
        return []

    def _save_dekg(self):
        """Saves the current DEKG to file."""
        os.makedirs(os.path.dirname(self.dekg_path), exist_ok=True)
        with open(self.dekg_path, 'w') as f:
            json.dump(self.dekg, f, indent=2)
        logging.info(f"DEKG saved to {self.dekg_path}")

    def causal_failure_analysis(self, problem_type: ProblemType, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates the Causal Failure Analysis Engine (CFAE) using a rule-based approach.
        In a real system, this would involve a fine-tuned LLM.
        """
        logging.info(f"CFAE initiated for {problem_type.value}: '{error_message}' in context {context}")
        
        root_cause = RootCause.UNKNOWN_ERROR
        proposed_solution = ProposedSolution.GENERIC_TROUBLESHOOT
        
        # Rule-based mapping for demonstration
        if "non-numeric" in error_message.lower() and "scale" in error_message.lower():
            root_cause = RootCause.NON_NUMERIC_DATA
            proposed_solution = ProposedSolution.CONVERT_DATA_TYPES
        elif "unhandled missing values" in error_message.lower():
            root_cause = RootCause.MISSING_VALUES
            proposed_solution = ProposedSolution.APPLY_MISSING_VALUE_HANDLING
        elif "target variable has only one unique value" in error_message.lower() or "target' column not found" in error_message.lower():
            root_cause = RootCause.NO_TARGET_VARIABLE
            proposed_solution = ProposedSolution.CHECK_TARGET_VARIABLE
        elif context.get('overfitting_detected'):
            root_cause = RootCause.MODEL_OVERFITTING
            proposed_solution = ProposedSolution.APPLY_REGULARIZATION
        elif problem_type == ProblemType.SUBOPTIMAL_PERFORMANCE:
            # Suboptimal performance needs deeper context, but for mock, let's assume overfitting if high train/low test
            if context.get('train_accuracy', 0) > 0.9 and context.get('test_accuracy', 0) < 0.7:
                 root_cause = RootCause.MODEL_OVERFITTING
                 proposed_solution = ProposedSolution.APPLY_REGULARIZATION
            else: # Placeholder for other suboptimal cases
                 root_cause = RootCause.INVALID_HYPERPARAMETERS # Could be underfitting too
                 proposed_solution = ProposedSolution.ADJUST_HYPERPARAMETERS
        
        logging.info(f"  Inferred Root Cause: {root_cause.value}")
        logging.info(f"  Proposed Solution: {proposed_solution.value}")

        return {
            "problem_type": problem_type.value,
            "error_message": error_message,
            "context": context,
            "root_cause": root_cause.value,
            "proposed_solution": proposed_solution.value
        }

    def add_case_to_dekg(self, case_data: Dict[str, Any]):
        """
        Adds a new case (failure scenario and resolution) to the DEKG.
        For conceptual DEKG, each case is a dictionary with an embedding.
        """
        case_string = f"Problem: {case_data['problem_type']}. Error: {case_data['error_message']}. Root Cause: {case_data['root_cause']}. Solution: {case_data['proposed_solution']}. Context: {json.dumps(case_data['context'])}"
        case_data['embedding'] = self.sentence_model.encode(case_string).tolist()
        self.dekg.append(case_data)
        logging.info("Case added to DEKG.")
        self._save_dekg()

    def retrieve_relevant_cases(self, current_context_str: str) -> List[Dict[str, Any]]:
        """
        Retrieves relevant cases from DEKG using Case-Based Reasoning (CBR).
        Uses sentence embeddings for semantic similarity.
        """
        if not self.dekg:
            return []

        current_embedding = self.sentence_model.encode(current_context_str)
        similarities = []
        for i, case in enumerate(self.dekg):
            if 'embedding' in case:
                case_embedding = np.array(case['embedding'])
                similarity = util.cos_sim(current_embedding, case_embedding).item()
                similarities.append((similarity, case))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k_cases = [case for sim, case in similarities[:self.cbr_top_k] if sim > 0.7] # Threshold similarity
        
        if top_k_cases:
            logging.info(f"Retrieved {len(top_k_cases)} relevant cases from DEKG.")
            for sim, case in similarities[:self.cbr_top_k]:
                logging.info(f"  - Case (Sim: {sim:.2f}): Problem: {case['problem_type']}, Root Cause: {case['root_cause']}, Solution: {case['proposed_solution']}")
        else:
            logging.info("No highly similar cases found in DEKG.")
        return top_k_cases

    def proactive_adaptive_planning(self, initial_params: dict, current_task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modifies the plan based on 'lessons learned' from DEKG before execution.
        """
        context_str = f"Task Type: Data Science. Initial Params: {initial_params}. Current Data Characteristics: {current_task_context.get('data_characteristics', 'N/A')}"
        
        relevant_cases = self.retrieve_relevant_cases(context_str)
        
        adapted_params = initial_params.copy()
        modifications_made = []

        # Apply adaptations based on relevant cases
        for case in relevant_cases:
            solution = ProposedSolution(case['proposed_solution'])
            if solution == ProposedSolution.APPLY_MISSING_VALUE_HANDLING:
                if not adapted_params.get('preprocessing', {}).get('handle_missing'):
                    adapted_params.setdefault('preprocessing', {})['handle_missing'] = True
                    modifications_made.append("Added missing value handling.")
            elif solution == ProposedSolution.CONVERT_DATA_TYPES:
                # This is harder to simulate directly in params, implies a pre-analysis step
                # For mock, we'll just note it.
                modifications_made.append("Considered data type conversion for numerical columns.")
            elif solution == ProposedSolution.APPLY_OUTLIER_HANDLING:
                if not adapted_params.get('preprocessing', {}).get('adaptive_outlier_handling'):
                    adapted_params.setdefault('preprocessing', {})['adaptive_outlier_handling'] = True
                    modifications_made.append("Added adaptive outlier handling.")
            elif solution == ProposedSolution.APPLY_REGULARIZATION:
                if not adapted_params.get('training', {}).get('add_regularization'):
                    adapted_params.setdefault('training', {})['add_regularization'] = True
                    # Also might adjust epochs or LR to mitigate overfitting
                    if adapted_params.get('training', {}).get('epochs', 0) > 10:
                        adapted_params['training']['epochs'] = 10 # Reduce epochs
                    modifications_made.append("Added regularization and adjusted epochs.")
            elif solution == ProposedSolution.ADJUST_HYPERPARAMETERS:
                # Generic adjustment for demonstration
                adapted_params.setdefault('training', {})['learning_rate'] = adapted_params.get('training', {}).get('learning_rate', 0.001) * 0.5 # Reduce LR
                modifications_made.append("Adjusted hyperparameters (e.g., reduced learning rate).")
            # Add more specific adaptations as needed

        if modifications_made:
            logging.info(f"Proactive Adaptation: Applied modifications: {', '.join(modifications_made)}")
        else:
            logging.info("Proactive Adaptation: No relevant cases found for adaptation or current plan already addresses.")

        return adapted_params

    def meta_policy_refinement(self, success: bool, strategy_applied: Dict[str, Any]):
        """
        Conceptually refines meta-policies based on the outcome of a task.
        In a full system, this would update internal planning heuristics.
        """
        if success:
            logging.info("Meta-Policy Refinement: Task successful. Reinforcing applied strategy and associated meta-policies.")
            # Logic to strengthen connections/weights for successful strategies in a real system
        else:
            logging.info("Meta-Policy Refinement: Task failed/suboptimal. Re-evaluating applied strategy and adjusting meta-policies.")
            # Logic to weaken connections/weights or explore alternative strategies

    def run_task(self, task_id: str, initial_params: dict, data_scenario: str = "default", 
                 simulate_preprocess_error: bool = False, preprocess_error_type: Optional[str] = None,
                 simulate_train_overfitting: bool = False):
        """Orchestrates a single data science task with SCEL intervention."""
        logging.info(f"\n--- Running Task: {task_id} ---")
        current_task_context = {
            "task_id": task_id,
            "simulate_preprocess_error": simulate_preprocess_error,
            "preprocess_error_type": preprocess_error_type,
            "simulate_train_overfitting": simulate_train_overfitting,
            "data_characteristics": data_scenario # Pass data characteristics for context
        }

        # 1. Proactive Adaptive Planning
        logging.info("Phase 1: Proactive Adaptive Planning...")
        adapted_params = self.proactive_adaptive_planning(initial_params, current_task_context)
        logging.info(f"  Initial Params: {initial_params}")
        logging.info(f"  Adapted Params: {adapted_params}")
        
        task_success = False
        final_score = None
        error_details = None
        applied_strategy = {"preprocessing": adapted_params.get("preprocessing", {}), "training": adapted_params.get("training", {})}

        try:
            # Generate mock data specific to the scenario
            mock_data = preprocess.generate_mock_data(scenario=data_scenario)
            
            # 2. Preprocessing
            logging.info("Phase 2: Data Preprocessing...")
            processed_data = preprocess.preprocess_data(mock_data, adapted_params.get('preprocessing', {}), current_task_context)
            logging.info("  Preprocessing successful.")

            # 3. Training
            logging.info("Phase 3: Model Training...")
            model_output = train.train_model(processed_data, adapted_params.get('training', {}), current_task_context)
            logging.info("  Training successful.")
            
            # 4. Evaluation
            logging.info("Phase 4: Model Evaluation...")
            # Pass true labels directly from model_output
            true_labels_for_eval = pd.Series(model_output.get("true_labels_test", []))
            
            eval_results = evaluate.evaluate_model(model_output, true_labels_for_eval, adapted_params.get('evaluation', {}), current_task_context)
            final_score = eval_results['metrics']['f1_score']
            
            if eval_results['metrics']['is_suboptimal'] or current_task_context.get('overfitting_detected'):
                logging.warning("Task resulted in suboptimal performance or detected overfitting.")
                # Engage CFAE for suboptimal performance
                analysis_result = self.causal_failure_analysis(
                    ProblemType.SUBOPTIMAL_PERFORMANCE,
                    f"Model performance (F1-score: {final_score:.2f}) is below threshold.",
                    {"task_id": task_id, "metrics": eval_results['metrics'], "current_params": adapted_params, 
                     "train_accuracy": model_output.get('train_accuracy'), "test_accuracy": model_output.get('test_accuracy'),
                     "overfitting_detected": current_task_context.get('overfitting_detected', False)}
                )
                self.add_case_to_dekg(analysis_result)
                task_success = False # Mark as failure for meta-policy, even if no hard error
                error_details = analysis_result
            else:
                logging.info(f"Task completed successfully with F1-score: {final_score:.4f}")
                task_success = True

        except (TypeError, ValueError) as e:
            logging.error(f"Task {task_id} failed with a known error: {e}")
            problem_type = ProblemType.UNKNOWN_ERROR
            if "PreprocessingError" in str(e):
                problem_type = ProblemType.PREPROCESSING_ERROR
            elif "TrainingError" in str(e):
                problem_type = ProblemType.TRAINING_ERROR
            elif "EvaluationError" in str(e):
                problem_type = ProblemType.EVALUATION_ERROR

            analysis_result = self.causal_failure_analysis(
                problem_type,
                str(e),
                {"task_id": task_id, "current_params": adapted_params, "last_successful_phase": "N/A"}
            )
            self.add_case_to_dekg(analysis_result)
            task_success = False
            error_details = analysis_result

        except Exception as e:
            logging.critical(f"Task {task_id} failed with an unexpected error: {e}", exc_info=True)
            analysis_result = self.causal_failure_analysis(
                ProblemType.UNKNOWN_ERROR,
                str(e),
                {"task_id": task_id, "current_params": adapted_params, "last_successful_phase": "N/A"}
            )
            self.add_case_to_dekg(analysis_result)
            task_success = False
            error_details = analysis_result

        # 5. Meta-Policy Refinement
        logging.info("Phase 5: Meta-Policy Refinement...")
        self.meta_policy_refinement(task_success, applied_strategy)

        logging.info(f"--- Task {task_id} Complete. Status: {'SUCCESS' if task_success else 'FAILURE/SUBOPTIMAL'} ---")
        if not task_success and error_details:
            logging.info(f"  Root Cause Identified: {error_details.get('root_cause')}")
            logging.info(f"  Proposed Solution: {error_details.get('proposed_solution')}")
        elif task_success:
            logging.info(f"  Final F1-score: {final_score:.4f}")
        
        return task_success, final_score, error_details

def plot_dekg_similarity(agent: SCELAgent, task_id: str, current_context_str: str, output_dir: str):
    """Visualizes similarity scores for a given context against DEKG cases."""
    if not agent.dekg:
        logging.info("DEKG is empty, cannot plot similarity.")
        return

    current_embedding = agent.sentence_model.encode(current_context_str)
    similarities = []
    case_labels = []
    
    for i, case in enumerate(agent.dekg):
        if 'embedding' in case:
            case_embedding = np.array(case['embedding'])
            similarity = util.cos_sim(current_embedding, case_embedding).item()
            similarities.append(similarity)
            case_labels.append(f"Case {i+1} ({case['root_cause']})")
    
    if not similarities:
        logging.info("No cases with embeddings in DEKG to plot similarity.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=similarities, y=case_labels, palette="viridis")
    plt.xlabel("Cosine Similarity")
    plt.title(f"DEKG Case Similarity for Task {task_id}")
    plt.xlim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dekg_similarity_task_{task_id}.pdf")
    plt.savefig(plot_path, format='pdf')
    plt.close()
    logging.info(f"DEKG similarity plot saved to {plot_path}")

if __name__ == '__main__':
    logging.info("Starting AutoMind SCEL Experiment...")
    
    # Load configuration
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize SCEL Agent
    scel_agent = SCELAgent(config)

    # --- Run simulated data science tasks ---
    results_summary = []
    image_output_dir = os.path.join(".research", "iteration1", "images")
    os.makedirs(image_output_dir, exist_ok=True)

    for task_config in config['task_configs']:
        task_id = task_config['id']
        initial_params = task_config['initial_params']
        data_scenario = task_config.get('data_scenario', 'default') # Retrieve data_scenario from config

        success, final_score, error_details = scel_agent.run_task(
            task_id,
            initial_params,
            data_scenario=data_scenario,
            simulate_preprocess_error=task_config.get('simulate_preprocess_error', False),
            preprocess_error_type=task_config.get('preprocess_error_type'),
            simulate_train_overfitting=task_config.get('simulate_train_overfitting', False)
        )
        
        results_summary.append({
            "task_id": task_id,
            "status": "SUCCESS" if success else "FAILURE/SUBOPTIMAL",
            "final_score": final_score,
            "error_details": error_details
        })

        # Plot DEKG similarity after each task to see how new cases might affect it
        # This context string should ideally be derived from the actual data/task at hand
        current_context_for_plot = f"Attempting task {task_id} with params {json.dumps(initial_params)}. Data scenario: {data_scenario}"
        plot_dekg_similarity(scel_agent, task_id, current_context_for_plot, image_output_dir)

    logging.info("\n--- Experiment Summary ---")
    for res in results_summary:
        logging.info(f"Task {res['task_id']}: Status: {res['status']}, F1-Score: {res['final_score']:.4f}" if res['final_score'] is not None else f"Task {res['task_id']}: Status: {res['status']}")
        if res['error_details']:
            logging.info(f"  Problem: {res['error_details']['problem_type']}, Root Cause: {res['error_details']['root_cause']}, Solution: {res['error_details']['proposed_solution']}")

    logging.info(f"\nDEKG contains {len(scel_agent.dekg)} learned cases.")
    logging.info(f"All evaluation plots and DEKG similarity plots saved to: {image_output_dir}")
    logging.info("AutoMind SCEL Experiment Complete.")
