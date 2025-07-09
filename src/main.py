#!/usr/bin/env python
"""
Main orchestration script for CTD-E experiments.
Runs all three experiments and generates comprehensive results with PDF plots.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from preprocess import prepare_sample_texts, create_external_corpus, save_preprocessed_data
from train import CTDEvaluator, baseline_evaluator, train_ctd_evaluator
from evaluate import (
    evaluate_tree_comparison, evaluate_retrieval_refinement, 
    evaluate_iterative_refinement, compute_evaluation_metrics,
    create_corpus_index, refine_node_with_retrieval
)

def setup_experiment_environment():
    """Set up the experiment environment and directories."""
    print("Setting up experiment environment...")
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    plt.switch_backend('Agg')
    
    print("Environment setup completed.")

def experiment1_dynamic_tree():
    """Experiment 1: Dynamic Critic Tree Construction."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Dynamic Critic Tree Construction")
    print("="*60)
    
    samples = prepare_sample_texts()
    print(f"Processing {len(samples)} samples...")
    
    evaluator = CTDEvaluator()
    
    ctd_e_results = []
    baseline_results = []
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {sample[:50]}...")
        
        tree = evaluator.generate_critic_tree(sample)
        ctd_e_results.append(tree)
        print(f"  CTD-E Tree: {tree['criterion'][:50]}...")
        print(f"  Subcriteria: {len(tree['subcriteria'])}")
        
        baseline = baseline_evaluator(sample)
        baseline_results.append(baseline)
        print(f"  Baseline criteria: {len(baseline)}")
    
    comparison_results = evaluate_tree_comparison(samples, ctd_e_results, baseline_results)
    print(f"\nComparison Results:")
    print(f"  Average CTD-E nodes: {comparison_results['avg_ctd_e_nodes']:.2f}")
    print(f"  Average baseline nodes: {comparison_results['avg_baseline_nodes']:.2f}")
    print(f"  Improvement ratio: {comparison_results['improvement_ratio']:.2f}")
    
    plot_tree_comparison(samples, ctd_e_results, baseline_results)
    
    return {
        "ctd_e_results": ctd_e_results,
        "baseline_results": baseline_results,
        "comparison_metrics": comparison_results
    }

def experiment2_retrieval_refinement():
    """Experiment 2: Retrieval-Augmented Node Refinement."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Retrieval-Augmented Node Refinement")
    print("="*60)
    
    corpus = create_external_corpus()
    print(f"Created corpus with {len(corpus)} documents")
    
    test_nodes = [
        {"criterion": "Evaluation for clarity", "explanation": "The sample text struggles with precise word choice."},
        {"criterion": "Evaluation for coherence", "explanation": "Ideas need better connection and flow."},
        {"criterion": "Evaluation for style", "explanation": "Writing style requires improvement."}
    ]
    
    ix = create_corpus_index(corpus)
    print("Created Whoosh index for corpus")
    
    refined_nodes = []
    original_lengths = []
    refined_lengths = []
    
    for i, node in enumerate(test_nodes):
        print(f"Processing node {i+1}/{len(test_nodes)}: {node['criterion']}")
        
        original_length = len(node.get("explanation", ""))
        original_lengths.append(original_length)
        
        refined_node = refine_node_with_retrieval(node, ix)
        refined_length = len(refined_node.get("explanation", ""))
        refined_lengths.append(refined_length)
        
        refined_nodes.append(refined_node)
        
        print(f"  Original length: {original_length}")
        print(f"  Refined length: {refined_length}")
        print(f"  Improvement: {refined_length - original_length}")
    
    plot_retrieval_refinement(original_lengths, refined_lengths)
    
    import shutil
    if os.path.exists("indexdir"):
        shutil.rmtree("indexdir")
    
    return {
        "original_nodes": test_nodes,
        "refined_nodes": refined_nodes,
        "length_improvements": [r - o for r, o in zip(refined_lengths, original_lengths)]
    }

def experiment3_iterative_refinement():
    """Experiment 3: Iterative Adversarial Refinement."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Iterative Adversarial Refinement")
    print("="*60)
    
    samples = [
        "The narrative is cohesive and the language is engaging.",
        "The report is informative and provides clear insights into the topic."
    ]
    
    print(f"Processing {len(samples)} samples through iterative refinement...")
    
    results_df = evaluate_iterative_refinement(samples, iterations=5)
    
    print(f"Generated {len(results_df)} evaluation records")
    print("\nIterative Refinement Results:")
    print(results_df.groupby('sample_id').agg({
        'score': ['first', 'last', 'mean'],
        'tree_depth': 'mean',
        'curriculum_weight': 'last'
    }).round(3))
    
    metrics = compute_evaluation_metrics(results_df)
    print(f"\nSummary Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    plot_iterative_refinement(results_df)
    
    return {
        "results_dataframe": results_df,
        "summary_metrics": metrics
    }

def plot_tree_comparison(samples, ctd_e_results, baseline_results):
    """Plot comparison between CTD-E trees and baseline criteria."""
    sample_ids = list(range(len(samples)))
    ctd_e_nodes = [1 + len(tree["subcriteria"]) for tree in ctd_e_results]
    baseline_nodes = [len(baseline) for baseline in baseline_results]
    
    x = np.arange(len(sample_ids))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, ctd_e_nodes, width, label="CTD-E Tree Nodes", color='skyblue')
    ax.bar(x + width/2, baseline_nodes, width, label="Baseline Criteria", color='lightcoral')
    
    ax.set_ylabel("Number of Nodes / Criteria")
    ax.set_title("Evaluation Tree Structure Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Sample {i+1}" for i in sample_ids])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_filename = ".research/iteration1/images/evaluation_tree_comparison.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches="tight")
    print(f"Saved tree comparison plot as {pdf_filename}")
    plt.close()

def plot_retrieval_refinement(original_lengths, refined_lengths):
    """Plot retrieval refinement results."""
    labels = ["Original", "Refined"]
    avg_lengths = [np.mean(original_lengths), np.mean(refined_lengths)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(labels, avg_lengths, color=["blue", "green"], alpha=0.7)
    ax1.set_ylabel("Average Length of Explanation (characters)")
    ax1.set_title("Before and After Retrieval Refinement")
    ax1.grid(True, alpha=0.3)
    
    improvements = [r - o for r, o in zip(refined_lengths, original_lengths)]
    ax2.bar(range(len(improvements)), improvements, color="orange", alpha=0.7)
    ax2.set_xlabel("Node Index")
    ax2.set_ylabel("Length Improvement (characters)")
    ax2.set_title("Individual Node Improvements")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_filename = ".research/iteration1/images/retrieval_refinement.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches="tight")
    print(f"Saved retrieval refinement plot as {pdf_filename}")
    plt.close()

def plot_iterative_refinement(df):
    """Plot iterative refinement results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for sample_id in df["sample_id"].unique():
        data = df[df["sample_id"] == sample_id]
        ax1.plot(data["iteration"], data["score"], marker="o", label=f"Sample {sample_id+1}")
    ax1.set_title("Score Evolution Over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Evaluation Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for sample_id in df["sample_id"].unique():
        data = df[df["sample_id"] == sample_id]
        ax2.plot(data["iteration"], data["tree_depth"], marker="s", label=f"Sample {sample_id+1}")
    ax2.set_title("Tree Depth Evolution Over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Tree Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for sample_id in df["sample_id"].unique():
        data = df[df["sample_id"] == sample_id]
        ax3.plot(data["iteration"], data["curriculum_weight"], marker="^", label=f"Sample {sample_id+1}")
    ax3.set_title("Curriculum Weight Evolution")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Curriculum Weight")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.hist(df["score"], bins=15, alpha=0.7, color="purple")
    ax4.set_title("Score Distribution")
    ax4.set_xlabel("Evaluation Score")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_filename = ".research/iteration1/images/iterative_refinement_comprehensive.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches="tight")
    print(f"Saved comprehensive iterative refinement plot as {pdf_filename}")
    plt.close()

def save_experiment_results(exp1_results, exp2_results, exp3_results):
    """Save all experiment results to files."""
    print("\nSaving experiment results...")
    
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "experiment1_summary": {
            "avg_ctd_e_nodes": exp1_results["comparison_metrics"]["avg_ctd_e_nodes"],
            "avg_baseline_nodes": exp1_results["comparison_metrics"]["avg_baseline_nodes"],
            "improvement_ratio": exp1_results["comparison_metrics"]["improvement_ratio"]
        },
        "experiment2_summary": {
            "total_nodes_processed": len(exp2_results["original_nodes"]),
            "avg_length_improvement": np.mean(exp2_results["length_improvements"])
        },
        "experiment3_summary": exp3_results["summary_metrics"]
    }
    
    with open(".research/iteration1/experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    exp3_results["results_dataframe"].to_csv(".research/iteration1/iterative_refinement_data.csv", index=False)
    
    print("Results saved to .research/iteration1/")

def set_status_stopped():
    """Set the status_enum to 'stopped' as required."""
    status_config = {
        "status_enum": "stopped",
        "completion_timestamp": datetime.now().isoformat(),
        "experiments_completed": ["dynamic_tree", "retrieval_refinement", "iterative_refinement"]
    }
    
    with open("config/experiment_status.json", "w") as f:
        json.dump(status_config, f, indent=2)
    
    print("Status set to 'stopped' in config/experiment_status.json")

def main():
    """Main function to orchestrate all CTD-E experiments."""
    print("Starting CTD-E (Critic Tree-Guided Dynamic Evaluation) Experiments")
    print("="*80)
    
    try:
        setup_experiment_environment()
        
        samples = prepare_sample_texts()
        corpus = create_external_corpus()
        save_preprocessed_data(samples, corpus)
        
        print("\nRunning all three CTD-E experiments...")
        
        exp1_results = experiment1_dynamic_tree()
        exp2_results = experiment2_retrieval_refinement()
        exp3_results = experiment3_iterative_refinement()
        
        save_experiment_results(exp1_results, exp2_results, exp3_results)
        
        set_status_stopped()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Results saved in .research/iteration1/")
        print("Plots saved as high-quality PDFs in .research/iteration1/images/")
        print("Status set to 'stopped'")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main():
    """Quick test function for main orchestration."""
    print("Running main orchestration test...")
    
    try:
        setup_experiment_environment()
        
        from preprocess import test_preprocess
        from train import test_train
        from evaluate import test_evaluate
        
        if not test_preprocess():
            return False
        if not test_train():
            return False
        if not test_evaluate():
            return False
        
        print("Main orchestration test completed successfully.\n")
        return True
        
    except Exception as e:
        print(f"Main orchestration test failed: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_main()
    else:
        main()
