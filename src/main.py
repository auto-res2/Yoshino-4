#!/usr/bin/env python3
"""
Experiment Script for ECGA Innovations:
1. Dynamic Online Knowledge Expansion (Experiment 1)
2. Executable Code Feedback Loop with Self-Debugging (Experiment 2)
3. Adaptive Task Complexity Strategy with Hierarchical, Tree-Based Exploration (Experiment 3)

Each experiment prints detailed output and saves plots as .pdf files using the proper filename format.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import fetch_arxiv_papers, update_knowledge_base, generate_buggy_code
from train import train_linear_regression
from evaluate import debug_loop, adaptive_strategy, tasks

sns.set(style="whitegrid")

def plot_loss(loss_history, title, filename):
    """
    Plot and save the training loss curve as a PDF.
    """
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, marker='o', label='Training Loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {filename}")

def experiment1():
    """
    Experiment 1: Dynamic Online Knowledge Expansion.
    This experiment fetches recent expert data, updates a simulated knowledge base,
    computes a freshness metric, and uses the updated KB to influence a downstream regression task.
    """
    print("\n===== Running Experiment 1: Dynamic Online Knowledge Expansion =====")
    
    papers = fetch_arxiv_papers(max_results=10)
    
    expert_kb = update_knowledge_base(papers)
    
    timestamps = [datetime.strptime(p["published"], "%Y-%m-%d").timestamp() for p in papers]
    freshness = sum(timestamps) / len(timestamps)
    print("Freshness Metric (average timestamp):", freshness)
    
    loss_hist_static, final_loss_static = train_linear_regression(static=True)
    loss_hist_dynamic, final_loss_dynamic = train_linear_regression(static=False)
    
    print(f"Final Loss (Static KB): {final_loss_static:.4f}")
    print(f"Final Loss (Dynamic KB): {final_loss_dynamic:.4f}")
    
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist_static, marker='o', label="Static KB")
    plt.plot(loss_hist_dynamic, marker='s', label="Dynamic KB")
    plt.title("Training Loss Comparison: Static vs. Dynamic Knowledge Base")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plot_filename = "../.research/iteration1/images/training_loss_experiment1.pdf"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_filename}")
    print("Experiment 1 completed.\n")

def experiment2():
    """
    Experiment 2: Executable Code Feedback Loop with Self-Debugging.
    This experiment simulates an iterative debugging cycle until a simple PyTorch regression task code runs error-free.
    """
    print("\n===== Running Experiment 2: Executable Code Feedback Loop with Self-Debugging =====")
    buggy_code = generate_buggy_code()
    fixed_code, iterations = debug_loop(buggy_code, max_iterations=3)
    print("\nFinal fixed code after", iterations, "iterations:")
    print(fixed_code)
    print("Experiment 2 completed.\n")

def experiment3():
    """
    Experiment 3: Adaptive Task Complexity Strategy with Hierarchical, Tree-Based Exploration.
    Simulate multiple tasks with varying complexity and record solution efficiency and performance.
    """
    print("\n===== Running Experiment 3: Adaptive Task Complexity Strategy =====")
    solution_tree = {}
    for task in tasks:
        node = adaptive_strategy(task)
        solution_tree[task] = node
    
    total_iterations = sum(node.iterations for node in solution_tree.values())
    print("Total iterations across tasks:", total_iterations)
    
    task_names = list(solution_tree.keys())
    performance_metrics = [solution_tree[t].metric for t in task_names]
    iterations_list = [solution_tree[t].iterations for t in task_names]
    
    plt.figure(figsize=(7,4))
    sns.barplot(x=task_names, y=performance_metrics, palette="viridis")
    plt.title("Performance Metric per Task")
    plt.xlabel("Task")
    plt.ylabel("Performance Metric")
    performance_filename = "../.research/iteration1/images/accuracy_adaptive_strategy.pdf"
    plt.tight_layout()
    plt.savefig(performance_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved performance metric plot to {performance_filename}")
    
    plt.figure(figsize=(7,4))
    sns.barplot(x=task_names, y=iterations_list, palette="rocket")
    plt.title("Iterations Needed per Task")
    plt.xlabel("Task")
    plt.ylabel("Iterations")
    iterations_filename = "../.research/iteration1/images/iterations_adaptive_strategy.pdf"
    plt.tight_layout()
    plt.savefig(iterations_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved iterations plot to {iterations_filename}")
    
    print("Experiment 3 completed.\n")

def test_experiments():
    """
    Run each experiment in a quick test mode.
    In a production run, these might run longer, but the test finishes immediately.
    """
    print("\n========= Starting Test of All Experiments =========")
    experiment1()
    experiment2()
    experiment3()
    print("All experiments executed successfully. Test complete.\n")
    
if __name__ == '__main__':
    test_experiments()
