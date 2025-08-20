import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import generate_streaming_data
from src.train import create_autoencoder_model
from src.evaluate import (
    run_pipeline, ResourceAllocationEnv, random_allocation_agent, 
    simulate_rl_agent, generate_explanation
)

def experiment_1():
    """Dynamic Domain Shift Anticipation Evaluation."""
    print('Running Experiment 1: Dynamic Domain Shift Anticipation Evaluation')
    
    data, labels = generate_streaming_data()
    autoencoder = create_autoencoder_model()
    
    perf_proactive, adapt_proactive = run_pipeline(data, proactive=True, model=autoencoder)
    perf_baseline, adapt_baseline = run_pipeline(data, proactive=False, model=autoencoder)

    plt.figure(figsize=(12, 6))
    plt.plot(perf_proactive, label='ARDSON with PDSA (Proactive)')
    plt.plot(perf_baseline, label='Baseline (Reactive Only)')
    plt.xlabel('Time Step')
    plt.ylabel('Simulated Performance Score')
    plt.title('Pipeline Performance under Domain Shift')
    plt.legend()
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    filename = '.research/iteration1/images/pipeline_performance_domain_shift.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f'Experiment 1 plot saved as {filename}')

def experiment_2():
    """Multi-Objective Resource Allocation via Reinforcement Learning."""
    print('Running Experiment 2: Multi-Objective Resource Allocation via Reinforcement Learning')
    
    env = ResourceAllocationEnv()
    baseline_reward = random_allocation_agent(env, episodes=1000)
    rl_reward = simulate_rl_agent(env, episodes=1000)
    
    print('Experiment 2 results:')
    print(f'  Random Baseline Average Reward: {baseline_reward:.2f}')
    print(f'  RL-based Allocation Average Reward: {rl_reward:.2f}')
    
    plt.figure(figsize=(8, 6))
    methods = ['Random Baseline', 'RL-based Allocation']
    rewards = [baseline_reward, rl_reward]
    plt.bar(methods, rewards, color=['red', 'blue'])
    plt.ylabel('Average Reward')
    plt.title('Resource Allocation Performance Comparison')
    
    filename = '.research/iteration1/images/resource_allocation_comparison.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f'Experiment 2 plot saved as {filename}')

def experiment_3():
    """Explainability-Centric Meta-Optimization Evaluation."""
    print('Running Experiment 3: Explainability-Centric Meta-Optimization Evaluation')
    
    adapt_events = ['domain_shift', 'resource_reallocation', 'curriculum_adjustment']
    expected_keywords = {
        'domain_shift': ['shift', 'distribution', 'semantic'],
        'resource_reallocation': ['resource', 'reallocating', 'budget'],
        'curriculum_adjustment': ['difficulty', 'scaling', 'knowledge']
    }
    
    for event in adapt_events:
        data_pattern = random.choice(['pattern_A', 'pattern_B', 'pattern_C'])
        explanation = generate_explanation(event, data_pattern)
        print(f'Event: {event}')
        print(f'Generated Explanation: {explanation}')
        matches = [kw for kw in expected_keywords[event] if kw in explanation]
        print(f'Matched Keywords: {matches}\n')
    
    counts = [len(expected_keywords[event]) for event in adapt_events]
    plt.figure(figsize=(8, 4))
    plt.bar(adapt_events, counts, color=['blue', 'green', 'red'])
    plt.xlabel('Adaptation Event')
    plt.ylabel('Expected Keyword Count')
    plt.title('Explanation Evaluation: Expected Keywords Count per Event')
    
    filename = '.research/iteration1/images/explanation_evaluation.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f'Experiment 3 plot saved as {filename}')

def main():
    """Main function to run all ARDSON experiments."""
    print('Starting ARDSON experiments...')
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    
    experiment_1()
    experiment_2()
    experiment_3()
    
    print('All experiments executed successfully.')
    print('Results saved to .research/iteration1/images/')

if __name__ == '__main__':
    main()
