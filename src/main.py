#!/usr/bin/env python
"""
Experiment Suite for evaluating SB-FSE versus a baseline first–order BFN.
This code implements three experiments:

Experiment 1: Convergence Efficiency with Sequential Conditioning on a 2D Gaussian Mixture.
Experiment 2: Accuracy of the Integrated Likelihood Approximation via Importance Weighting on 1D Gaussian.
Experiment 3: Scalability and Robustness on Noisy, Increasingly Complex Data using MNIST.

All plots are saved as PDF files.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from .train import train_score_net_2d, train_score_net_1d, train_score_net_img
    from .evaluate import (baseline_sampler, sb_fse_sampler, evaluate_convergence, 
                          sbfse_likelihood_estimate, baseline_sampler_img, sbfse_sampler_img,
                          run_scalability_experiment)
    from .preprocess import generate_true_sample, get_mnist_dataloader, true_likelihood
except ImportError:
    from train import train_score_net_2d, train_score_net_1d, train_score_net_img
    from evaluate import (baseline_sampler, sb_fse_sampler, evaluate_convergence, 
                         sbfse_likelihood_estimate, baseline_sampler_img, sbfse_sampler_img,
                         run_scalability_experiment)
    from preprocess import generate_true_sample, get_mnist_dataloader, true_likelihood

def experiment1_convergence(device='cpu'):
    print("Starting Experiment 1: Convergence Efficiency with Sequential Conditioning")
    score_net = train_score_net_2d(device=device)
    
    mean_error_baseline, errors_baseline = evaluate_convergence(baseline_sampler, score_net, generate_true_sample, steps=50, iterations=100, device=device)
    mean_error_sbfse, errors_sbfse = evaluate_convergence(sb_fse_sampler, score_net, generate_true_sample, steps=50, iterations=100, device=device)
    
    plt.figure(figsize=(8, 5))
    plt.plot(errors_baseline, label="Baseline")
    plt.plot(errors_sbfse, label="SB-FSE")
    plt.xlabel("Iteration index")
    plt.ylabel("Mean Squared Error")
    plt.title("Convergence Comparison: Baseline vs SB-FSE")
    plt.legend()
    filename = ".research/iteration1/images/training_loss_baseline_vs_sbfse.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Experiment 1 plot saved as {filename}")
    plt.close()

def experiment2_likelihood(device='cpu'):
    print("Starting Experiment 2: Likelihood Approximation via Importance Weighting")
    net_1d = train_score_net_1d(device=device)
    
    x_init = 2.5
    approx_likelihood = sbfse_likelihood_estimate(net_1d, x_init, steps=30, device=device)
    true_like = true_likelihood(x_init, mean=0.0, variance=1.0)
    print(f"SB-FSE Approximated Integrated Likelihood: {approx_likelihood:.5f}")
    print(f"Analytically computed True Likelihood: {true_like:.5f}")
    
    plt.figure(figsize=(6, 5))
    methods = ['SB-FSE Approx.', 'True Likelihood']
    values = [approx_likelihood, true_like]
    plt.bar(methods, values, color=['blue', 'green'])
    plt.ylabel("Likelihood Value")
    plt.title("Integrated Likelihood: SB-FSE vs True")
    filename = ".research/iteration1/images/likelihood_approximation.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Experiment 2 plot saved as {filename}")
    plt.close()

def experiment3_scalability(device='cpu'):
    print("Starting Experiment 3: Scalability and Robustness on MNIST with Added Noise")
    data_loader = get_mnist_dataloader(batch_size=16, shuffle=True)
    score_net_img = train_score_net_img(data_loader, device=device)
    
    print("Running baseline image sampler evaluation...")
    baseline_metrics, baseline_times = run_scalability_experiment(baseline_sampler_img, score_net_img, data_loader, device=device)
    print("Running SB-FSE image sampler evaluation...")
    sbfse_metrics, sbfse_times = run_scalability_experiment(sbfse_sampler_img, score_net_img, data_loader, device=device)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline_metrics, label="Baseline")
    plt.plot(sbfse_metrics, label="SB-FSE")
    plt.xlabel("Batch index")
    plt.ylabel("Quality Metric (MSE)")
    plt.title("Sample Quality under Added Noise")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(baseline_times, label="Baseline")
    plt.plot(sbfse_times, label="SB-FSE")
    plt.xlabel("Batch index")
    plt.ylabel("Time per Batch (sec)")
    plt.title("Computational Cost")
    plt.legend()
    
    plt.tight_layout()
    filename = ".research/iteration1/images/scalability_metrics.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Experiment 3 plot saved as {filename}")
    plt.close()

def test_experiments():
    """
    This test function runs each experiment with reduced iterations/epochs so that it finishes quickly.
    This helps in verifying that all code executes correctly.
    """
    print("=== Starting quick tests for all experiments ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\nRunning test for Experiment 1 (convergence)...")
    try:
        from .train import ScoreNet
    except ImportError:
        from train import ScoreNet
    input_dim, hidden_dim, output_dim = 2, 16, 2
    score_net = ScoreNet(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)
    for epoch in range(10):
        x_true = generate_true_sample(n_samples=16).to(device)
        optimizer.zero_grad()
        loss = torch.mean(score_net(x_true)**2)
        loss.backward()
        optimizer.step()
    _ , _ = evaluate_convergence(baseline_sampler, score_net, generate_true_sample, steps=10, iterations=5, device=device)
    print("Experiment 1 test completed.")
    
    print("\nRunning test for Experiment 2 (likelihood approximation)...")
    try:
        from .train import ScoreNet1D
    except ImportError:
        from train import ScoreNet1D
    net_1d = ScoreNet1D().to(device)
    optimizer = torch.optim.Adam(net_1d.parameters(), lr=1e-3)
    for epoch in range(10):
        x_train = torch.randn((16, 1), device=device)
        target = -x_train
        optimizer.zero_grad()
        loss = torch.mean((net_1d(x_train) - target)**2)
        loss.backward()
        optimizer.step()
    approx_like = sbfse_likelihood_estimate(net_1d, 2.5, steps=5, device=device)
    true_like = true_likelihood(2.5, 0.0, 1.0)
    print(f"Test Likelihood: SB-FSE approximated = {approx_like:.5f}, True = {true_like:.5f}")
    print("Experiment 2 test completed.")
    
    print("\nRunning test for Experiment 3 (scalability)...")
    try:
        from .train import ScoreNet
    except ImportError:
        from train import ScoreNet
    data_loader = get_mnist_dataloader(batch_size=4, shuffle=True)
    score_net_img = ScoreNet(28*28, 64, 28*28).to(device)
    optimizer = torch.optim.Adam(score_net_img.parameters(), lr=1e-3)
    for epoch in range(1):
        for images, _ in data_loader:
            images = images.to(device)
            inputs = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = score_net_img(inputs)
            loss = torch.mean((outputs - inputs)**2)
            loss.backward()
            optimizer.step()
    bm, bt = run_scalability_experiment(baseline_sampler_img, score_net_img, data_loader, device=device)
    sm, st = run_scalability_experiment(sbfse_sampler_img, score_net_img, data_loader, device=device)
    print("Experiment 3 test completed.")
    
    print("\n=== All quick tests completed successfully ===")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)
    
    experiment1_convergence(device=device)
    experiment2_likelihood(device=device)
    experiment3_scalability(device=device)
    
    test_experiments()
    
    print("All experiments executed. Check the PDF files for the plots and the console output for details.")
