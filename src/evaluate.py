import time
import numpy as np
import torch
try:
    from .preprocess import generate_true_sample, add_noise, true_likelihood
except ImportError:
    from preprocess import generate_true_sample, add_noise, true_likelihood

def baseline_sampler(net, steps=50, device='cpu'):
    """Start with a random initialization for 2D points."""
    x = torch.randn((1, 2), device=device)
    trajectory = [x.clone().cpu().numpy()]
    for step in range(steps):
        with torch.no_grad():
            noise_pred = x - net(x)
        x = x + 0.1 * noise_pred
        trajectory.append(x.clone().cpu().numpy())
    return x, trajectory

def sb_fse_sampler(net, steps=50, device='cpu'):
    """Start with a random initialization for 2D points."""
    x = torch.randn((1, 2), device=device)
    trajectory = [x.clone().cpu().numpy()]
    for step in range(steps):
        with torch.no_grad():
            score = net(x)
            uncertainty = torch.exp(-0.5 * torch.sum(score**2, dim=1, keepdim=True))
            correction = uncertainty * score
        x = x + 0.1 * correction
        trajectory.append(x.clone().cpu().numpy())
    return x, trajectory

def evaluate_convergence(sampler, net, true_mode_func, steps=50, iterations=100, device='cpu'):
    errors = []
    for i in range(iterations):
        x_final, _ = sampler(net, steps=steps, device=device)
        true_sample = true_mode_func().to(device)
        mse = torch.mean((x_final - true_sample)**2).item()
        errors.append(mse)
    mean_error = np.mean(errors)
    print(f"Evaluation complete over {iterations} iterations. Mean MSE error = {mean_error}")
    return mean_error, np.array(errors)

def sbfse_likelihood_estimate(net, x_init, steps=30, device='cpu'):
    """Start with an initial sample (1D, wrapped in a tensor of shape [1,1])"""
    x = torch.tensor([[x_init]], device=device, dtype=torch.float32)
    likelihood_accum = 0.0
    weight_accum = 0.0
    sigma_sq = 1.0
    for step in range(steps):
        with torch.no_grad():
            score = net(x)
            uncertainty = torch.abs(score)
            weight = torch.exp(-uncertainty)
        likelihood_accum += weight.item()
        weight_accum += 1.0
        x = x + 0.1 * weight * score
    integrated_likelihood = likelihood_accum / weight_accum
    return integrated_likelihood

def baseline_sampler_img(net, steps=20, device='cpu'):
    """For MNIST (flattened dimensions 28*28)"""
    x = torch.randn((1, 28*28), device=device)
    for step in range(steps):
        with torch.no_grad():
            noise_pred = x - net(x)
        x = x + 0.05 * noise_pred
    return x, None

def sbfse_sampler_img(net, steps=20, device='cpu'):
    x = torch.randn((1, 28*28), device=device)
    for step in range(steps):
        with torch.no_grad():
            score = net(x)
            uncertainty = torch.exp(-torch.abs(score))
            correction = uncertainty * score
        x = x + 0.05 * correction
    return x, None

def run_scalability_experiment(sampler_func, model, data_loader, device='cpu'):
    sample_metrics = []
    times_list = []
    batch_count = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        data_noisy = add_noise(data, noise_std=0.2)
        start_time = time.time()
        batch_samples = []
        for i in range(data_noisy.size(0)):
            sample, _ = sampler_func(model, steps=20, device=device)
            sample_img = sample.view(1, 1, 28, 28)
            batch_samples.append(sample_img)
        elapsed = time.time() - start_time
        times_list.append(elapsed)
        batch_samples_tensor = torch.cat(batch_samples, dim=0)
        mse_metric = torch.mean((data_noisy - batch_samples_tensor)**2).item()
        sample_metrics.append(mse_metric)
        batch_count += 1
        print(f"Batch {batch_idx}: MSE Metric = {mse_metric:.5f}, Time = {elapsed:.5f} sec")
        if batch_idx >= 10:
            break
    return sample_metrics, times_list
