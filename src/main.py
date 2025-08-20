"""
AutoMind++ Framework - Main Experimental Script
Orchestrates all three experiments: Multi-Modal Fusion, Meta-Adaptation, and Privacy-Preserving KD.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from preprocess import (
    generate_multimodal_data, 
    preprocess_text_embeddings, 
    preprocess_statistical_features,
    detect_anomalies,
    generate_sensitive_data
)
from train import (
    train_multimodal_fusion,
    train_with_meta_adaptation, 
    train_privacy_preserving_kd,
    MultiModalFusionModule
)
from evaluate import (
    evaluate_multimodal_fusion,
    evaluate_meta_adaptation,
    evaluate_privacy_preserving_kd,
    generate_comprehensive_report
)

import torchvision.models as models


def setup_environment():
    """Setup the experimental environment and check GPU availability."""
    print("=" * 80)
    print("AutoMind++ Framework - Experimental Setup")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        
        if "T4" in gpu_name or gpu_memory >= 15:
            print("✓ GPU is compatible with Tesla T4 requirements (16GB VRAM)")
        else:
            print(f"⚠ Warning: GPU may not meet Tesla T4 specs, but proceeding...")
    else:
        device = torch.device('cpu')
        print("⚠ No GPU available, using CPU (this may be slow)")
    
    output_dir = ".research/iteration1/images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    print("✓ Random seeds set for reproducibility")
    
    return device, output_dir


def run_experiment_1_multimodal_fusion(device, output_dir, num_samples=50):
    """
    Experiment 1: Multi-Modal Interactive Feedback Fusion
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Multi-Modal Interactive Feedback Fusion")
    print("=" * 60)
    
    start_time = time.time()
    
    print("Generating multi-modal synthetic data...")
    data = generate_multimodal_data(num_samples=num_samples, seed=42)
    
    print("Preprocessing text embeddings...")
    text_embeddings = preprocess_text_embeddings(data['texts'])
    
    print("Extracting image features using CNN...")
    cnn = models.resnet18(pretrained=False)
    cnn.fc = torch.nn.Identity()  # Remove final classification layer
    cnn.eval()
    
    plot_features = []
    with torch.no_grad():
        for img in data['plot_images']:
            img_batch = img.unsqueeze(0)
            feat = cnn(img_batch).squeeze(0)
            plot_features.append(feat)
    plot_features = torch.stack(plot_features)
    
    print("Preprocessing statistical features...")
    stat_vectors, scaler = preprocess_statistical_features(data['stat_summaries'])
    
    print("Detecting anomalies in statistical data...")
    anomaly_preds = detect_anomalies(stat_vectors)
    num_anomalies = np.sum(anomaly_preds == -1)
    print(f"Detected {num_anomalies} anomalies out of {num_samples} samples")
    
    print("Training multi-modal fusion model...")
    model, training_history = train_multimodal_fusion(
        text_embeddings, plot_features, stat_vectors, 
        data['noise_flags'], epochs=5, lr=0.001
    )
    
    labels = torch.randint(0, 2, (num_samples,))
    
    print("Evaluating multi-modal fusion performance...")
    exp1_results = evaluate_multimodal_fusion(
        model, text_embeddings, plot_features, stat_vectors,
        labels, data['noise_flags'], save_dir=output_dir
    )
    
    duration = time.time() - start_time
    print(f"\n✓ Experiment 1 completed in {duration:.2f} seconds")
    print(f"✓ Multi-modal fusion accuracy: {exp1_results['accuracy']:.3f}")
    print(f"✓ Mean modality weights - Text: {exp1_results['mean_text_weight']:.3f}, "
          f"Plot: {exp1_results['mean_plot_weight']:.3f}, "
          f"Stats: {exp1_results['mean_stat_weight']:.3f}")
    
    return exp1_results


def run_experiment_2_meta_adaptation(device, output_dir, num_iterations=5):
    """
    Experiment 2: Human-Guided Rapid Meta-Adaptation
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Human-Guided Rapid Meta-Adaptation")
    print("=" * 60)
    
    start_time = time.time()
    
    print("Training classifier with meta-adaptation...")
    model, history = train_with_meta_adaptation(
        num_iterations=num_iterations, 
        epochs_per_iter=3, 
        lr=0.01
    )
    
    print("Evaluating meta-adaptation performance...")
    exp2_results = evaluate_meta_adaptation(history, save_dir=output_dir)
    
    duration = time.time() - start_time
    print(f"\n✓ Experiment 2 completed in {duration:.2f} seconds")
    print(f"✓ Final adaptation accuracy: {exp2_results['final_accuracy']:.3f}")
    print(f"✓ Accuracy improvement: {exp2_results['accuracy_improvement']:.3f}")
    print(f"✓ Adaptation stability: {exp2_results['adaptation_stability']:.3f}")
    
    return exp2_results


def run_experiment_3_privacy_preserving_kd(device, output_dir, epochs=3):
    """
    Experiment 3: Dual-Stage Privacy-Preserving Knowledge Distillation
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Privacy-Preserving Knowledge Distillation")
    print("=" * 60)
    
    start_time = time.time()
    
    print("Training privacy-preserving knowledge distillation...")
    teacher, student, history = train_privacy_preserving_kd(
        epochs=epochs, lr=0.1, noise_multiplier=1.0
    )
    
    print("Generating test data for evaluation...")
    test_data = generate_sensitive_data(n_samples=200, input_dim=10, seed=123)
    
    print("Evaluating privacy-preserving knowledge distillation...")
    exp3_results = evaluate_privacy_preserving_kd(
        teacher, student, test_data, history, save_dir=output_dir
    )
    
    duration = time.time() - start_time
    print(f"\n✓ Experiment 3 completed in {duration:.2f} seconds")
    print(f"✓ Teacher accuracy: {exp3_results['teacher_accuracy']:.3f}")
    print(f"✓ Student accuracy: {exp3_results['student_accuracy']:.3f}")
    print(f"✓ Knowledge retention: {exp3_results['knowledge_retention']:.3f}")
    if exp3_results['privacy_budget_used'] is not None:
        print(f"✓ Privacy budget used: {exp3_results['privacy_budget_used']:.3f}")
    
    return exp3_results


def main():
    """Main experimental pipeline for AutoMind++ framework."""
    
    device, output_dir = setup_environment()
    
    experiment_start = time.time()
    
    try:
        print("\nStarting AutoMind++ experimental pipeline...")
        
        exp1_results = run_experiment_1_multimodal_fusion(device, output_dir, num_samples=50)
        
        exp2_results = run_experiment_2_meta_adaptation(device, output_dir, num_iterations=5)
        
        exp3_results = run_experiment_3_privacy_preserving_kd(device, output_dir, epochs=3)
        
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        comprehensive_report = generate_comprehensive_report(
            exp1_results, exp2_results, exp3_results, save_dir=output_dir
        )
        
        total_duration = time.time() - experiment_start
        print("\n" + "=" * 80)
        print("AUTOMIND++ EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"Total Execution Time: {total_duration:.2f} seconds")
        print(f"Device Used: {device}")
        print(f"Output Directory: {output_dir}")
        
        print("\nExperiment Results:")
        print(f"  1. Multi-Modal Fusion Accuracy: {exp1_results['accuracy']:.3f}")
        print(f"  2. Meta-Adaptation Final Accuracy: {exp2_results['final_accuracy']:.3f}")
        print(f"  3. Privacy-Preserving Student Accuracy: {exp3_results['student_accuracy']:.3f}")
        
        print(f"\nOverall Framework Performance:")
        print(f"  Average Accuracy: {comprehensive_report['overall_performance']['average_accuracy']:.3f}")
        print(f"  Best Experiment: {comprehensive_report['overall_performance']['best_performing_experiment']}")
        print(f"  Framework Robustness: {comprehensive_report['overall_performance']['framework_robustness_score']:.3f}")
        
        print(f"\nGenerated Output Files:")
        pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
        for pdf_file in sorted(pdf_files):
            print(f"  - {pdf_file}")
        
        print("\n✓ All experiments completed successfully!")
        print("✓ All plots saved as high-quality PDFs for academic use")
        print("✓ AutoMind++ framework evaluation complete")
        
        status_enum = "stopped"
        print(f"\n✓ Status set to: {status_enum}")
        
        return comprehensive_report
        
    except Exception as e:
        print(f"\n❌ Error during experiment execution: {str(e)}")
        print(f"❌ Experiment failed after {time.time() - experiment_start:.2f} seconds")
        raise e


if __name__ == "__main__":
    print(f"AutoMind++ Framework started at {datetime.now()}")
    results = main()
    print(f"AutoMind++ Framework completed at {datetime.now()}")
