#!/usr/bin/env python
"""
Training module for CTD-E (Critic Tree-Guided Dynamic Evaluation).
Implements the CTDEvaluator class with transformer model initialization and dynamic critic tree construction.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import Dict, List, Any, Optional

def create_tree_node(criterion: str, explanation: Optional[str] = None) -> Dict[str, Any]:
    """Create a tree node for the critic tree structure."""
    return {
        "criterion": criterion,
        "explanation": explanation,
        "subcriteria": []
    }

class CTDEvaluator:
    """Critic Tree-Guided Dynamic Evaluator using transformer models."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """Initialize the CTD evaluator with a transformer model."""
        print(f"Initializing CTD-E Evaluator with model: {model_name}")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_reasoning(self, prompt: str, max_length: int = 50) -> str:
        """Generate reasoning text using the transformer model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
        
        except Exception as e:
            print(f"Error in generate_reasoning: {e}")
            return f"Generated reasoning for: {prompt[:50]}..."
    
    def generate_critic_tree(self, sample_text: str) -> Dict[str, Any]:
        """Generate a hierarchical critic tree for evaluation."""
        sample_text = sample_text[:200] if len(sample_text) > 200 else sample_text
        
        base_prompt = f"Analyze this writing sample and provide an overall quality criterion: {sample_text}"
        high_level = self.generate_reasoning(base_prompt, max_length=30)
        
        if not high_level:
            high_level = "Overall writing quality assessment"
        
        tree = create_tree_node(high_level)
        
        aspects = ["clarity", "coherence", "style"]
        
        for aspect in aspects:
            aspect_prompt = f"For the text '{sample_text[:100]}...', evaluate {aspect}:"
            sub_reasoning = self.generate_reasoning(aspect_prompt, max_length=25)
            
            if not sub_reasoning:
                sub_reasoning = f"Evaluation of {aspect} in the given text"
            
            sub_node = create_tree_node(f"Evaluation for {aspect}", sub_reasoning)
            tree["subcriteria"].append(sub_node)
        
        return tree
    
    def save_model_state(self, save_dir: str = "models") -> None:
        """Save the current model state."""
        os.makedirs(save_dir, exist_ok=True)
        
        model_info = {
            "model_name": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "gpt2",
            "device": str(self.device),
            "tokenizer_vocab_size": len(self.tokenizer)
        }
        
        with open(os.path.join(save_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model state saved to {save_dir}/")

def baseline_evaluator(sample_text: str) -> List[Dict[str, str]]:
    """Baseline evaluator that outputs flat criteria without hierarchical refinement."""
    criteria = [
        {
            "criterion": "Overall quality",
            "explanation": f"Text is evaluated for general quality: {sample_text[:50]}..."
        },
        {
            "criterion": "Clarity",
            "explanation": "Look at sentence structure and readability."
        },
        {
            "criterion": "Style",
            "explanation": "Evaluate tone and writing style."
        }
    ]
    return criteria

def train_ctd_evaluator(samples: List[str], model_name: str = "gpt2") -> CTDEvaluator:
    """Train/initialize the CTD evaluator with sample data."""
    print("Training CTD-E evaluator...")
    
    evaluator = CTDEvaluator(model_name=model_name)
    
    for i, sample in enumerate(samples[:3]):
        print(f"Processing training sample {i+1}/{min(3, len(samples))}")
        tree = evaluator.generate_critic_tree(sample)
        print(f"Generated tree with {len(tree['subcriteria'])} subcriteria")
    
    evaluator.save_model_state()
    
    print("CTD-E evaluator training completed.")
    return evaluator

def test_train():
    """Quick test function for training module."""
    print("Running training test...")
    
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "In an increasingly globalized world, decisions shape international landscapes."
    ]
    
    try:
        evaluator = CTDEvaluator()
        
        for i, sample in enumerate(samples):
            print(f"\nProcessing sample {i+1}: {sample}")
            tree = evaluator.generate_critic_tree(sample)
            print(f"CTD-E Tree: {tree['criterion']}")
            print(f"Subcriteria count: {len(tree['subcriteria'])}")
            
            baseline = baseline_evaluator(sample)
            print(f"Baseline criteria count: {len(baseline)}")
        
        print("Training test completed successfully.\n")
        return True
        
    except Exception as e:
        print(f"Training test failed: {e}")
        return False

if __name__ == '__main__':
    test_train()
