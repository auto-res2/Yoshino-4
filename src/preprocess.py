#!/usr/bin/env python
"""
Data preprocessing for CTD-E experiments.
Handles sample text preparation and corpus creation for retrieval-augmented experiments.
"""

import os
import json
from typing import List, Dict, Any

def prepare_sample_texts() -> List[str]:
    """Prepare sample writing texts for evaluation experiments."""
    samples = [
        "The quick brown fox jumps over the lazy dog. This sentence demonstrates basic clarity and structure.",
        "In an increasingly globalized world, decisions shape international landscapes with far-reaching consequences.",
        "The narrative is cohesive and the language is engaging, providing readers with clear insights.",
        "The report is informative and provides clear insights into the topic under investigation.",
        "Academic writing requires precision, clarity, and logical flow to effectively communicate complex ideas.",
        "Effective communication involves understanding your audience and adapting your message accordingly."
    ]
    return samples

def create_external_corpus() -> List[str]:
    """Create external corpus for retrieval-augmented experiments."""
    corpus = [
        "A good writing style should be clear, engaging, and free of grammatical errors.",
        "Clarity in writing involves precise word choice and logical sentence structure.",
        "Coherence is achieved when ideas are well connected and flow naturally.",
        "Academic writing requires formal tone and evidence-based arguments.",
        "Effective paragraphs have clear topic sentences and supporting details.",
        "Transitions between ideas help maintain logical flow in writing.",
        "Conciseness eliminates unnecessary words while preserving meaning.",
        "Active voice generally creates more direct and engaging prose.",
        "Proper citation and referencing are essential in academic contexts.",
        "Revision and editing are crucial steps in the writing process."
    ]
    return corpus

def validate_sample_data(samples: List[str]) -> bool:
    """Validate that sample data meets basic requirements."""
    if not samples:
        return False
    
    for sample in samples:
        if not isinstance(sample, str) or len(sample.strip()) == 0:
            return False
        if len(sample.split()) < 3:  # Minimum word count
            return False
    
    return True

def save_preprocessed_data(samples: List[str], corpus: List[str], output_dir: str = "data") -> None:
    """Save preprocessed data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "sample_texts.json"), "w") as f:
        json.dump(samples, f, indent=2)
    
    with open(os.path.join(output_dir, "external_corpus.json"), "w") as f:
        json.dump(corpus, f, indent=2)
    
    print(f"Preprocessed data saved to {output_dir}/")

def load_preprocessed_data(data_dir: str = "data") -> tuple:
    """Load preprocessed data from files."""
    samples_path = os.path.join(data_dir, "sample_texts.json")
    corpus_path = os.path.join(data_dir, "external_corpus.json")
    
    samples = []
    corpus = []
    
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            samples = json.load(f)
    
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            corpus = json.load(f)
    
    return samples, corpus

def test_preprocess():
    """Quick test function for preprocessing."""
    print("Running preprocessing test...")
    
    samples = prepare_sample_texts()
    corpus = create_external_corpus()
    
    print(f"Prepared {len(samples)} sample texts")
    print(f"Created corpus with {len(corpus)} documents")
    
    if validate_sample_data(samples):
        print("Sample data validation: PASSED")
    else:
        print("Sample data validation: FAILED")
        return False
    
    save_preprocessed_data(samples, corpus)
    loaded_samples, loaded_corpus = load_preprocessed_data()
    
    if loaded_samples == samples and loaded_corpus == corpus:
        print("Save/load functionality: PASSED")
    else:
        print("Save/load functionality: FAILED")
        return False
    
    print("Preprocessing test completed successfully.\n")
    return True

if __name__ == '__main__':
    test_preprocess()
