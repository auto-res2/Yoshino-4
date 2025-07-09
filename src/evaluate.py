#!/usr/bin/env python
"""
Evaluation module for CTD-E experiments.
Implements retrieval-augmented node refinement and iterative adversarial refinement.
"""

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os
import shutil
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Tuple

def create_corpus_index(corpus: List[str], index_dir: str = "indexdir") -> Any:
    """Create a Whoosh index for the external corpus."""
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.mkdir(index_dir)
    
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    
    for idx, doc in enumerate(corpus):
        writer.add_document(id=str(idx), content=doc)
    
    writer.commit()
    return ix

def retrieve_guidance(ix: Any, query_text: str, top_n: int = 1) -> List[str]:
    """Retrieve top matching passages from the external corpus."""
    qp = QueryParser("content", schema=ix.schema)
    q = qp.parse(query_text)
    
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=top_n)
        retrieved = [hit["content"] for hit in results]
    
    return retrieved

def refine_node_with_retrieval(node: Dict[str, Any], corpus_index: Any) -> Dict[str, Any]:
    """Refine a CTD-E node using retrieval-augmented information."""
    query_text = node.get("explanation", node.get("criterion", ""))
    
    if not query_text:
        return node
    
    retrieved_info = retrieve_guidance(corpus_index, query_text, top_n=1)
    
    if retrieved_info:
        refined_explanation = node.get("explanation", "") + " Retrieved guidelines: " + "; ".join(retrieved_info)
        refined_node = node.copy()
        refined_node["explanation"] = refined_explanation
        return refined_node
    
    return node

def perturb_text(text: str) -> str:
    """Apply adversarial perturbation to text."""
    adversarial_fragments = [
        "However, this statement contradicts itself.",
        "Despite previous clarity, confusion ensues.",
        "The argument lacks coherent structure.",
        "This creates unnecessary ambiguity."
    ]
    
    words = text.split()
    if len(words) > 3:
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, random.choice(adversarial_fragments))
    
    return " ".join(words)

def ctd_evaluate_score(text: str, curriculum_weight: float = 1.0) -> Tuple[float, int]:
    """Evaluate text using CTD-E method, returning score and tree depth."""
    base_score = 8.0
    
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    length_penalty = max(0, (word_count - 50) * 0.01)  # Penalty for very long texts
    complexity_penalty = (word_count % 7) * 0.1  # Pseudo-random complexity penalty
    curriculum_adjustment = (1.0 / curriculum_weight) * complexity_penalty
    
    score = max(1.0, base_score - length_penalty - curriculum_adjustment)
    
    tree_depth = 2 if curriculum_adjustment > 0.5 else 1
    
    return score, tree_depth

def evaluate_tree_comparison(samples: List[str], ctd_e_results: List[Dict], baseline_results: List[List]) -> Dict[str, Any]:
    """Evaluate comparison between CTD-E trees and baseline criteria."""
    ctd_e_node_counts = []
    baseline_node_counts = []
    
    for ctd_tree, baseline in zip(ctd_e_results, baseline_results):
        ctd_count = 1 + len(ctd_tree.get("subcriteria", []))
        baseline_count = len(baseline)
        
        ctd_e_node_counts.append(ctd_count)
        baseline_node_counts.append(baseline_count)
    
    results = {
        "ctd_e_nodes": ctd_e_node_counts,
        "baseline_nodes": baseline_node_counts,
        "avg_ctd_e_nodes": np.mean(ctd_e_node_counts),
        "avg_baseline_nodes": np.mean(baseline_node_counts),
        "improvement_ratio": np.mean(ctd_e_node_counts) / np.mean(baseline_node_counts)
    }
    
    return results

def evaluate_retrieval_refinement(nodes: List[Dict], corpus: List[str]) -> Dict[str, Any]:
    """Evaluate the effectiveness of retrieval-augmented refinement."""
    ix = create_corpus_index(corpus)
    
    original_lengths = []
    refined_lengths = []
    
    for node in nodes:
        original_length = len(node.get("explanation", ""))
        original_lengths.append(original_length)
        
        refined_node = refine_node_with_retrieval(node, ix)
        refined_length = len(refined_node.get("explanation", ""))
        refined_lengths.append(refined_length)
    
    results = {
        "original_lengths": original_lengths,
        "refined_lengths": refined_lengths,
        "avg_improvement": np.mean(refined_lengths) - np.mean(original_lengths),
        "improvement_ratio": np.mean(refined_lengths) / np.mean(original_lengths) if np.mean(original_lengths) > 0 else 1.0
    }
    
    if os.path.exists("indexdir"):
        shutil.rmtree("indexdir")
    
    return results

def evaluate_iterative_refinement(samples: List[str], iterations: int = 5) -> pd.DataFrame:
    """Evaluate iterative adversarial refinement process."""
    results = []
    
    for sample_id, sample in enumerate(samples):
        current_sample = sample
        curriculum_weight = 1.0
        
        for iteration in range(1, iterations + 1):
            adversarial_sample = perturb_text(current_sample)
            score, tree_depth = ctd_evaluate_score(adversarial_sample, curriculum_weight)
            
            curriculum_weight += 0.2 * (10 - score)
            
            results.append({
                "sample_id": sample_id,
                "iteration": iteration,
                "score": score,
                "tree_depth": tree_depth,
                "curriculum_weight": curriculum_weight,
                "text_length": len(adversarial_sample)
            })
            
            current_sample = adversarial_sample
    
    return pd.DataFrame(results)

def compute_evaluation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary metrics from evaluation results."""
    metrics = {
        "avg_final_score": df[df["iteration"] == df["iteration"].max()]["score"].mean(),
        "score_improvement": df.groupby("sample_id")["score"].last().mean() - df.groupby("sample_id")["score"].first().mean(),
        "avg_tree_depth": df["tree_depth"].mean(),
        "curriculum_convergence": df.groupby("sample_id")["curriculum_weight"].last().mean()
    }
    
    return metrics

def test_evaluate():
    """Quick test function for evaluation module."""
    print("Running evaluation test...")
    
    samples = [
        "The narrative is cohesive and the language is engaging.",
        "The report is informative and provides clear insights into the topic."
    ]
    
    corpus = [
        "A good writing style should be clear, engaging, and free of grammatical errors.",
        "Clarity in writing involves precise word choice and logical sentence structure."
    ]
    
    try:
        test_nodes = [
            {"criterion": "Clarity", "explanation": "The sample text struggles with precise word choice."},
            {"criterion": "Style", "explanation": "Writing style needs improvement."}
        ]
        
        retrieval_results = evaluate_retrieval_refinement(test_nodes, corpus)
        print(f"Retrieval refinement - Avg improvement: {retrieval_results['avg_improvement']:.2f}")
        
        iterative_results = evaluate_iterative_refinement(samples, iterations=3)
        print(f"Iterative refinement - Generated {len(iterative_results)} evaluation records")
        
        metrics = compute_evaluation_metrics(iterative_results)
        print(f"Final metrics: {metrics}")
        
        print("Evaluation test completed successfully.\n")
        return True
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        return False

if __name__ == '__main__':
    test_evaluate()
