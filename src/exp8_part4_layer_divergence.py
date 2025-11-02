"""
Experiment 8 Part 4: Layer-by-Layer Divergence Localization

This script traces through the network layer-by-layer to identify WHERE
the representation shift occurs that eventually leads to different token predictions.

Purpose:
--------
1. Load representations from both GPUs at the token BEFORE divergence
2. Compare representations at multiple points in the network:
   - Input embeddings
   - After layer 0 (input layernorm, post-attention layernorm)
   - After last layer (before final norm)
   - After final norm
3. Compute alignment with top-2 unembedding vectors at each stage
4. Identify at which layer the "rank flip" first appears

Key Question:
-------------
At what point in the forward pass does the representation start favoring
the second-choice token over the first-choice token?
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_for_unembedding(model_path: str):
    """Load model to access unembedding weights (lm_head)"""
    print("Loading model to extract unembedding matrix...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    unembedding_matrix = model.lm_head.weight.detach().cpu().numpy()
    
    print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    
    return unembedding_matrix, tokenizer

def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute dot product (actual logit value)"""
    return np.dot(vec1, vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return np.dot(vec1_norm, vec2_norm)

def analyze_layer_progression(
    gpu1_dir: Path,
    gpu2_dir: Path,
    question_id: int,
    divergence_idx: int,
    unembedding_matrix: np.ndarray,
    tokenizer
) -> Dict:
    """
    Analyze how representations at different layers align with top-2 unembedding vectors.
    """
    
    # Analyze the token BEFORE divergence (where both GPUs still generated same token)
    analysis_idx = divergence_idx - 1
    
    if analysis_idx < 0:
        print(f"Divergence at index 0, cannot analyze.")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Question {question_id}: Analyzing token {analysis_idx} (before divergence at {divergence_idx})")
    print(f"{'='*80}")
    
    # First, load GPU1 data to establish reference top-2
    gpu1_question_dir = gpu1_dir / f"question_{question_id:02d}"
    with open(gpu1_question_dir / "words.json", 'r') as f:
        gpu1_words_data = json.load(f)
        gpu1_top_10_ids_at_divergence = gpu1_words_data['top_10_token_ids'][divergence_idx]
        gpu1_top_5_at_divergence = gpu1_words_data['top_5_words'][divergence_idx]
    
    # Use GPU1's top-2 as REFERENCE for both GPUs
    reference_top_2_ids = gpu1_top_10_ids_at_divergence[:2]
    reference_top_2_words = gpu1_top_5_at_divergence[:2]
    
    print(f"\nREFERENCE Top-2 (from GPU1): {reference_top_2_words} (ids: {reference_top_2_ids})")
    
    results = {
        'question_id': question_id,
        'analysis_index': analysis_idx,
        'divergence_index': divergence_idx,
        'reference_top_2_ids': [int(x) for x in reference_top_2_ids],
        'reference_top_2_words': reference_top_2_words,
        'gpu1': {},
        'gpu2': {}
    }
    
    # Load data from both GPUs
    for gpu_name, gpu_dir in [('gpu1', gpu1_dir), ('gpu2', gpu2_dir)]:
        question_dir = gpu_dir / f"question_{question_id:02d}"
        
        # Load generated words to know what diverges
        with open(question_dir / "words.json", 'r') as f:
            words_data = json.load(f)
            word_at_analysis = words_data['generated_words'][analysis_idx]
            word_at_divergence = words_data['generated_words'][divergence_idx]
            this_gpu_top_5_at_divergence = words_data['top_5_words'][divergence_idx]
        
        print(f"\n{gpu_name.upper()}:")
        print(f"  Token at analysis index {analysis_idx}: '{word_at_analysis}'")
        print(f"  Token at divergence {divergence_idx}: '{word_at_divergence}'")
        print(f"  This GPU's top-5 at divergence: {this_gpu_top_5_at_divergence}")
        print(f"  Using reference top-2 for comparison: {reference_top_2_words}")
        
        # Load all representations at analysis_idx
        input_embeddings = np.load(question_dir / "input_embeddings.npy")
        layer0_input_ln = np.load(question_dir / "layer0_input_layernorm_outputs.npy")
        layer0_post_attn_ln = np.load(question_dir / "layer0_post_attention_layernorm_outputs.npy")
        last_layer_before_norm = np.load(question_dir / "last_layer_before_norm_outputs.npy")
        final_norm = np.load(question_dir / "final_norm_outputs.npy")
        
        # Extract representations at analysis_idx
        reps = {
            'input_embedding': input_embeddings[analysis_idx],
            'layer0_input_ln': layer0_input_ln[analysis_idx],
            'layer0_post_attn_ln': layer0_post_attn_ln[analysis_idx],
            'last_layer_before_norm': last_layer_before_norm[analysis_idx],
            'final_norm': final_norm[analysis_idx]
        }
        
        # Get unembedding vectors for REFERENCE top-2 tokens
        unembed_top1 = unembedding_matrix[reference_top_2_ids[0]]
        unembed_top2 = unembedding_matrix[reference_top_2_ids[1]]
        
        # Compute BOTH dot products and cosine similarities
        stage_results = {}
        
        print(f"\n  {'Stage':<30} {'Top-1 Dot':<15} {'Top-2 Dot':<15} {'Diff':<15} "
              f"{'Top-1 Cos':<15} {'Top-2 Cos':<15} {'Favors (Dot)':<12}")
        print(f"  {'-'*110}")
        
        for stage_name, rep in reps.items():
            # DOT PRODUCTS (actual logit computation)
            dot_top1 = dot_product(rep, unembed_top1)
            dot_top2 = dot_product(rep, unembed_top2)
            dot_diff = dot_top1 - dot_top2
            dot_favors = "Top-1" if dot_diff > 0 else "Top-2"
            
            # COSINE SIMILARITIES (for reference)
            cos_top1 = cosine_similarity(rep, unembed_top1)
            cos_top2 = cosine_similarity(rep, unembed_top2)
            
            stage_results[stage_name] = {
                'top1_dot_product': float(dot_top1),
                'top2_dot_product': float(dot_top2),
                'dot_difference': float(dot_diff),
                'top1_cosine': float(cos_top1),
                'top2_cosine': float(cos_top2),
                'favors': dot_favors  # Use dot product for "favors"
            }
            
            print(f"  {stage_name:<30} {dot_top1:<15.6f} {dot_top2:<15.6f} {dot_diff:<+15.6f} "
                  f"{cos_top1:<15.6f} {cos_top2:<15.6f} {dot_favors:<12}")
        
        results[gpu_name] = {
            'word_at_analysis': word_at_analysis,
            'word_at_divergence': word_at_divergence,
            'this_gpu_top_5': this_gpu_top_5_at_divergence,
            'stage_results': stage_results
        }
    
    return results

def compare_gpus_and_find_flip_point(results: Dict) -> Dict:
    """
    Compare both GPUs and identify where the "flip" first appears.
    """
    
    print(f"\n{'='*80}")
    print("CROSS-GPU COMPARISON: Finding where representations diverge")
    print(f"{'='*80}")
    
    gpu1_stages = results['gpu1']['stage_results']
    gpu2_stages = results['gpu2']['stage_results']
    
    comparison = {}
    
    print(f"\n{'Stage':<30} {'GPU1 Favors':<12} {'GPU2 Favors':<12} {'Agreement':<12} {'GPU1 Diff':<15} {'GPU2 Diff':<15}")
    print(f"{'-'*100}")
    
    for stage_name in gpu1_stages.keys():
        gpu1_favors = gpu1_stages[stage_name]['favors']
        gpu2_favors = gpu2_stages[stage_name]['favors']
        agree = "✓" if gpu1_favors == gpu2_favors else "✗ DIVERGE"
        gpu1_diff = gpu1_stages[stage_name]['dot_difference']
        gpu2_diff = gpu2_stages[stage_name]['dot_difference']
        
        comparison[stage_name] = {
            'gpu1_favors': gpu1_favors,
            'gpu2_favors': gpu2_favors,
            'agreement': agree == "✓",
            'gpu1_dot_difference': gpu1_diff,
            'gpu2_dot_difference': gpu2_diff
        }
        
        print(f"{stage_name:<30} {gpu1_favors:<12} {gpu2_favors:<12} {agree:<12} {gpu1_diff:<+15.6f} {gpu2_diff:<+15.6f}")
    
    # Find first stage where they disagree
    first_divergence = None
    for stage_name in ['input_embedding', 'layer0_input_ln', 'layer0_post_attn_ln', 
                       'last_layer_before_norm', 'final_norm']:
        if not comparison[stage_name]['agreement']:
            first_divergence = stage_name
            break
    
    print(f"\n{'KEY FINDINGS':^100}")
    print(f"{'-'*100}")
    
    if first_divergence:
        print(f"• First divergence appears at: {first_divergence}")
        print(f"  GPU1 favors: {comparison[first_divergence]['gpu1_favors']}")
        print(f"  GPU2 favors: {comparison[first_divergence]['gpu2_favors']}")
    else:
        print(f"• Both GPUs favor the same token at all stages!")
        print(f"  (Divergence must come from tie-breaking or numerical precision in logit computation)")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer divergence localization")
    parser.add_argument("gpu1_dir", type=str, help="Path to GPU1 exp8_part1 results")
    parser.add_argument("gpu2_dir", type=str, help="Path to GPU2 exp8_part1 results")
    parser.add_argument("divergence_file", type=str, help="Path to exp8_part2 divergence_analysis.json")
    parser.add_argument("--model_path", type=str, 
                       default="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct",
                       help="Path to model")
    parser.add_argument("--output_file", type=str, default="exp8_part4_layer_divergence_analysis.json")
    
    args = parser.parse_args()
    
    gpu1_dir = Path(args.gpu1_dir)
    gpu2_dir = Path(args.gpu2_dir)
    
    # Load model to get unembedding matrix
    unembedding_matrix, tokenizer = load_model_for_unembedding(args.model_path)
    
    # Load divergence analysis
    print(f"\nLoading divergence analysis from {args.divergence_file}...")
    with open(args.divergence_file, 'r') as f:
        divergence_data = json.load(f)
    
    all_results = {}
    
    # Analyze each question
    for question_key, question_data in divergence_data.items():
        if question_data.get('status') == 'no_divergence':
            print(f"\n{question_key}: No divergence found, skipping.")
            continue
        
        question_id = int(question_key.split('_')[1])
        divergence_idx = question_data['divergence_index']
        
        # Analyze layer progression
        results = analyze_layer_progression(
            gpu1_dir=gpu1_dir,
            gpu2_dir=gpu2_dir,
            question_id=question_id,
            divergence_idx=divergence_idx,
            unembedding_matrix=unembedding_matrix,
            tokenizer=tokenizer
        )
        
        if not results:
            continue
        
        # Compare GPUs and find flip point
        comparison = compare_gpus_and_find_flip_point(results)
        
        results['comparison'] = comparison
        all_results[question_key] = results
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to {args.output_file}")
    print(f"{'='*80}")
    
    # Summary across all questions
    print(f"\n{'SUMMARY ACROSS ALL QUESTIONS':^80}")
    print(f"{'-'*80}")
    
    first_divergence_counts = {}
    no_divergence_count = 0
    
    for question_key, result in all_results.items():
        comparison = result['comparison']
        found_divergence = False
        for stage in ['input_embedding', 'layer0_input_ln', 'layer0_post_attn_ln', 
                     'last_layer_before_norm', 'final_norm']:
            if not comparison[stage]['agreement']:
                first_divergence_counts[stage] = first_divergence_counts.get(stage, 0) + 1
                found_divergence = True
                break
        
        if not found_divergence:
            no_divergence_count += 1
    
    print("\nFirst divergence point frequency:")
    for stage in ['input_embedding', 'layer0_input_ln', 'layer0_post_attn_ln', 
                 'last_layer_before_norm', 'final_norm']:
        count = first_divergence_counts.get(stage, 0)
        if count > 0:
            print(f"  {stage:<30}: {count} question(s)")
    
    if no_divergence_count > 0:
        print(f"  {'No divergence (all stages agree)':<30}: {no_divergence_count} question(s)")

if __name__ == "__main__":
    main()

"""
Example usage:

python exp8_part4_layer_divergence.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A5000_exp8_part1_2025-10-30_12-09-20" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A6000_exp8_part1_2025-10-30_12-13-27" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_divergence_analysis.json" \
    --output_file exp8_part4_layer_divergence_analysis.json
"""