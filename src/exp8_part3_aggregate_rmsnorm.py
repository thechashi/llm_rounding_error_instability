"""
Experiment 8 Part 3: RMSNorm Impact Analysis at Divergence Points

This script analyzes how the final RMSNorm affects alignment with top-5 
unembedding vectors across all questions, focusing on divergence indices.

Purpose:
--------
1. For each question, analyze the token BEFORE divergence
2. Compute angle changes caused by RMSNorm for top-5 tokens
3. Aggregate statistics across all questions
4. Determine if RMSNorm systematically causes ranking shifts
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
    
    # Extract unembedding matrix (lm_head weight)
    unembedding_matrix = model.lm_head.weight.detach().cpu().numpy()
    
    print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    
    return unembedding_matrix, tokenizer

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return np.dot(vec1_norm, vec2_norm)

def analyze_token_at_index(
    result_dir: Path,
    question_id: int,
    token_idx: int,
    unembedding_matrix: np.ndarray,
    tokenizer,
    top_k: int = 5
) -> Dict:
    """
    Analyze how RMSNorm affects alignment with top-k unembedding vectors
    at a specific token index.
    """
    
    question_dir = result_dir / f"question_{question_id:02d}"
    
    # Load the actual top-5 words that were predicted
    with open(question_dir / "words.json", 'r') as f:
        words_data = json.load(f)
        generated_word = words_data['generated_words'][token_idx]
        top_5_words = words_data['top_5_words'][token_idx]
        top_10_token_ids = words_data['top_10_token_ids'][token_idx]
    
    # Get top-k token IDs
    top_k_token_ids = top_10_token_ids[:top_k]
    
    # Load representations BEFORE and AFTER final norm
    last_layer_before_norm = np.load(question_dir / "last_layer_before_norm_outputs.npy")
    final_norm_outputs = np.load(question_dir / "final_norm_outputs.npy")
    
    rep_before_norm = last_layer_before_norm[token_idx]
    rep_after_norm = final_norm_outputs[token_idx]
    
    # Get unembedding vectors for top-k tokens
    top_k_unembedding_vectors = unembedding_matrix[top_k_token_ids]
    
    # Compute similarities BEFORE norm
    similarities_before = []
    for rank, (token_id, unembedding_vec, word) in enumerate(zip(top_k_token_ids, top_k_unembedding_vectors, top_5_words), 1):
        cos_sim = cosine_similarity(rep_before_norm, unembedding_vec)
        angle_deg = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
        similarities_before.append({
            'rank': rank,
            'word': word,
            'token_id': int(token_id),
            'cosine_similarity': float(cos_sim),
            'angle_degrees': float(angle_deg)
        })
    
    # Compute similarities AFTER norm
    similarities_after = []
    for rank, (token_id, unembedding_vec, word) in enumerate(zip(top_k_token_ids, top_k_unembedding_vectors, top_5_words), 1):
        cos_sim = cosine_similarity(rep_after_norm, unembedding_vec)
        angle_deg = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
        similarities_after.append({
            'rank': rank,
            'word': word,
            'token_id': int(token_id),
            'cosine_similarity': float(cos_sim),
            'angle_degrees': float(angle_deg)
        })
    
    # Sort by cosine similarity to check ranking
    sorted_before = sorted(similarities_before, key=lambda x: x['cosine_similarity'], reverse=True)
    sorted_after = sorted(similarities_after, key=lambda x: x['cosine_similarity'], reverse=True)
    
    rank_changed = (sorted_before[0]['word'] != sorted_after[0]['word'])
    
    # Compute angle changes
    angle_changes = []
    for i in range(top_k):
        angle_change = similarities_after[i]['angle_degrees'] - similarities_before[i]['angle_degrees']
        angle_changes.append({
            'rank': i + 1,
            'word': similarities_before[i]['word'],
            'angle_before': similarities_before[i]['angle_degrees'],
            'angle_after': similarities_after[i]['angle_degrees'],
            'angle_change': float(angle_change)
        })
    
    results = {
        'question_id': question_id,
        'token_index': token_idx,
        'generated_word': generated_word,
        'top_k_words': top_5_words[:top_k],
        'before_norm': similarities_before,
        'after_norm': similarities_after,
        'ranking_changed': rank_changed,
        'most_similar_before': sorted_before[0]['word'],
        'most_similar_after': sorted_after[0]['word'],
        'angle_changes': angle_changes
    }
    
    return results

def compute_aggregate_statistics(all_results: List[Dict], top_k: int = 5) -> Dict:
    """Compute aggregate statistics across all analyzed tokens"""
    
    total_tokens = len(all_results)
    ranking_changed_count = sum(1 for r in all_results if r['ranking_changed'])
    
    # Aggregate angle changes for each rank position
    angle_changes_by_rank = {i+1: [] for i in range(top_k)}
    cosine_sim_before_by_rank = {i+1: [] for i in range(top_k)}
    cosine_sim_after_by_rank = {i+1: [] for i in range(top_k)}
    
    for result in all_results:
        for change_data in result['angle_changes']:
            rank = change_data['rank']
            angle_changes_by_rank[rank].append(change_data['angle_change'])
        
        for before_data in result['before_norm']:
            rank = before_data['rank']
            cosine_sim_before_by_rank[rank].append(before_data['cosine_similarity'])
        
        for after_data in result['after_norm']:
            rank = after_data['rank']
            cosine_sim_after_by_rank[rank].append(after_data['cosine_similarity'])
    
    # Compute statistics
    statistics = {
        'total_tokens_analyzed': total_tokens,
        'ranking_changed_count': ranking_changed_count,
        'ranking_changed_percentage': (ranking_changed_count / total_tokens * 100) if total_tokens > 0 else 0,
        'by_rank': {}
    }
    
    for rank in range(1, top_k + 1):
        angle_changes = angle_changes_by_rank[rank]
        cos_sim_before = cosine_sim_before_by_rank[rank]
        cos_sim_after = cosine_sim_after_by_rank[rank]
        
        statistics['by_rank'][f'rank_{rank}'] = {
            'mean_angle_change_deg': float(np.mean(angle_changes)) if angle_changes else 0,
            'std_angle_change_deg': float(np.std(angle_changes)) if angle_changes else 0,
            'mean_cosine_sim_before': float(np.mean(cos_sim_before)) if cos_sim_before else 0,
            'mean_cosine_sim_after': float(np.mean(cos_sim_after)) if cos_sim_after else 0,
            'mean_cosine_sim_change': float(np.mean(cos_sim_after) - np.mean(cos_sim_before)) if (cos_sim_before and cos_sim_after) else 0
        }
    
    return statistics

def print_summary(statistics: Dict, top_k: int = 5):
    """Print a summary of the aggregate statistics"""
    
    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS ACROSS ALL QUESTIONS")
    print(f"{'='*80}")
    
    print(f"\nTotal tokens analyzed: {statistics['total_tokens_analyzed']}")
    print(f"Ranking changes: {statistics['ranking_changed_count']} ({statistics['ranking_changed_percentage']:.2f}%)")
    
    print(f"\n{'STATISTICS BY RANK POSITION':^80}")
    print(f"{'-'*80}")
    print(f"{'Rank':<8} {'Avg Angle':<15} {'Std Angle':<15} {'Avg Cos Before':<18} {'Avg Cos After':<18}")
    print(f"{'':^8} {'Change (°)':<15} {'Change (°)':<15} {'':<18} {'':<18}")
    print(f"{'-'*80}")
    
    for rank in range(1, top_k + 1):
        rank_key = f'rank_{rank}'
        rank_stats = statistics['by_rank'][rank_key]
        
        print(f"{rank:<8} "
              f"{rank_stats['mean_angle_change_deg']:<+15.6f} "
              f"{rank_stats['std_angle_change_deg']:<15.6f} "
              f"{rank_stats['mean_cosine_sim_before']:<18.6f} "
              f"{rank_stats['mean_cosine_sim_after']:<18.6f}")
    
    print(f"\n{'KEY INSIGHTS':^80}")
    print(f"{'-'*80}")
    
    # Find rank with largest mean angle change
    max_change_rank = max(range(1, top_k + 1), 
                         key=lambda r: abs(statistics['by_rank'][f'rank_{r}']['mean_angle_change_deg']))
    max_change_value = statistics['by_rank'][f'rank_{max_change_rank}']['mean_angle_change_deg']
    
    print(f"• Rank {max_change_rank} shows the largest average angle change: {max_change_value:+.6f}°")
    
    # Check if rank 1 is most affected
    rank1_change = statistics['by_rank']['rank_1']['mean_angle_change_deg']
    print(f"• Rank 1 (top prediction) average angle change: {rank1_change:+.6f}°")
    
    if abs(rank1_change) > 0.01:
        print(f"  → RMSNorm significantly affects the top prediction!")
    else:
        print(f"  → RMSNorm has minimal effect on the top prediction")
    
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze RMSNorm impact across all questions at divergence")
    parser.add_argument("result_dir", type=str, help="Path to exp8_part1 results directory")
    parser.add_argument("divergence_file", type=str, help="Path to exp8_part2 divergence_analysis.json")
    parser.add_argument("--model_path", type=str, 
                       default="/home/chashi/Research/Llama-3.1-8B-Instruct",
                       help="Path to model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to analyze")
    parser.add_argument("--output_file", type=str, default="rmsnorm_aggregate_analysis.json")
    parser.add_argument("--analyze_before_divergence", action='store_true',
                       help="Analyze token BEFORE divergence (where both agree)")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    # Load model to get unembedding matrix
    unembedding_matrix, tokenizer = load_model_for_unembedding(args.model_path)
    
    # Load divergence analysis
    print(f"\nLoading divergence analysis from {args.divergence_file}...")
    with open(args.divergence_file, 'r') as f:
        divergence_data = json.load(f)
    
    all_results = []
    
    # Analyze each question at divergence point
    for question_key, question_data in divergence_data.items():
        if question_data.get('status') == 'no_divergence':
            print(f"{question_key}: No divergence found, skipping.")
            continue
        
        question_id = int(question_key.split('_')[1])
        divergence_idx = question_data['divergence_index']
        
        # Analyze either at divergence or one token before
        if args.analyze_before_divergence:
            analysis_idx = divergence_idx - 1
            if analysis_idx < 0:
                print(f"{question_key}: Divergence at index 0, cannot analyze previous token.")
                continue
            print(f"\n{question_key}: Analyzing token {analysis_idx} (BEFORE divergence at {divergence_idx})")
        else:
            analysis_idx = divergence_idx
            print(f"\n{question_key}: Analyzing token {analysis_idx} (AT divergence)")
        
        try:
            results = analyze_token_at_index(
                result_dir=result_dir,
                question_id=question_id,
                token_idx=analysis_idx,
                unembedding_matrix=unembedding_matrix,
                tokenizer=tokenizer,
                top_k=args.top_k
            )
            
            all_results.append(results)
            
            # Print individual result summary
            print(f"  Generated: '{results['generated_word']}'")
            print(f"  Top-{args.top_k}: {results['top_k_words']}")
            print(f"  Ranking changed: {results['ranking_changed']}")
            if results['ranking_changed']:
                print(f"    Before: '{results['most_similar_before']}' → After: '{results['most_similar_after']}'")
            
        except Exception as e:
            print(f"  Error analyzing {question_key}: {e}")
            continue
    
    if not all_results:
        print("\nNo results to aggregate. Exiting.")
        return
    
    # Compute aggregate statistics
    statistics = compute_aggregate_statistics(all_results, top_k=args.top_k)
    
    # Print summary
    print_summary(statistics, top_k=args.top_k)
    
    # Prepare output
    output_data = {
        'metadata': {
            'result_dir': str(result_dir),
            'divergence_file': args.divergence_file,
            'top_k': args.top_k,
            'analyzed_before_divergence': args.analyze_before_divergence
        },
        'aggregate_statistics': statistics,
        'individual_results': all_results
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Analysis complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()

"""
Example usage:

# Analyze AT divergence point
python exp8_part3_aggregate_rmsnorm.py \
    "/home/chashi/Research/llm_rounding_error_instability/results/exp8_part1_2025-10-30_12-13-27" \
    "/home/chashi/Research/llm_rounding_error_instability/results/exp8_part2_divergence_analysis.json" \
    --top_k 5 \
    --output_file exp8_part3_rmsnorm_at_divergence.json

# Analyze BEFORE divergence point (where both GPUs still agree)
python exp8_part3_aggregate_rmsnorm.py \
    results/GPU0_exp8_part1_2025-11-02/ \
    divergence_analysis.json \
    --top_k 5 \
    --analyze_before_divergence \
    --output_file rmsnorm_before_divergence.json
"""
