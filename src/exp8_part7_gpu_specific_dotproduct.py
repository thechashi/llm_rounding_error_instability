"""
Experiment 8 Part 7: GPU-Specific Dot Product Analysis at Divergence Points

This script computes dot products at divergence and pre-divergence tokens only.

Usage:
------
python exp8_part7_gpu_specific_dotproduct.py \
    /path/to/gpu1/results \
    --divergence_json divergence_analysis.json \
    --output gpu1_dotproduct_divergence_analysis.json
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_on_gpu(model_path: str):
    """Load model with automatic device mapping"""
    print(f"Loading model with automatic device mapping...")
    print(f"Model path: {model_path}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    lm_head_weight = model.lm_head.weight
    num_layers = len(model.model.layers)
    
    print(f"✓ Model loaded with device_map='auto'")
    print(f"  lm_head weight shape: {lm_head_weight.shape}")
    print(f"  Number of layers: {num_layers}")
    print(f"  lm_head device: {lm_head_weight.device}")
    print(f"  Dtype: {lm_head_weight.dtype}")
    
    lm_head_device = lm_head_weight.device
    
    return model, tokenizer, lm_head_weight, num_layers, lm_head_device

def get_stage_names(num_layers: int) -> List[str]:
    """Get all stage names in order"""
    stages = ['input_embeddings']
    
    for layer_idx in range(num_layers):
        stages.extend([
            f'layer{layer_idx}_input_layernorm',
            f'layer{layer_idx}_self_attn',
            f'layer{layer_idx}_attn_o_proj',
            f'layer{layer_idx}_post_attention_layernorm',
            f'layer{layer_idx}_mlp',
        ])
    
    stages.extend([
        'last_layer_before_norm',
        'final_norm'
    ])
    
    return stages

def analyze_token_at_stage(
    question_dir: Path,
    token_idx: int,
    stage_name: str,
    lm_head_weight: torch.Tensor,
    lm_head_device: torch.device,
    top_2_token_ids: List[int]
) -> Dict:
    """
    Analyze a single token at a single stage.
    Returns dot product results or None if stage doesn't exist.
    """
    stage_file = question_dir / f"{stage_name}.npy"
    
    if not stage_file.exists():
        return None
    
    # Load representation from disk
    rep_np = np.load(stage_file)
    
    if token_idx >= rep_np.shape[0]:
        return None
    
    rep_at_position = rep_np[token_idx]
    
    # Check if dimensions match (skip MLP intermediate layers)
    if rep_at_position.shape[0] != lm_head_weight.shape[1]:
        return None
    
    # Get unembedding vectors for top-2 tokens
    unembed_vec_0 = lm_head_weight[top_2_token_ids[0]]
    unembed_vec_1 = lm_head_weight[top_2_token_ids[1]]
    
    # Move to same device as lm_head
    rep_gpu = torch.from_numpy(rep_at_position).to(lm_head_device)
    
    # Compute dot products ON THIS GPU
    dot_0 = torch.dot(rep_gpu, unembed_vec_0).cpu().item()
    dot_1 = torch.dot(rep_gpu, unembed_vec_1).cpu().item()
    
    difference = dot_0 - dot_1
    favors = "Top-1" if difference > 0 else "Top-2"
    
    return {
        'dot_product_top1': float(dot_0),
        'dot_product_top2': float(dot_1),
        'difference': float(difference),
        'favors': favors,
        'margin': abs(float(difference))
    }

def analyze_question(
    question_dir: Path,
    question_id: int,
    divergence_index: int,
    lm_head_weight: torch.Tensor,
    lm_head_device: torch.device,
    tokenizer,
    num_layers: int
) -> Dict:
    """
    Analyze one question at divergence and pre-divergence positions.
    """
    
    print(f"\n{'='*100}")
    print(f"ANALYZING QUESTION {question_id}")
    print(f"  Divergence at token index: {divergence_index}")
    print(f"{'='*100}")
    
    # Load words.json to get token predictions
    words_file = question_dir / "words.json"
    if not words_file.exists():
        print(f"  ✗ words.json not found, skipping.")
        return None
    
    with open(words_file, 'r') as f:
        words_data = json.load(f)
    
    generated_words = words_data['generated_words']
    top_5_words = words_data['top_5_words']
    top_10_token_ids = words_data['top_10_token_ids']
    
    num_tokens = len(generated_words)
    print(f"  Total tokens generated: {num_tokens}")
    
    # Get all stage names
    stage_names = get_stage_names(num_layers)
    
    # Analyze two positions: before divergence and at divergence
    positions_to_analyze = []
    
    # Before divergence (if exists)
    if divergence_index > 0:
        positions_to_analyze.append({
            'label': 'before_divergence',
            'token_idx': divergence_index - 1
        })
    
    # At divergence
    if divergence_index < num_tokens:
        positions_to_analyze.append({
            'label': 'at_divergence',
            'token_idx': divergence_index
        })
    
    results_by_position = {}
    
    for position_info in positions_to_analyze:
        label = position_info['label']
        token_idx = position_info['token_idx']
        
        print(f"\n  {'─'*96}")
        print(f"  POSITION: {label.upper()} (token index {token_idx})")
        print(f"  {'─'*96}")
        
        # Get top-2 tokens at this position
        top_2_token_ids = top_10_token_ids[token_idx][:2]
        top_2_words = top_5_words[token_idx][:2]
        generated_word = generated_words[token_idx]
        
        print(f"    Generated token: '{generated_word}'")
        print(f"    Top-2 predictions: {top_2_words} (ids: {top_2_token_ids})")
        
        # Analyze all stages at this position
        stage_results = {}
        
        for stage_name in stage_names:
            result = analyze_token_at_stage(
                question_dir=question_dir,
                token_idx=token_idx,
                stage_name=stage_name,
                lm_head_weight=lm_head_weight,
                lm_head_device=lm_head_device,
                top_2_token_ids=top_2_token_ids
            )
            
            if result is not None:
                stage_results[stage_name] = result
        
        print(f"    Stages analyzed: {len(stage_results)}")
        
        # Print ALL stages in a table format
        print(f"\n    {'Stage':<40} {'Favors':<8} {'Difference':<15} {'Margin':<12}")
        print(f"    {'-'*80}")
        
        for stage_name in stage_names:
            if stage_name in stage_results:
                s = stage_results[stage_name]
                print(f"    {stage_name:<40} {s['favors']:<8} {s['difference']:>+14.8f} {s['margin']:>11.8f}")
        
        # Check if final_norm has very small margin
        if 'final_norm' in stage_results:
            margin = stage_results['final_norm']['margin']
            if margin < 0.01:
                print(f"\n    ⚠️  WARNING: Final norm margin is VERY SMALL: {margin:.10f}")
                print(f"        This token prediction is highly sensitive to numerical precision!")
        
        results_by_position[label] = {
            'token_index': token_idx,
            'generated_word': generated_word,
            'top_2_words': top_2_words,
            'top_2_token_ids': [int(x) for x in top_2_token_ids],
            'stage_results': stage_results
        }
    
    return {
        'question_id': question_id,
        'divergence_index': divergence_index,
        'positions': results_by_position
    }

def print_comparison(results: List[Dict], num_layers: int):
    """Print comparison between before_divergence and at_divergence for ALL stages"""
    
    print(f"\n{'='*120}")
    print(f"DIVERGENCE COMPARISON - ALL STAGES")
    print(f"{'='*120}")
    
    for question_result in results:
        if question_result is None:
            continue
        
        print(f"\n{'─'*120}")
        print(f"Question {question_result['question_id']} (Divergence at index {question_result['divergence_index']})")
        print(f"{'─'*120}")
        
        positions = question_result['positions']
        
        if 'before_divergence' not in positions or 'at_divergence' not in positions:
            print("  ⚠️  Missing position data")
            continue
        
        before = positions['before_divergence']
        at_div = positions['at_divergence']
        
        before_stages = before.get('stage_results', {})
        at_div_stages = at_div.get('stage_results', {})
        
        # Get all stage names in order
        stage_names = get_stage_names(num_layers)
        
        # Print header
        print(f"\n{'Stage':<40} {'Before Divergence':<35} {'At Divergence':<35}")
        print(f"{'':40} {'Favors':<10} {'Diff':<12} {'Margin':<11} {'Favors':<10} {'Diff':<12} {'Margin':<11}")
        print(f"{'-'*120}")
        
        # Print all stages
        for stage_name in stage_names:
            before_result = before_stages.get(stage_name)
            at_div_result = at_div_stages.get(stage_name)
            
            if before_result is None and at_div_result is None:
                continue
            
            # Before divergence values
            if before_result:
                before_favors = before_result['favors']
                before_diff = before_result['difference']
                before_margin = before_result['margin']
                before_str = f"{before_favors:<10} {before_diff:>+11.8f} {before_margin:>10.8f}"
            else:
                before_str = f"{'N/A':<10} {'':>11} {'':>10}"
            
            # At divergence values
            if at_div_result:
                at_div_favors = at_div_result['favors']
                at_div_diff = at_div_result['difference']
                at_div_margin = at_div_result['margin']
                at_div_str = f"{at_div_favors:<10} {at_div_diff:>+11.8f} {at_div_margin:>10.8f}"
            else:
                at_div_str = f"{'N/A':<10} {'':>11} {'':>10}"
            
            # Check for change in favored token
            change_marker = ""
            if before_result and at_div_result:
                if before_result['favors'] != at_div_result['favors']:
                    change_marker = " ⚠️ FLIP"
            
            print(f"{stage_name:<40} {before_str}   {at_div_str}{change_marker}")
        
        # Summary statistics
        print(f"\n{'SUMMARY':^120}")
        print(f"{'-'*120}")
        
        if 'final_norm' in before_stages and 'final_norm' in at_div_stages:
            before_margin = before_stages['final_norm']['margin']
            at_div_margin = at_div_stages['final_norm']['margin']
            
            print(f"Final norm margin comparison:")
            print(f"  Before divergence: {before_margin:.10f}")
            print(f"  At divergence:     {at_div_margin:.10f}")
            print(f"  Change:            {at_div_margin - before_margin:+.10f}")
            
            if at_div_margin < 0.01:
                print(f"  ⚠️  Critically small margin at divergence!")

def main():
    parser = argparse.ArgumentParser(description="GPU-specific dot product analysis at divergence points")
    parser.add_argument("results_dir", type=str, 
                       help="Path to results directory (contains question_XX folders)")
    parser.add_argument("--divergence_json", type=str, required=True,
                       help="Path to JSON file with divergence indices")
    parser.add_argument("--model_path", type=str,
                       default="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct",
                       help="Path to model")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load divergence indices
    print(f"Loading divergence indices from {args.divergence_json}...")
    with open(args.divergence_json, 'r') as f:
        divergence_data = json.load(f)
    
    # Load model with automatic device mapping
    model, tokenizer, lm_head_weight, num_layers, lm_head_device = load_model_on_gpu(
        args.model_path
    )
    
    # Analyze each question
    all_results = []
    
    for question_key, question_info in divergence_data.items():
        # Extract question number from key like "question_01"
        question_id = int(question_key.split("_")[1])
        divergence_index = question_info['divergence_index']
        
        question_dir = results_dir / f"question_{question_id:02d}"
        
        if not question_dir.exists():
            print(f"\n⚠️  Question directory not found: {question_dir}")
            continue
        
        result = analyze_question(
            question_dir=question_dir,
            question_id=question_id,
            divergence_index=divergence_index,
            lm_head_weight=lm_head_weight,
            lm_head_device=lm_head_device,
            tokenizer=tokenizer,
            num_layers=num_layers
        )
        
        if result:
            all_results.append(result)
    
    # Print comparison summary with ALL stages
    print_comparison(all_results, num_layers)
    
    # Save results
    print(f"\n{'='*100}")
    print(f"Saving results to {args.output}...")
    
    # Get GPU info for metadata
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append({
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9
        })
    
    output_data = {
        'gpu_info': gpu_info,
        'lm_head_device': str(lm_head_device),
        'model_path': args.model_path,
        'results_dir': str(results_dir),
        'divergence_json': args.divergence_json,
        'num_questions_analyzed': len(all_results),
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved!")
    print(f"{'='*100}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

"""
EXAMPLE USAGE:
==============

On GPU1 machine (e.g., 2x A5000 24GB):
---------------------------------------
python exp8_part7_gpu_specific_dotproduct.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A5000x2_exp8_part5_comprehensive_2025-11-02_12-49-05" \
    --divergence_json "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_comprehensive_divergence_analysis.json" \
    --output exp8_part7_A5000_divergence_dotproducts.json

On GPU2 machine (e.g., A6000):
-------------------------------
python exp8_part7_gpu_specific_dotproduct.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A6000_exp8_part5_comprehensive_2025-11-02_12-55-07" \
    --divergence_json "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_comprehensive_divergence_analysis.json" \
    --output A6000_divergence_dotproducts.json

Analyze specific questions only:
---------------------------------
python exp8_part7_gpu_specific_dotproduct.py \
    /path/to/results \
    --divergence_json divergence_data.json \
    --output results.json \
    --questions 1,2,3
"""