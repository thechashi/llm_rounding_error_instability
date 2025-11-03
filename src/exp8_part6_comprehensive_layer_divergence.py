"""
Experiment 8 Part 6: Comprehensive Layer-by-Layer Divergence Localization

This script traces through ALL submodules in ALL layers to identify WHERE
the representation shift occurs that eventually leads to different token predictions.

Purpose:
--------
1. Load representations from both GPUs at the token BEFORE divergence
2. Compare representations at ALL points in the network:
   - Input embeddings
   - For each layer: input_layernorm, self_attn, attn_o_proj, post_attention_layernorm,
     mlp_gate_proj, mlp_up_proj, mlp_act_fn, mlp_down_proj, mlp
   - Last layer (before final norm)
   - Final norm
3. Compute alignment with top-2 unembedding vectors at each stage
4. Identify at which exact submodule the "rank flip" first appears

Key Question:
-------------
At what exact point in the forward pass does the representation start favoring
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
    num_layers = len(model.model.layers)
    
    print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    print(f"Number of layers in model: {num_layers}")
    
    return unembedding_matrix, tokenizer, num_layers

def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute dot product (actual logit value)"""
    return np.dot(vec1, vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return np.dot(vec1_norm, vec2_norm)

def get_ordered_stage_names(num_layers: int) -> List[str]:
    """
    Generate ordered list of all stage names in the forward pass order.
    """
    stages = ['input_embeddings']
    
    for layer_idx in range(num_layers):
        stages.extend([
            f'layer{layer_idx}_input_layernorm',
            f'layer{layer_idx}_self_attn',
            f'layer{layer_idx}_attn_o_proj',
            f'layer{layer_idx}_post_attention_layernorm',
            f'layer{layer_idx}_mlp_gate_proj',
            f'layer{layer_idx}_mlp_up_proj',
            f'layer{layer_idx}_mlp_act_fn',
            f'layer{layer_idx}_mlp_down_proj',
            f'layer{layer_idx}_mlp',
        ])
    
    stages.extend([
        'last_layer_before_norm',
        'final_norm'
    ])
    
    return stages

def load_all_representations(question_dir: Path, analysis_idx: int, num_layers: int) -> Dict[str, np.ndarray]:
    """
    Load ALL representations from all available .npy files at the analysis index.
    """
    reps = {}
    
    # Load input embeddings
    try:
        input_embeddings = np.load(question_dir / "input_embeddings.npy")
        reps['input_embeddings'] = input_embeddings[analysis_idx]
    except:
        print(f"  Warning: Could not load input_embeddings.npy")
    
    # Load all layer submodules
    for layer_idx in range(num_layers):
        submodules = [
            'input_layernorm',
            'self_attn',
            'attn_o_proj',
            'post_attention_layernorm',
            'mlp_gate_proj',
            'mlp_up_proj',
            'mlp_act_fn',
            'mlp_down_proj',
            'mlp',
        ]
        
        for submodule in submodules:
            filename = f"layer{layer_idx}_{submodule}.npy"
            filepath = question_dir / filename
            
            if filepath.exists():
                try:
                    data = np.load(filepath)
                    reps[f'layer{layer_idx}_{submodule}'] = data[analysis_idx]
                except Exception as e:
                    print(f"  Warning: Could not load {filename}: {e}")
            else:
                # Some submodules might not exist (e.g., act_fn if not a module)
                pass
    
    # Load final layers
    try:
        last_layer = np.load(question_dir / "last_layer_before_norm.npy")
        reps['last_layer_before_norm'] = last_layer[analysis_idx]
    except:
        print(f"  Warning: Could not load last_layer_before_norm.npy")
    
    try:
        final_norm = np.load(question_dir / "final_norm.npy")
        reps['final_norm'] = final_norm[analysis_idx]
    except:
        print(f"  Warning: Could not load final_norm.npy")
    
    return reps

def analyze_layer_progression(
    gpu1_dir: Path,
    gpu2_dir: Path,
    question_id: int,
    divergence_idx: int,
    unembedding_matrix: np.ndarray,
    tokenizer,
    num_layers: int
) -> Dict:
    """
    Analyze how representations at ALL submodules align with top-2 unembedding vectors.
    """
    
    # Analyze the token BEFORE divergence
    analysis_idx = divergence_idx - 1
    
    if analysis_idx < 0:
        print(f"Divergence at index 0, cannot analyze.")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Question {question_id}: Analyzing token {analysis_idx} (before divergence at {divergence_idx})")
    print(f"{'='*80}")
    
    # Load GPU1 data to establish reference top-2
    gpu1_question_dir = gpu1_dir / f"question_{question_id:02d}"
    with open(gpu1_question_dir / "words.json", 'r') as f:
        gpu1_words_data = json.load(f)
        gpu1_top_10_ids_at_divergence = gpu1_words_data['top_10_token_ids'][divergence_idx]
        gpu1_top_5_at_divergence = gpu1_words_data['top_5_words'][divergence_idx]
    
    # Use GPU1's top-2 as REFERENCE
    reference_top_2_ids = gpu1_top_10_ids_at_divergence[:2]
    reference_top_2_words = gpu1_top_5_at_divergence[:2]
    
    print(f"\nANALYSIS CONTEXT:")
    print(f"{'-'*80}")
    print(f"At token position {analysis_idx}, after generating ' pressures':")
    print(f"  - Both GPUs had just generated the SAME token")
    print(f"  - This token gets added to the context")
    print(f"  - Now we analyze how the representations evolve through the network")
    print(f"")
    print(f"At token position {divergence_idx} (the NEXT token):")
    print(f"  - GPU1 predicts: '{reference_top_2_words[0]}' (token_id={reference_top_2_ids[0]})")
    print(f"  - GPU2 predicts: '{reference_top_2_words[1]}' (token_id={reference_top_2_ids[1]})")
    print(f"")
    print(f"WHAT WE'RE CHECKING:")
    print(f"  For each layer/submodule, we compute:")
    print(f"    dot_product(representation, unembedding['{reference_top_2_words[0]}'])")
    print(f"    dot_product(representation, unembedding['{reference_top_2_words[1]}'])")
    print(f"  This tells us which token the representation 'leans toward' at that stage.")
    print(f"{'-'*80}")
    
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
    
    # Get ordered stage names
    ordered_stages = get_ordered_stage_names(num_layers)
    
    # Load data from both GPUs
    for gpu_name, gpu_dir in [('gpu1', gpu1_dir), ('gpu2', gpu2_dir)]:
        question_dir = gpu_dir / f"question_{question_id:02d}"
        
        # Load generated words
        with open(question_dir / "words.json", 'r') as f:
            words_data = json.load(f)
            word_at_analysis = words_data['generated_words'][analysis_idx]
            word_at_divergence = words_data['generated_words'][divergence_idx]
            this_gpu_top_5_at_divergence = words_data['top_5_words'][divergence_idx]
        
        print(f"\n{gpu_name.upper()}:")
        print(f"  Token at analysis index {analysis_idx}: '{word_at_analysis}'")
        print(f"  Token at divergence {divergence_idx}: '{word_at_divergence}'")
        print(f"  This GPU's top-5 at divergence: {this_gpu_top_5_at_divergence}")
        
        # Load ALL representations
        reps = load_all_representations(question_dir, analysis_idx, num_layers)
        
        print(f"  Loaded {len(reps)} representations")
        
        # Get unembedding vectors for REFERENCE top-2 tokens
        unembed_top1 = unembedding_matrix[reference_top_2_ids[0]]
        unembed_top2 = unembedding_matrix[reference_top_2_ids[1]]
        
        # Compute alignments for all stages
        stage_results = {}
        
        print(f"\n  Computing alignments for {len(reps)} stages...")
        skipped_count = 0
        
        for stage_name in ordered_stages:
            if stage_name not in reps:
                continue
            
            rep = reps[stage_name]
            
            # Check if shapes match - skip if they don't (e.g., MLP intermediate dimensions)
            if rep.shape[0] != unembed_top1.shape[0]:
                skipped_count += 1
                continue
            
            # DOT PRODUCTS (actual logit computation)
            dot_top1 = dot_product(rep, unembed_top1)
            dot_top2 = dot_product(rep, unembed_top2)
            dot_diff = dot_top1 - dot_top2
            dot_favors = "Top-1" if dot_diff > 0 else "Top-2"
            
            # COSINE SIMILARITIES
            cos_top1 = cosine_similarity(rep, unembed_top1)
            cos_top2 = cosine_similarity(rep, unembed_top2)
            
            stage_results[stage_name] = {
                'top1_dot_product': float(dot_top1),
                'top2_dot_product': float(dot_top2),
                'dot_difference': float(dot_diff),
                'top1_cosine': float(cos_top1),
                'top2_cosine': float(cos_top2),
                'favors': dot_favors
            }
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} stages due to shape mismatch (e.g., MLP intermediate activations)")
        
        results[gpu_name] = {
            'word_at_analysis': word_at_analysis,
            'word_at_divergence': word_at_divergence,
            'this_gpu_top_5': this_gpu_top_5_at_divergence,
            'stage_results': stage_results
        }
    
    return results

def print_detailed_comparison(results: Dict, num_layers: int):
    """
    Print detailed comparison showing all stages with dot product values.
    """
    gpu1_stages = results['gpu1']['stage_results']
    gpu2_stages = results['gpu2']['stage_results']
    
    ref_top1 = results['reference_top_2_words'][0]
    ref_top2 = results['reference_top_2_words'][1]
    
    print(f"\n{'='*120}")
    print("DETAILED STAGE-BY-STAGE COMPARISON")
    print(f"Reference: Top-1='{ref_top1}' vs Top-2='{ref_top2}'")
    print(f"{'='*120}")
    
    print(f"\n{'Stage':<40} {'GPU1':<35} {'GPU2':<35} {'Agree':<10}")
    print(f"{'':40} {'Favors':<10} {'Diff':<12} {'Margin':<10} {'Favors':<10} {'Diff':<12} {'Margin':<10}")
    print(f"{'-'*120}")
    
    ordered_stages = get_ordered_stage_names(num_layers)
    
    for stage_name in ordered_stages:
        if stage_name not in gpu1_stages or stage_name not in gpu2_stages:
            continue
        
        gpu1_data = gpu1_stages[stage_name]
        gpu2_data = gpu2_stages[stage_name]
        
        gpu1_favors = gpu1_data['favors']
        gpu2_favors = gpu2_data['favors']
        
        gpu1_diff = gpu1_data['dot_difference']
        gpu2_diff = gpu2_data['dot_difference']
        
        # Calculate margins (how much stronger is the favored token)
        gpu1_margin = abs(gpu1_diff)
        gpu2_margin = abs(gpu2_diff)
        
        agree = "✓" if gpu1_favors == gpu2_favors else "✗ DIVERGE"
        
        print(f"{stage_name:<40} "
              f"{gpu1_favors:<10} {gpu1_diff:+11.6f} {gpu1_margin:9.6f}   "
              f"{gpu2_favors:<10} {gpu2_diff:+11.6f} {gpu2_margin:9.6f}   "
              f"{agree:<10}")

def compare_gpus_and_find_flip_point(results: Dict, num_layers: int) -> Dict:
    """
    Compare both GPUs and identify where the "flip" first appears.
    """
    
    print(f"\n{'='*80}")
    print("FINDING FIRST DIVERGENCE POINT")
    print(f"{'='*80}")
    
    ref_top1 = results['reference_top_2_words'][0]
    ref_top2 = results['reference_top_2_words'][1]
    ref_top1_id = results['reference_top_2_ids'][0]
    ref_top2_id = results['reference_top_2_ids'][1]
    
    print(f"\nReference tokens: Top-1='{ref_top1}' (id={ref_top1_id}) vs Top-2='{ref_top2}' (id={ref_top2_id})")
    
    gpu1_stages = results['gpu1']['stage_results']
    gpu2_stages = results['gpu2']['stage_results']
    
    comparison = {}
    ordered_stages = get_ordered_stage_names(num_layers)
    
    # Build comparison for all stages
    for stage_name in ordered_stages:
        if stage_name not in gpu1_stages or stage_name not in gpu2_stages:
            continue
        
        gpu1_favors = gpu1_stages[stage_name]['favors']
        gpu2_favors = gpu2_stages[stage_name]['favors']
        gpu1_diff = gpu1_stages[stage_name]['dot_difference']
        gpu2_diff = gpu2_stages[stage_name]['dot_difference']
        gpu1_top1_dot = gpu1_stages[stage_name]['top1_dot_product']
        gpu1_top2_dot = gpu1_stages[stage_name]['top2_dot_product']
        gpu2_top1_dot = gpu2_stages[stage_name]['top1_dot_product']
        gpu2_top2_dot = gpu2_stages[stage_name]['top2_dot_product']
        
        comparison[stage_name] = {
            'gpu1_favors': gpu1_favors,
            'gpu2_favors': gpu2_favors,
            'agreement': gpu1_favors == gpu2_favors,
            'gpu1_dot_difference': gpu1_diff,
            'gpu2_dot_difference': gpu2_diff,
            'gpu1_top1_dot': gpu1_top1_dot,
            'gpu1_top2_dot': gpu1_top2_dot,
            'gpu2_top1_dot': gpu2_top1_dot,
            'gpu2_top2_dot': gpu2_top2_dot,
        }
    
    # Find first stage where they disagree
    first_divergence = None
    first_divergence_layer = None
    first_divergence_submodule = None
    
    for stage_name in ordered_stages:
        if stage_name not in comparison:
            continue
        
        if not comparison[stage_name]['agreement']:
            first_divergence = stage_name
            
            # Parse layer and submodule info
            if stage_name.startswith('layer') and stage_name not in ['last_layer_before_norm']:
                parts = stage_name.split('_', 1)
                layer_part = parts[0]  # e.g., 'layer0'
                submodule_part = parts[1] if len(parts) > 1 else ''  # e.g., 'input_layernorm'
                
                # Extract layer number
                try:
                    first_divergence_layer = int(layer_part.replace('layer', ''))
                    first_divergence_submodule = submodule_part
                except:
                    pass
            
            break
    
    print(f"\n{'KEY FINDINGS':^80}")
    print(f"{'-'*80}")
    
    if first_divergence:
        comp_data = comparison[first_divergence]
        print(f"✗ First divergence appears at: {first_divergence}")
        if first_divergence_layer is not None:
            print(f"  Layer: {first_divergence_layer}")
            print(f"  Submodule: {first_divergence_submodule}")
        
        print(f"\n  GPU1 favors: {comp_data['gpu1_favors']}")
        print(f"    Dot with '{ref_top1}': {comp_data['gpu1_top1_dot']:+.6f}")
        print(f"    Dot with '{ref_top2}': {comp_data['gpu1_top2_dot']:+.6f}")
        print(f"    Difference (Top1 - Top2): {comp_data['gpu1_dot_difference']:+.6f}")
        
        print(f"\n  GPU2 favors: {comp_data['gpu2_favors']}")
        print(f"    Dot with '{ref_top1}': {comp_data['gpu2_top1_dot']:+.6f}")
        print(f"    Dot with '{ref_top2}': {comp_data['gpu2_top2_dot']:+.6f}")
        print(f"    Difference (Top1 - Top2): {comp_data['gpu2_dot_difference']:+.6f}")
        
        # Show context: what came before?
        print(f"\n  Context - stages immediately before divergence:")
        stage_list = [s for s in ordered_stages if s in comparison]
        first_idx = stage_list.index(first_divergence)
        
        for i in range(max(0, first_idx - 3), first_idx):
            prev_stage = stage_list[i]
            prev_comp = comparison[prev_stage]
            print(f"    {prev_stage:<50}")
            print(f"      GPU1: {prev_comp['gpu1_favors']:<8} (diff: {prev_comp['gpu1_dot_difference']:+.6f})")
            print(f"      GPU2: {prev_comp['gpu2_favors']:<8} (diff: {prev_comp['gpu2_dot_difference']:+.6f})")
            print(f"      {'✓ agree' if prev_comp['agreement'] else '✗ differ'}")
        
    else:
        print(f"✓ Both GPUs favor the same token at ALL stages!")
        print(f"  (Divergence must come from tie-breaking or numerical precision in final logit computation)")
        
        # Show the final stages to see how close it was
        print(f"\n  Final stages comparison:")
        final_stages = ['layer31_mlp', 'last_layer_before_norm', 'final_norm']
        for stage_name in final_stages:
            if stage_name in comparison:
                comp_data = comparison[stage_name]
                print(f"\n  {stage_name}:")
                print(f"    GPU1: {comp_data['gpu1_favors']}")
                print(f"      Dot with '{ref_top1}': {comp_data['gpu1_top1_dot']:+.6f}")
                print(f"      Dot with '{ref_top2}': {comp_data['gpu1_top2_dot']:+.6f}")
                print(f"      Difference: {comp_data['gpu1_dot_difference']:+.6f}")
                print(f"    GPU2: {comp_data['gpu2_favors']}")
                print(f"      Dot with '{ref_top1}': {comp_data['gpu2_top1_dot']:+.6f}")
                print(f"      Dot with '{ref_top2}': {comp_data['gpu2_top2_dot']:+.6f}")
                print(f"      Difference: {comp_data['gpu2_dot_difference']:+.6f}")
    
    comparison_summary = {
        'first_divergence_stage': first_divergence,
        'first_divergence_layer': first_divergence_layer,
        'first_divergence_submodule': first_divergence_submodule,
        'detailed_comparison': comparison
    }
    
    return comparison_summary

def main():
    parser = argparse.ArgumentParser(description="Comprehensive layer-by-layer divergence localization")
    parser.add_argument("gpu1_dir", type=str, help="Path to GPU1 exp8_part1 results")
    parser.add_argument("gpu2_dir", type=str, help="Path to GPU2 exp8_part1 results")
    parser.add_argument("divergence_file", type=str, help="Path to exp8_part2 divergence_analysis.json")
    parser.add_argument("--model_path", type=str, 
                       default="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct",
                       help="Path to model")
    parser.add_argument("--output_file", type=str, 
                       default="exp8_part6_comprehensive_layer_divergence_analysis.json")
    parser.add_argument("--verbose", action="store_true", help="Print detailed stage-by-stage comparison")
    
    args = parser.parse_args()
    
    gpu1_dir = Path(args.gpu1_dir)
    gpu2_dir = Path(args.gpu2_dir)
    
    # Load model to get unembedding matrix and number of layers
    unembedding_matrix, tokenizer, num_layers = load_model_for_unembedding(args.model_path)
    
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
            tokenizer=tokenizer,
            num_layers=num_layers
        )
        
        if not results:
            continue
        
        # Print detailed comparison if verbose
        if args.verbose:
            print_detailed_comparison(results, num_layers)
        
        # Compare GPUs and find flip point
        comparison_summary = compare_gpus_and_find_flip_point(results, num_layers)
        
        results['comparison_summary'] = comparison_summary
        all_results[question_key] = results
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to {args.output_file}")
    print(f"{'='*80}")
    
    # Summary across all questions
    print(f"\n{'SUMMARY ACROSS ALL QUESTIONS':^80}")
    print(f"{'='*80}")
    
    first_divergence_counts = {}
    layer_divergence_counts = {}
    submodule_divergence_counts = {}
    no_divergence_count = 0
    
    for question_key, result in all_results.items():
        comp_summary = result['comparison_summary']
        
        first_div = comp_summary['first_divergence_stage']
        first_layer = comp_summary['first_divergence_layer']
        first_submod = comp_summary['first_divergence_submodule']
        
        if first_div:
            # Count by full stage name
            first_divergence_counts[first_div] = first_divergence_counts.get(first_div, 0) + 1
            
            # Count by layer
            if first_layer is not None:
                layer_divergence_counts[first_layer] = layer_divergence_counts.get(first_layer, 0) + 1
            
            # Count by submodule type
            if first_submod:
                submodule_divergence_counts[first_submod] = submodule_divergence_counts.get(first_submod, 0) + 1
        else:
            no_divergence_count += 1
    
    print("\nFirst divergence by layer:")
    for layer in sorted(layer_divergence_counts.keys()):
        count = layer_divergence_counts[layer]
        print(f"  Layer {layer:<3}: {count} question(s)")
    
    print("\nFirst divergence by submodule type:")
    for submodule in sorted(submodule_divergence_counts.keys()):
        count = submodule_divergence_counts[submodule]
        print(f"  {submodule:<30}: {count} question(s)")
    
    print("\nTop 10 most common first divergence points:")
    sorted_divs = sorted(first_divergence_counts.items(), key=lambda x: x[1], reverse=True)
    for stage, count in sorted_divs[:10]:
        print(f"  {stage:<50}: {count} question(s)")
    
    if no_divergence_count > 0:
        print(f"\n  {'No divergence at any stage':<50}: {no_divergence_count} question(s)")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

"""
Example usage:

python exp8_part6_comprehensive_layer_divergence_fixed.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A5000x2_exp8_part5_comprehensive_2025-11-02_12-49-05" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A6000_exp8_part5_comprehensive_2025-11-02_12-55-07" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_comprehensive_divergence_analysis.json" \
    --output_file exp8_part6_comprehensive_layer_divergence_analysis.json \
    --verbose

Without verbose:
python exp8_part6_comprehensive_layer_divergence_fixed.py \
    "path/to/gpu1/results" \
    "path/to/gpu2/results" \
    "path/to/divergence_analysis.json"
"""

"""
Example usage:

python exp8_part6_comprehensive_layer_divergence.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A5000x2_exp8_part5_comprehensive_2025-11-02_12-49-05" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A6000_exp8_part5_comprehensive_2025-11-02_12-55-07" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_comprehensive_divergence_analysis.json" \
    --output_file exp8_part6_comprehensive_layer_divergence_analysis.json \
    --verbose

Without verbose:
python exp8_part6_comprehensive_layer_divergence.py \
    "path/to/gpu1/results" \
    "path/to/gpu2/results" \
    "path/to/divergence_analysis.json"
"""