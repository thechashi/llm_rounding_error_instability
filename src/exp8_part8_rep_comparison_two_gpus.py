"""
GPU Representation Direct Comparison

For each question, compare representations between GPU1 and GPU2 at ALL stages.
Computes: cosine similarity, L2 distance, number of changed values, percentage changed,
and at what decimal precision they differ (extended to 1e-15).
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    vec1_norm = vec1_flat / (np.linalg.norm(vec1_flat) + 1e-10)
    vec2_norm = vec2_flat / (np.linalg.norm(vec2_flat) + 1e-10)
    return float(np.dot(vec1_norm, vec2_norm))

def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance"""
    return float(np.linalg.norm(vec1.flatten() - vec2.flatten()))

def count_changed_values(vec1: np.ndarray, vec2: np.ndarray, tolerance: float = 1e-10) -> Tuple[int, float]:
    """
    Count how many values differ between two arrays.
    Returns: (num_changed, percentage_changed)
    """
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    # Count values that differ beyond tolerance
    diff = np.abs(vec1_flat - vec2_flat)
    num_changed = np.sum(diff > tolerance)
    total = len(vec1_flat)
    percentage = (num_changed / total) * 100.0 if total > 0 else 0.0
    
    return int(num_changed), float(percentage)

def analyze_decimal_precision(vec1: np.ndarray, vec2: np.ndarray) -> Dict:
    """
    Analyze at what decimal precision the values differ.
    Returns dict with counts at different decimal places from 1e-1 to 1e-15.
    """
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    diff = np.abs(vec1_flat - vec2_flat)
    
    # Filter out exact matches
    nonzero_diff = diff[diff > 0]
    
    if len(nonzero_diff) == 0:
        return {
            'exact_matches': len(vec1_flat),
            'total_values': len(vec1_flat)
        }
    
    # Categorize by decimal precision where difference appears
    # Extended from 1e-7 to 1e-15
    precision_counts = {
        'differs_at_1e-1': int(np.sum(nonzero_diff >= 1e-1)),    # 0.X
        'differs_at_1e-2': int(np.sum(nonzero_diff >= 1e-2)),    # 0.0X
        'differs_at_1e-3': int(np.sum(nonzero_diff >= 1e-3)),    # 0.00X
        'differs_at_1e-4': int(np.sum(nonzero_diff >= 1e-4)),    # 0.000X
        'differs_at_1e-5': int(np.sum(nonzero_diff >= 1e-5)),    # 0.0000X
        'differs_at_1e-6': int(np.sum(nonzero_diff >= 1e-6)),    # 0.00000X
        'differs_at_1e-7': int(np.sum(nonzero_diff >= 1e-7)),    # 0.000000X
        'differs_at_1e-8': int(np.sum(nonzero_diff >= 1e-8)),    # 0.0000000X
        'differs_at_1e-9': int(np.sum(nonzero_diff >= 1e-9)),    # 0.00000000X
        'differs_at_1e-10': int(np.sum(nonzero_diff >= 1e-10)),  # 0.000000000X
        'differs_at_1e-11': int(np.sum(nonzero_diff >= 1e-11)),  # 0.0000000000X
        'differs_at_1e-12': int(np.sum(nonzero_diff >= 1e-12)),  # 0.00000000000X
        'differs_at_1e-13': int(np.sum(nonzero_diff >= 1e-13)),  # 0.000000000000X
        'differs_at_1e-14': int(np.sum(nonzero_diff >= 1e-14)),  # 0.0000000000000X
        'differs_at_1e-15': int(np.sum(nonzero_diff >= 1e-15)),  # 0.00000000000000X
        'differs_below_1e-15': int(np.sum(nonzero_diff < 1e-15)),  # Extremely small differences
    }
    
    precision_counts.update({
        'exact_matches': int(np.sum(diff == 0)),
        'total_values': len(vec1_flat),
        'min_diff': float(np.min(nonzero_diff)),
        'max_diff': float(np.max(nonzero_diff)),
        'mean_diff': float(np.mean(nonzero_diff)),
        'median_diff': float(np.median(nonzero_diff))
    })
    
    return precision_counts

def get_all_stage_names(num_layers: int) -> List[str]:
    """Generate list of all stage names"""
    stages = ['input_embeddings']
    
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
    
    for layer_idx in range(num_layers):
        for submodule in submodules:
            stages.append(f'layer{layer_idx}_{submodule}')
    
    stages.extend(['last_layer_before_norm', 'final_norm'])
    
    return stages

def load_representation(question_dir: Path, stage_name: str, token_idx: int) -> np.ndarray:
    """Load a specific representation from file"""
    if stage_name == 'input_embeddings':
        filepath = question_dir / "input_embeddings.npy"
    elif stage_name in ['last_layer_before_norm', 'final_norm']:
        filepath = question_dir / f"{stage_name}.npy"
    else:
        filepath = question_dir / f"{stage_name}.npy"
    
    if not filepath.exists():
        return None
    
    try:
        data = np.load(filepath)
        return data[token_idx]
    except Exception as e:
        return None

def compare_representations_at_stage(
    gpu1_rep: np.ndarray,
    gpu2_rep: np.ndarray,
    stage_name: str
) -> Dict:
    """Compare two representations and compute all metrics"""
    
    # Check if shapes match
    if gpu1_rep.shape != gpu2_rep.shape:
        return {
            'error': 'shape_mismatch',
            'gpu1_shape': gpu1_rep.shape,
            'gpu2_shape': gpu2_rep.shape
        }
    
    # Compute all metrics
    cos_sim = cosine_similarity(gpu1_rep, gpu2_rep)
    l2_dist = l2_distance(gpu1_rep, gpu2_rep)
    num_changed, pct_changed = count_changed_values(gpu1_rep, gpu2_rep)
    decimal_analysis = analyze_decimal_precision(gpu1_rep, gpu2_rep)
    
    return {
        'stage': stage_name,
        'shape': list(gpu1_rep.shape),
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'num_values_changed': num_changed,
        'percentage_changed': pct_changed,
        'decimal_precision_analysis': decimal_analysis
    }

def analyze_question(
    gpu1_dir: Path,
    gpu2_dir: Path,
    question_id: int,
    token_idx: int,
    num_layers: int
) -> Dict:
    """Analyze all stages for a single question at a specific token index"""
    
    print(f"\n{'='*80}")
    print(f"Question {question_id}: Analyzing token index {token_idx}")
    print(f"{'='*80}")
    
    gpu1_question_dir = gpu1_dir / f"question_{question_id:02d}"
    gpu2_question_dir = gpu2_dir / f"question_{question_id:02d}"
    
    # Load words to show context
    with open(gpu1_question_dir / "words.json", 'r') as f:
        gpu1_words = json.load(f)
    with open(gpu2_question_dir / "words.json", 'r') as f:
        gpu2_words = json.load(f)
    
    if token_idx < len(gpu1_words['generated_words']):
        token_at_idx = gpu1_words['generated_words'][token_idx]
        print(f"Token at index {token_idx}: '{token_at_idx}'")
    
    results = {
        'question_id': question_id,
        'token_index': token_idx,
        'stages': {}
    }
    
    all_stages = get_all_stage_names(num_layers)
    
    processed = 0
    skipped = 0
    
    for stage_name in all_stages:
        # Load representations from both GPUs
        gpu1_rep = load_representation(gpu1_question_dir, stage_name, token_idx)
        gpu2_rep = load_representation(gpu2_question_dir, stage_name, token_idx)
        
        if gpu1_rep is None or gpu2_rep is None:
            skipped += 1
            continue
        
        # Compare
        comparison = compare_representations_at_stage(gpu1_rep, gpu2_rep, stage_name)
        
        if 'error' not in comparison:
            results['stages'][stage_name] = comparison
            processed += 1
        else:
            skipped += 1
    
    print(f"Processed {processed} stages, skipped {skipped} stages")
    
    return results

def print_summary(results: Dict):
    """Print a readable summary of the comparison"""
    
    print(f"\n{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Stage':<50} {'Cosine Sim':<12} {'L2 Dist':<12} {'% Changed':<10}")
    print(f"{'-'*80}")
    
    for stage_name, stage_data in results['stages'].items():
        cos_sim = stage_data['cosine_similarity']
        l2_dist = stage_data['l2_distance']
        pct_changed = stage_data['percentage_changed']
        
        print(f"{stage_name:<50} {cos_sim:>11.9f} {l2_dist:>11.6f} {pct_changed:>9.2f}%")
    
    # Find stages with most difference
    print(f"\n{'STAGES WITH MOST DIFFERENCE':^80}")
    print(f"{'-'*80}")
    
    sorted_by_l2 = sorted(
        results['stages'].items(),
        key=lambda x: x[1]['l2_distance'],
        reverse=True
    )
    
    print("\nTop 10 by L2 distance:")
    for stage_name, stage_data in sorted_by_l2[:10]:
        print(f"  {stage_name:<50} L2={stage_data['l2_distance']:.6f}")
    
    print("\nTop 10 by percentage changed:")
    sorted_by_pct = sorted(
        results['stages'].items(),
        key=lambda x: x[1]['percentage_changed'],
        reverse=True
    )
    for stage_name, stage_data in sorted_by_pct[:10]:
        pct = stage_data['percentage_changed']
        num = stage_data['num_values_changed']
        print(f"  {stage_name:<50} {pct:.2f}% ({num} values)")

def print_decimal_analysis(results: Dict):
    """Print analysis of decimal precision differences (extended to 1e-15)"""
    
    print(f"\n{'DECIMAL PRECISION ANALYSIS':^80}")
    print(f"{'='*80}")
    
    # Aggregate across all stages
    precision_levels = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-7', 
                        '1e-8', '1e-9', '1e-10', '1e-11', '1e-12', '1e-13', 
                        '1e-14', '1e-15']
    
    aggregated = {level: 0 for level in precision_levels}
    aggregated['below_1e-15'] = 0
    
    for stage_data in results['stages'].values():
        dec_analysis = stage_data['decimal_precision_analysis']
        for level in precision_levels:
            aggregated[level] += dec_analysis.get(f'differs_at_{level}', 0)
        aggregated['below_1e-15'] += dec_analysis.get('differs_below_1e-15', 0)
    
    print("\nAggregated across all stages:")
    for level in precision_levels:
        count = aggregated[level]
        print(f"  Differs at {level} or larger: {count:,} values")
    print(f"  Differs below 1e-15:        {aggregated['below_1e-15']:,} values")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(results: Dict, output_prefix: str = "gpu_comparison"):
    """
    Create comprehensive visualizations of GPU comparison results.
    """
    stages = results['stages']
    
    if not stages:
        print("No stages to plot!")
        return
    
    # Extract data
    stage_names = []
    cosine_sims = []
    l2_dists = []
    pct_changed = []
    
    for stage_name, stage_data in stages.items():
        stage_names.append(stage_name)
        cosine_sims.append(stage_data['cosine_similarity'])
        l2_dists.append(stage_data['l2_distance'])
        pct_changed.append(stage_data['percentage_changed'])
    
    # Parse layer numbers for coloring
    layer_nums = []
    for name in stage_names:
        if name.startswith('layer') and name not in ['last_layer_before_norm']:
            try:
                layer_num = int(name.split('_')[0].replace('layer', ''))
                layer_nums.append(layer_num)
            except:
                layer_nums.append(-1)
        else:
            layer_nums.append(-1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Cosine Similarity over stages
    ax1 = fig.add_subplot(gs[0, :])
    scatter1 = ax1.scatter(range(len(stage_names)), cosine_sims, 
                          c=layer_nums, cmap='viridis', alpha=0.6, s=20)
    ax1.plot(range(len(stage_names)), cosine_sims, 'b-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Stage Index', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Cosine Similarity Between GPU1 and GPU2 Representations', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(cosine_sims) - 0.0001, 1.0])
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Layer Number', fontsize=10)
    
    # Plot 2: L2 Distance over stages
    ax2 = fig.add_subplot(gs[1, :])
    scatter2 = ax2.scatter(range(len(stage_names)), l2_dists, 
                          c=layer_nums, cmap='viridis', alpha=0.6, s=20)
    ax2.plot(range(len(stage_names)), l2_dists, 'r-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Stage Index', fontsize=12)
    ax2.set_ylabel('L2 Distance', fontsize=12)
    ax2.set_title('L2 Distance Between GPU1 and GPU2 Representations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Layer Number', fontsize=10)
    
    # Plot 3: Percentage Changed
    ax3 = fig.add_subplot(gs[2, 0])
    scatter3 = ax3.scatter(range(len(stage_names)), pct_changed, 
                          c=layer_nums, cmap='viridis', alpha=0.6, s=20)
    ax3.plot(range(len(stage_names)), pct_changed, 'g-', alpha=0.3, linewidth=0.5)
    ax3.set_xlabel('Stage Index', fontsize=12)
    ax3.set_ylabel('Percentage Changed (%)', fontsize=12)
    ax3.set_title('Percentage of Values Changed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Layer Number', fontsize=10)
    
    # Plot 4: Combined metric (inverse cosine similarity * L2 distance)
    ax4 = fig.add_subplot(gs[2, 1])
    combined_metric = [(1 - cs) * l2 for cs, l2 in zip(cosine_sims, l2_dists)]
    scatter4 = ax4.scatter(range(len(stage_names)), combined_metric, 
                          c=layer_nums, cmap='viridis', alpha=0.6, s=20)
    ax4.plot(range(len(stage_names)), combined_metric, 'm-', alpha=0.3, linewidth=0.5)
    ax4.set_xlabel('Stage Index', fontsize=12)
    ax4.set_ylabel('Combined Divergence Metric', fontsize=12)
    ax4.set_title('Combined Divergence: (1 - Cosine) × L2', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Layer Number', fontsize=10)
    
    plt.suptitle(f'Question {results["question_id"]} - Token {results["token_index"]}: GPU Representation Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_prefix}_q{results["question_id"]}_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved overview plot: {output_prefix}_q{results['question_id']}_overview.png")
    plt.close()
    
    # Create per-layer breakdown plot
    plot_per_layer_breakdown(results, output_prefix)
    
    # Create decimal precision plot (extended)
    plot_decimal_precision(results, output_prefix)

def plot_per_layer_breakdown(results: Dict, output_prefix: str):
    """
    Create a detailed breakdown showing different submodules per layer.
    """
    stages = results['stages']
    
    # Organize by layer and submodule
    layer_data = {}
    submodule_types = ['input_layernorm', 'self_attn', 'attn_o_proj', 
                       'post_attention_layernorm', 'mlp_gate_proj', 
                       'mlp_up_proj', 'mlp_act_fn', 'mlp_down_proj', 'mlp']
    
    for stage_name, stage_data in stages.items():
        if stage_name.startswith('layer') and stage_name not in ['last_layer_before_norm']:
            parts = stage_name.split('_', 1)
            if len(parts) == 2:
                layer_part = parts[0]
                submodule_part = parts[1]
                
                try:
                    layer_num = int(layer_part.replace('layer', ''))
                    if layer_num not in layer_data:
                        layer_data[layer_num] = {}
                    layer_data[layer_num][submodule_part] = stage_data
                except:
                    pass
    
    if not layer_data:
        return
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Prepare data matrices
    layers = sorted(layer_data.keys())
    
    cosine_matrix = []
    l2_matrix = []
    pct_matrix = []
    
    for layer_num in layers:
        cosine_row = []
        l2_row = []
        pct_row = []
        
        for submod in submodule_types:
            if submod in layer_data[layer_num]:
                data = layer_data[layer_num][submod]
                cosine_row.append(data['cosine_similarity'])
                l2_row.append(data['l2_distance'])
                pct_row.append(data['percentage_changed'])
            else:
                cosine_row.append(np.nan)
                l2_row.append(np.nan)
                pct_row.append(np.nan)
        
        cosine_matrix.append(cosine_row)
        l2_matrix.append(l2_row)
        pct_matrix.append(pct_row)
    
    cosine_matrix = np.array(cosine_matrix)
    l2_matrix = np.array(l2_matrix)
    pct_matrix = np.array(pct_matrix)
    
    # Plot 1: Cosine Similarity Heatmap
    im1 = axes[0, 0].imshow(cosine_matrix.T, aspect='auto', cmap='RdYlGn', 
                            vmin=0.999, vmax=1.0, interpolation='nearest')
    axes[0, 0].set_xlabel('Layer Number', fontsize=12)
    axes[0, 0].set_ylabel('Submodule', fontsize=12)
    axes[0, 0].set_title('Cosine Similarity by Layer and Submodule', fontsize=12, fontweight='bold')
    axes[0, 0].set_yticks(range(len(submodule_types)))
    axes[0, 0].set_yticklabels(submodule_types, fontsize=8)
    axes[0, 0].set_xticks(range(0, len(layers), 4))
    axes[0, 0].set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: L2 Distance Heatmap (log scale)
    l2_log = np.log10(l2_matrix + 1e-10)
    im2 = axes[0, 1].imshow(l2_log.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0, 1].set_xlabel('Layer Number', fontsize=12)
    axes[0, 1].set_ylabel('Submodule', fontsize=12)
    axes[0, 1].set_title('L2 Distance (log10) by Layer and Submodule', fontsize=12, fontweight='bold')
    axes[0, 1].set_yticks(range(len(submodule_types)))
    axes[0, 1].set_yticklabels(submodule_types, fontsize=8)
    axes[0, 1].set_xticks(range(0, len(layers), 4))
    axes[0, 1].set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
    plt.colorbar(im2, ax=axes[0, 1], label='log10(L2 Distance)')
    
    # Plot 3: Percentage Changed Heatmap
    im3 = axes[1, 0].imshow(pct_matrix.T, aspect='auto', cmap='Blues', 
                            vmin=0, vmax=100, interpolation='nearest')
    axes[1, 0].set_xlabel('Layer Number', fontsize=12)
    axes[1, 0].set_ylabel('Submodule', fontsize=12)
    axes[1, 0].set_title('Percentage Changed by Layer and Submodule', fontsize=12, fontweight='bold')
    axes[1, 0].set_yticks(range(len(submodule_types)))
    axes[1, 0].set_yticklabels(submodule_types, fontsize=8)
    axes[1, 0].set_xticks(range(0, len(layers), 4))
    axes[1, 0].set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
    plt.colorbar(im3, ax=axes[1, 0], label='Percentage (%)')
    
    # Plot 4: Average metrics per layer
    layer_avg_l2 = np.nanmean(l2_matrix, axis=1)
    layer_avg_cos = np.nanmean(cosine_matrix, axis=1)
    layer_avg_pct = np.nanmean(pct_matrix, axis=1)
    
    ax4_1 = axes[1, 1]
    ax4_2 = ax4_1.twinx()
    
    ln1 = ax4_1.plot(layers, layer_avg_l2, 'r-o', label='Avg L2 Distance', linewidth=2, markersize=4)
    ax4_1.set_xlabel('Layer Number', fontsize=12)
    ax4_1.set_ylabel('Average L2 Distance', fontsize=12, color='r')
    ax4_1.tick_params(axis='y', labelcolor='r')
    ax4_1.set_yscale('log')
    ax4_1.grid(True, alpha=0.3)
    
    ln2 = ax4_2.plot(layers, 1 - layer_avg_cos, 'b-s', label='Avg (1 - Cosine)', linewidth=2, markersize=4)
    ax4_2.set_ylabel('Average (1 - Cosine Similarity)', fontsize=12, color='b')
    ax4_2.tick_params(axis='y', labelcolor='b')
    ax4_2.set_yscale('log')
    
    axes[1, 1].set_title('Layer-wise Divergence Trend', fontsize=12, fontweight='bold')
    
    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax4_1.legend(lns, labs, loc='upper left')
    
    plt.suptitle(f'Question {results["question_id"]} - Token {results["token_index"]}: Per-Layer Breakdown', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_prefix}_q{results["question_id"]}_layer_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved layer breakdown plot: {output_prefix}_q{results['question_id']}_layer_breakdown.png")
    plt.close()

def plot_decimal_precision(results: Dict, output_prefix: str):
    """
    Visualize the decimal precision analysis (extended to 1e-15).
    """
    stages = results['stages']
    
    # Aggregate decimal precision data
    precision_levels = ['differs_at_1e-1', 'differs_at_1e-2', 'differs_at_1e-3', 
                       'differs_at_1e-4', 'differs_at_1e-5', 'differs_at_1e-6', 
                       'differs_at_1e-7', 'differs_at_1e-8', 'differs_at_1e-9',
                       'differs_at_1e-10', 'differs_at_1e-11', 'differs_at_1e-12',
                       'differs_at_1e-13', 'differs_at_1e-14', 'differs_at_1e-15']
    
    aggregated = {level: 0 for level in precision_levels}
    aggregated['exact_matches'] = 0
    aggregated['differs_below_1e-15'] = 0
    
    for stage_data in stages.values():
        dec_analysis = stage_data['decimal_precision_analysis']
        for level in precision_levels:
            aggregated[level] += dec_analysis.get(level, 0)
        aggregated['exact_matches'] += dec_analysis.get('exact_matches', 0)
        aggregated['differs_below_1e-15'] += dec_analysis.get('differs_below_1e-15', 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Bar chart of precision levels
    labels = ['≥1e-1', '≥1e-2', '≥1e-3', '≥1e-4', '≥1e-5', '≥1e-6', '≥1e-7',
              '≥1e-8', '≥1e-9', '≥1e-10', '≥1e-11', '≥1e-12', '≥1e-13', '≥1e-14', '≥1e-15']
    values = [aggregated[level] for level in precision_levels]
    colors = plt.cm.Reds(np.linspace(0.9, 0.3, len(labels)))
    
    axes[0].bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Difference Magnitude', fontsize=12)
    axes[0].set_ylabel('Number of Values', fontsize=12)
    axes[0].set_title('Distribution of Difference Magnitudes (1e-1 to 1e-15)', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (label, value) in enumerate(zip(labels, values)):
        if value > 0:
            axes[0].text(i, value, f'{value:,}', ha='center', va='bottom', fontsize=7)
    
    # Plot 2: Cumulative distribution
    cumulative = []
    running_sum = 0
    for value in values:
        running_sum += value
        cumulative.append(running_sum)
    
    axes[1].plot(labels, cumulative, 'b-o', linewidth=2, markersize=8)
    axes[1].fill_between(range(len(labels)), cumulative, alpha=0.3)
    axes[1].set_xlabel('Difference Magnitude', fontsize=12)
    axes[1].set_ylabel('Cumulative Count', fontsize=12)
    axes[1].set_title('Cumulative Distribution of Differences', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add total and exact match info
    total_values = running_sum + aggregated['exact_matches']
    fig.text(0.5, 0.02, 
             f"Total values analyzed: {total_values:,} | Exact matches: {aggregated['exact_matches']:,} ({aggregated['exact_matches']/total_values*100:.2f}%) | Below 1e-15: {aggregated['differs_below_1e-15']:,}",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Question {results["question_id"]} - Token {results["token_index"]}: Extended Decimal Precision Analysis (1e-1 to 1e-15)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(f'{output_prefix}_q{results["question_id"]}_decimal_precision.png', dpi=300, bbox_inches='tight')
    print(f"Saved decimal precision plot: {output_prefix}_q{results['question_id']}_decimal_precision.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Direct GPU representation comparison (extended precision)")
    parser.add_argument("gpu1_dir", type=str, help="Path to GPU1 results directory")
    parser.add_argument("gpu2_dir", type=str, help="Path to GPU2 results directory")
    parser.add_argument("--question_ids", type=int, nargs='+', 
                       help="Question IDs to analyze (default: all found)")
    parser.add_argument("--token_idx", type=int, default=None,
                       help="Specific token index to analyze (default: analyze before divergence)")
    parser.add_argument("--divergence_file", type=str, default=None,
                       help="Path to divergence analysis JSON (to auto-select token_idx)")
    parser.add_argument("--num_layers", type=int, default=32,
                       help="Number of transformer layers (default: 32)")
    parser.add_argument("--output_file", type=str,
                       default="exp8_part8_f32_tok199_gpu_representation_comparison_extended.json")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed summaries")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    gpu1_dir = Path(args.gpu1_dir)
    gpu2_dir = Path(args.gpu2_dir)
    
    # Determine which questions to analyze
    if args.question_ids:
        question_ids = args.question_ids
    else:
        # Find all question directories in GPU1
        question_dirs = sorted(gpu1_dir.glob("question_*"))
        question_ids = [int(d.name.split('_')[1]) for d in question_dirs]
    
    print(f"Analyzing {len(question_ids)} questions: {question_ids}")
    
    # Load divergence data if provided
    divergence_data = None
    if args.divergence_file:
        print(f"Loading divergence analysis from {args.divergence_file}")
        with open(args.divergence_file, 'r') as f:
            divergence_data = json.load(f)
    
    all_results = {}
    
    for question_id in question_ids:
        # Determine token index to analyze
        if args.token_idx is not None:
            token_idx = args.token_idx
        elif divergence_data:
            # Use token before divergence
            question_key = f"question_0{question_id}"
            if question_key in divergence_data:
                div_idx = divergence_data[question_key].get('divergence_index')
                if div_idx and div_idx > 0:
                    token_idx = div_idx - 1
                else:
                    print(f"Question {question_id}: No divergence found, skipping")
                    continue
            else:
                print(f"Question {question_id}: Not in divergence data, skipping")
                continue
        else:
            print(f"Question {question_id}: No token_idx specified and no divergence file, skipping")
            continue
        
        # Analyze this question
        results = analyze_question(
            gpu1_dir=gpu1_dir,
            gpu2_dir=gpu2_dir,
            question_id=question_id,
            token_idx=token_idx,
            num_layers=args.num_layers
        )
        
        all_results[f"question_{question_id}"] = results
        
        if args.verbose:
            print_summary(results)
            print_decimal_analysis(results)
        
        # Generate plots if requested
        if args.plot:
            output_prefix = args.output_file.replace('.json', '')
            plot_results(results, output_prefix)
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to {args.output_file}")
    
    if args.plot:
        print(f"Plots saved with prefix: {args.output_file.replace('.json', '')}")

if __name__ == "__main__":
    main()

"""
Example usage:

# Analyze all questions using divergence file
python exp8_part8_rep_comparison_two_gpus.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A5000_exp8_part5_comprehensive_float32_2025-11-04_10-32-08" \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/A6000_exp8_part5_comprehensive_float32_2025-11-04_10-38-04" \
    --divergence_file "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp8_part2_comprehensive_divergence_analysis.json" \
    --token_index 0 \
    --verbose

# Analyze specific questions at specific token index
python gpu_comparison.py \
    "/path/to/gpu1/results" \
    "/path/to/gpu2/results" \
    --question_ids 1 5 10 \
    --token_idx 50 \
    --verbose

# Analyze specific question with detailed output
python gpu_comparison.py \
    "/path/to/gpu1/results" \
    "/path/to/gpu2/results" \
    --question_ids 1 \
    --token_idx 50 \
    --num_layers 32 \
    --output_file "question1_comparison.json" \
    --verbose
"""