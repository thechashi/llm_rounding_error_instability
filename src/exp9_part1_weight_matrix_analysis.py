"""
Experiment 5B: Weight Matrix Statistics and Distribution Analysis

Analyzes the weight matrices of attention projection layers (Q, K, V, O projections)
across all transformer layers, providing fine-grained statistics on weight distributions.

Purpose:
--------
Instead of analyzing activations, this script extracts and analyzes:
1. Q_proj weight matrix statistics
2. K_proj weight matrix statistics
3. V_proj weight matrix statistics
4. O_proj weight matrix statistics

For each projection, we compute:
- Shape and dimensionality info
- Statistical properties (mean, std, min, max, percentiles)
- Distribution characteristics (skewness, kurtosis)
- Magnitude distributions
- Sign distributions
- Frobenius norm and spectral properties (via SVD)

This helps understand:
- Whether weight initialization is uniform across layers
- If certain layers have weights with different magnitudes
- Potential numerical properties of the weight matrices
- Correlations between weight properties and instability

Methodology:
------------
1. Load Llama model in float32
2. Iterate through all transformer layers
3. For each layer, access self_attn module
4. Extract q_proj, k_proj, v_proj, o_proj weight matrices
5. Compute comprehensive statistics for each
6. Save results and create visualizations

Output:
-------
- Per-layer weight statistics (CSV)
- Distribution histograms
- Heatmaps of weight magnitudes across layers
- Summary statistics table
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import pandas as pd
import os
from datetime import datetime

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Load Llama model in float32"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32
    )
    return model, tokenizer

def compute_weight_statistics(weight_matrix):
    """
    Compute comprehensive statistics for a weight matrix.
    
    Args:
        weight_matrix: torch tensor of shape (out_features, in_features)
    
    Returns:
        Dictionary with statistics
    """
    w = weight_matrix.detach().cpu().numpy()
    w_flat = w.flatten()
    w_abs = np.abs(w_flat)
    
    stats_dict = {
        'shape': w.shape,
        'out_features': w.shape[0],
        'in_features': w.shape[1],
        'total_params': w.size,
        
        # Basic statistics
        'mean': np.mean(w_flat),
        'std': np.std(w_flat),
        'var': np.var(w_flat),
        'median': np.median(w_flat),
        'min': np.min(w_flat),
        'max': np.max(w_flat),
        'range': np.ptp(w_flat),
        
        # Absolute value statistics
        'mean_abs': np.mean(w_abs),
        'std_abs': np.std(w_abs),
        'median_abs': np.median(w_abs),
        'min_abs': np.min(w_abs),
        'max_abs': np.max(w_abs),
        
        # Distribution shape
        'skewness': stats.skew(w_flat),
        'kurtosis': stats.kurtosis(w_flat),
        
        # Sign distribution
        'n_positive': np.sum(w_flat > 0),
        'n_negative': np.sum(w_flat < 0),
        'n_zero': np.sum(w_flat == 0),
        'pct_positive': (np.sum(w_flat > 0) / len(w_flat)) * 100,
        'pct_negative': (np.sum(w_flat < 0) / len(w_flat)) * 100,
        'pct_zero': (np.sum(w_flat == 0) / len(w_flat)) * 100,
        
        # Percentiles
        'percentile_1': np.percentile(w_abs, 1),
        'percentile_5': np.percentile(w_abs, 5),
        'percentile_10': np.percentile(w_abs, 10),
        'percentile_25': np.percentile(w_abs, 25),
        'percentile_50': np.percentile(w_abs, 50),
        'percentile_75': np.percentile(w_abs, 75),
        'percentile_90': np.percentile(w_abs, 90),
        'percentile_95': np.percentile(w_abs, 95),
        'percentile_99': np.percentile(w_abs, 99),
        
        # Norms
        'frobenius_norm': np.linalg.norm(w, 'fro'),
        'spectral_norm': np.linalg.norm(w, 2),  # Largest singular value
        'nuclear_norm': np.linalg.norm(w, 'nuc'),  # Sum of singular values
        
        # Magnitude ranges
        'n_geq_1e-1': np.sum(w_abs >= 1e-1),
        'n_1e-2_to_1e-1': np.sum((w_abs >= 1e-2) & (w_abs < 1e-1)),
        'n_1e-3_to_1e-2': np.sum((w_abs >= 1e-3) & (w_abs < 1e-2)),
        'n_1e-4_to_1e-3': np.sum((w_abs >= 1e-4) & (w_abs < 1e-3)),
        'n_lt_1e-4': np.sum(w_abs < 1e-4),
    }
    
    return stats_dict, w

def analyze_all_projections(model, save_dir):
    """
    Extract and analyze all Q, K, V, O projection weights across layers.
    
    Args:
        model: Llama model
        save_dir: Directory to save results
    """
    print("="*80)
    print("ANALYZING WEIGHT MATRICES FOR ATTENTION PROJECTIONS")
    print("="*80)
    
    num_layers = len(model.model.layers)
    projection_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # Store results
    all_results = {proj: [] for proj in projection_types}
    all_weights = {proj: [] for proj in projection_types}
    
    # Iterate through layers
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx}/{num_layers-1}:")
        
        layer = model.model.layers[layer_idx]
        self_attn = layer.self_attn
        
        for proj_name in projection_types:
            # Get the projection module
            if proj_name == 'q_proj':
                proj_module = self_attn.q_proj
            elif proj_name == 'k_proj':
                proj_module = self_attn.k_proj
            elif proj_name == 'v_proj':
                proj_module = self_attn.v_proj
            elif proj_name == 'o_proj':
                proj_module = self_attn.o_proj
            
            # Extract weight matrix
            weight = proj_module.weight  # Shape: (out_features, in_features)
            
            # Compute statistics
            stats_dict, weight_np = compute_weight_statistics(weight)
            
            # Add layer info
            stats_dict['layer'] = layer_idx
            stats_dict['projection'] = proj_name
            
            all_results[proj_name].append(stats_dict)
            all_weights[proj_name].append(weight_np)
            
            print(f"  {proj_name:8s}: shape={stats_dict['shape']}, "
                  f"mean={stats_dict['mean']:.6e}, std={stats_dict['std']:.6e}, "
                  f"frob_norm={stats_dict['frobenius_norm']:.6f}")
    
    return all_results, all_weights

def create_summary_dataframe(all_results):
    """Convert results to pandas DataFrame for easy analysis"""
    dfs = {}
    
    for proj_name, results_list in all_results.items():
        df = pd.DataFrame(results_list)
        # Reorder columns for readability
        cols = ['layer', 'projection', 'shape', 'total_params', 
                'mean', 'std', 'min', 'max', 'mean_abs', 'std_abs',
                'frobenius_norm', 'spectral_norm', 'skewness', 'kurtosis',
                'pct_positive', 'pct_negative']
        df = df[cols]
        dfs[proj_name] = df
    
    return dfs

def plot_weight_statistics(all_results, save_dir):
    """Create visualization plots for weight statistics"""
    print("\nGenerating plots...")
    
    projection_types = list(all_results.keys())
    num_layers = len(all_results[projection_types[0]])
    
    # Extract metrics for plotting
    layers = list(range(num_layers))
    
    # Plot 1: Frobenius norm across layers
    fig, ax = plt.subplots(figsize=(14, 6))
    for proj_name in projection_types:
        frob_norms = [all_results[proj_name][i]['frobenius_norm'] 
                      for i in range(num_layers)]
        ax.plot(layers, frob_norms, marker='o', label=proj_name, linewidth=2)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Frobenius Norm', fontsize=12)
    ax.set_title('Weight Matrix Frobenius Norm Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'frobenius_norms.pdf'), dpi=300, bbox_inches='tight')
    print("Saved: frobenius_norms.pdf")
    plt.close()
    
    # Plot 2: Spectral norm across layers
    fig, ax = plt.subplots(figsize=(14, 6))
    for proj_name in projection_types:
        spec_norms = [all_results[proj_name][i]['spectral_norm'] 
                      for i in range(num_layers)]
        ax.plot(layers, spec_norms, marker='s', label=proj_name, linewidth=2)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Spectral Norm (Largest Singular Value)', fontsize=12)
    ax.set_title('Weight Matrix Spectral Norm Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_norms.pdf'), dpi=300, bbox_inches='tight')
    print("Saved: spectral_norms.pdf")
    plt.close()
    
    # Plot 3: Mean and Std of weights across layers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for proj_name in projection_types:
        means = [all_results[proj_name][i]['mean'] for i in range(num_layers)]
        ax1.plot(layers, means, marker='o', label=proj_name, linewidth=2)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mean Weight Value', fontsize=12)
    ax1.set_title('Mean Weight Value Across Layers', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for proj_name in projection_types:
        stds = [all_results[proj_name][i]['std'] for i in range(num_layers)]
        ax2.plot(layers, stds, marker='s', label=proj_name, linewidth=2)
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Std Dev of Weights', fontsize=12)
    ax2.set_title('Weight Standard Deviation Across Layers', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_std_weights.pdf'), dpi=300, bbox_inches='tight')
    print("Saved: mean_std_weights.pdf")
    plt.close()
    
    # Plot 4: Distribution shapes (skewness and kurtosis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for proj_name in projection_types:
        skews = [all_results[proj_name][i]['skewness'] for i in range(num_layers)]
        ax1.plot(layers, skews, marker='o', label=proj_name, linewidth=2)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Skewness', fontsize=12)
    ax1.set_title('Weight Distribution Skewness Across Layers', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    for proj_name in projection_types:
        kurts = [all_results[proj_name][i]['kurtosis'] for i in range(num_layers)]
        ax2.plot(layers, kurts, marker='s', label=proj_name, linewidth=2)
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Kurtosis', fontsize=12)
    ax2.set_title('Weight Distribution Kurtosis Across Layers', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_shapes.pdf'), dpi=300, bbox_inches='tight')
    print("Saved: distribution_shapes.pdf")
    plt.close()
    
    # Plot 5: Heatmap of mean absolute values per layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, proj_name in enumerate(projection_types):
        means_abs = [all_results[proj_name][i]['mean_abs'] for i in range(num_layers)]
        
        ax = axes[idx]
        bars = ax.bar(layers, means_abs, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Mean Absolute Weight', fontsize=11)
        ax.set_title(f'{proj_name.upper()} - Mean Absolute Weight', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color bars by value
        for i, bar in enumerate(bars):
            if means_abs[i] > np.mean(means_abs) + np.std(means_abs):
                bar.set_color('red')
                bar.set_alpha(0.7)
            elif means_abs[i] < np.mean(means_abs) - np.std(means_abs):
                bar.set_color('blue')
                bar.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_abs_per_layer.pdf'), dpi=300, bbox_inches='tight')
    print("Saved: mean_abs_per_layer.pdf")
    plt.close()

def plot_weight_distributions(all_weights, save_dir):
    """Plot histogram distributions of weight values"""
    print("\nGenerating distribution histograms...")
    
    projection_types = list(all_weights.keys())
    num_layers = len(all_weights[projection_types[0]])
    
    # Sample a few layers for visualization
    sample_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    
    for proj_name in projection_types:
        fig, axes = plt.subplots(1, len(sample_layers), figsize=(18, 4))
        
        for ax_idx, layer_idx in enumerate(sample_layers):
            weights = all_weights[proj_name][layer_idx].flatten()
            
            ax = axes[ax_idx]
            ax.hist(weights, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Weight Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_val = np.mean(weights)
            std_val = np.std(weights)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.4e}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.legend(fontsize=9)
        
        fig.suptitle(f'{proj_name.upper()} - Weight Value Distributions', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'distribution_{proj_name}.pdf'), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: distribution_{proj_name}.pdf")
        plt.close()

def print_comprehensive_summary(all_results):
    """Print detailed summary statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE WEIGHT MATRIX STATISTICS SUMMARY")
    print("="*80)
    
    projection_types = list(all_results.keys())
    num_layers = len(all_results[projection_types[0]])
    
    for proj_name in projection_types:
        print(f"\n{'='*80}")
        print(f"{proj_name.upper()} PROJECTION")
        print(f"{'='*80}")
        
        frob_norms = [all_results[proj_name][i]['frobenius_norm'] for i in range(num_layers)]
        spec_norms = [all_results[proj_name][i]['spectral_norm'] for i in range(num_layers)]
        means = [all_results[proj_name][i]['mean'] for i in range(num_layers)]
        stds = [all_results[proj_name][i]['std'] for i in range(num_layers)]
        skews = [all_results[proj_name][i]['skewness'] for i in range(num_layers)]
        
        print(f"\nFrobenius Norm:")
        print(f"  Mean across layers:   {np.mean(frob_norms):.6f}")
        print(f"  Std across layers:    {np.std(frob_norms):.6f}")
        print(f"  Min (layer {np.argmin(frob_norms)}):           {np.min(frob_norms):.6f}")
        print(f"  Max (layer {np.argmax(frob_norms)}):           {np.max(frob_norms):.6f}")
        
        print(f"\nSpectral Norm:")
        print(f"  Mean across layers:   {np.mean(spec_norms):.6f}")
        print(f"  Std across layers:    {np.std(spec_norms):.6f}")
        print(f"  Min (layer {np.argmin(spec_norms)}):           {np.min(spec_norms):.6f}")
        print(f"  Max (layer {np.argmax(spec_norms)}):           {np.max(spec_norms):.6f}")
        
        print(f"\nMean Weight Value:")
        print(f"  Mean across layers:   {np.mean(means):.6e}")
        print(f"  Std across layers:    {np.std(means):.6e}")
        print(f"  Min (layer {np.argmin(means)}):           {np.min(means):.6e}")
        print(f"  Max (layer {np.argmax(means)}):           {np.max(means):.6e}")
        
        print(f"\nWeight Std Dev:")
        print(f"  Mean across layers:   {np.mean(stds):.6e}")
        print(f"  Std across layers:    {np.std(stds):.6e}")
        print(f"  Min (layer {np.argmin(stds)}):           {np.min(stds):.6e}")
        print(f"  Max (layer {np.argmax(stds)}):           {np.max(stds):.6e}")
        
        print(f"\nSkewness:")
        print(f"  Mean across layers:   {np.mean(skews):.6f}")
        print(f"  Std across layers:    {np.std(skews):.6f}")
        print(f"  Range:                {np.min(skews):.6f} to {np.max(skews):.6f}")

def main():
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("../results", f"exp9_part1_weight_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Load model
    print("[1/6] Loading model...")
    model, tokenizer = load_model()
    
    # Analyze all projections
    print("[2/6] Analyzing weight matrices...")
    all_results, all_weights = analyze_all_projections(model, results_dir)
    
    # Create DataFrames
    print("[3/6] Creating summary DataFrames...")
    dfs = create_summary_dataframe(all_results)
    
    # Save DataFrames
    print("[4/6] Saving results to CSV...")
    for proj_name, df in dfs.items():
        csv_path = os.path.join(results_dir, f'weights_{proj_name}_statistics.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    # Create plots
    print("[5/6] Creating visualizations...")
    plot_weight_statistics(all_results, results_dir)
    plot_weight_distributions(all_weights, results_dir)
    
    # Print summary
    print("[6/6] Printing summary...")
    print_comprehensive_summary(all_results)
    
    # Save raw results as numpy
    print("\nSaving raw weight matrices...")
    for proj_name, weights_list in all_weights.items():
        np_path = os.path.join(results_dir, f'weights_{proj_name}_raw.npz')
        np.savez(np_path, *weights_list)
        print(f"Saved: {np_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
