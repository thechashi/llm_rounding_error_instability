import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import sys

def plot_representation_comparison(repr_file1, repr_file2, output_file='repr_comparison.png', divergence_idx=None):
    """Plot cosine similarity and L2 distance across all positions"""
    
    # Load representations
    repr1 = np.load(repr_file1)
    repr2 = np.load(repr_file2)
    
    num_positions = min(len(repr1), len(repr2))
    positions = np.arange(num_positions)
    
    # Calculate metrics for each position
    cosine_sims = []
    l2_distances = []
    
    print(f"Calculating metrics for {num_positions} positions...")
    for i in range(num_positions):
        # Cosine similarity
        cos_sim = 1 - cosine(repr1[i], repr2[i])
        cosine_sims.append(cos_sim)
        
        # L2 distance
        l2_dist = np.linalg.norm(repr1[i] - repr2[i])
        l2_distances.append(l2_dist)
    
    cosine_sims = np.array(cosine_sims)
    l2_distances = np.array(l2_distances)
    
    # Truncate data if divergence index is specified
    if divergence_idx is not None and 0 <= divergence_idx < num_positions:
        plot_positions = positions[:divergence_idx+1]
        plot_cosine_sims = cosine_sims[:divergence_idx+1]
        plot_l2_distances = l2_distances[:divergence_idx+1]
        print(f"\nPlotting up to divergence index: {divergence_idx}")
    else:
        plot_positions = positions
        plot_cosine_sims = cosine_sims
        plot_l2_distances = l2_distances
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Cosine Similarity
    ax1.plot(plot_positions, plot_cosine_sims, linewidth=1, color='blue', alpha=0.7)
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    title_suffix = f' (Up to Divergence @ {divergence_idx})' if divergence_idx is not None else ''
    ax1.set_title(f'Cosine Similarity Across Token Positions{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(plot_positions))
    
    # Add statistics text box for cosine similarity
    stats_text1 = f'Mean: {np.mean(plot_cosine_sims):.10f}\nStd: {np.std(plot_cosine_sims):.10f}\nMin: {np.min(plot_cosine_sims):.10f}\nMax: {np.max(plot_cosine_sims):.10f}'
    ax1.text(0.02, 0.02, stats_text1, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add divergence line if specified
    if divergence_idx is not None and 0 <= divergence_idx < num_positions:
        ax1.axvline(x=divergence_idx, color='red', linestyle='--', linewidth=2, label='Divergence')
        # Annotate with cosine similarity at divergence
        cos_at_div = cosine_sims[divergence_idx]
        ax1.annotate(f'Divergence @ {divergence_idx}\nCos Sim: {cos_at_div:.10f}',
                    xy=(divergence_idx, cos_at_div), xytext=(10, 20),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
        ax1.legend(loc='best')
    
    # Plot 2: L2 Distance
    ax2.plot(plot_positions, plot_l2_distances, linewidth=1, color='red', alpha=0.7)
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('L2 Distance', fontsize=12)
    ax2.set_title(f'L2 Distance Across Token Positions{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(plot_positions))
    
    # Add statistics text box for L2 distance
    stats_text2 = f'Mean: {np.mean(plot_l2_distances):.6f}\nStd: {np.std(plot_l2_distances):.6f}\nMin: {np.min(plot_l2_distances):.6f}\nMax: {np.max(plot_l2_distances):.6f}'
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add divergence line if specified
    if divergence_idx is not None and 0 <= divergence_idx < num_positions:
        ax2.axvline(x=divergence_idx, color='red', linestyle='--', linewidth=2, label='Divergence')
        # Annotate with L2 distance at divergence
        l2_at_div = l2_distances[divergence_idx]
        ax2.annotate(f'Divergence @ {divergence_idx}\nL2 Dist: {l2_at_div:.6f}',
                    xy=(divergence_idx, l2_at_div), xytext=(10, -20),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
        ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Number of positions: {num_positions}")
    print(f"\nCosine Similarity:")
    print(f"  Mean:   {np.mean(cosine_sims):.10f}")
    print(f"  Std:    {np.std(cosine_sims):.10f}")
    print(f"  Min:    {np.min(cosine_sims):.10f}")
    print(f"  Max:    {np.max(cosine_sims):.10f}")
    print(f"  Median: {np.median(cosine_sims):.10f}")
    print(f"\nL2 Distance:")
    print(f"  Mean:   {np.mean(l2_distances):.6f}")
    print(f"  Std:    {np.std(l2_distances):.6f}")
    print(f"  Min:    {np.min(l2_distances):.6f}")
    print(f"  Max:    {np.max(l2_distances):.6f}")
    print(f"  Median: {np.median(l2_distances):.6f}")
    
    if divergence_idx is not None and 0 <= divergence_idx < num_positions:
        print(f"\nDivergence Point (Index {divergence_idx}):")
        print(f"  Cosine Similarity: {cosine_sims[divergence_idx]:.10f}")
        print(f"  L2 Distance:       {l2_distances[divergence_idx]:.6f}")
    
    print(f"{'='*60}\n")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_repr.py <repr_file1.npy> <repr_file2.npy> [output.png] [divergence_index]")
        print("Example: python plot_repr.py repr1.npy repr2.npy comparison.png 50")
        sys.exit(1)
    
    repr_file1 = sys.argv[1]
    repr_file2 = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].isdigit() else 'repr_comparison.png'
    divergence_idx = None
    
    # Handle divergence index - could be 3rd or 4th argument
    if len(sys.argv) > 3:
        if sys.argv[3].isdigit():
            divergence_idx = int(sys.argv[3])
        elif len(sys.argv) > 4 and sys.argv[4].isdigit():
            divergence_idx = int(sys.argv[4])
    
    plot_representation_comparison(repr_file1, repr_file2, output_file, divergence_idx)
"""
python3 src/experiment4_part4_rep_com_plot.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A5000_2x24GB/question_01/representations.npy" "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A6000_48GB/question_01/representations.npy" exp4_q1_reps.png 52
"""