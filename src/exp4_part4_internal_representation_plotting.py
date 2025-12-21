"""
Experiment 4 Part 4: Representation Comparison Visualization

This script creates visualization plots comparing hidden representations from
different GPU runs, showing how similarity evolves across generation steps and
highlighting divergence points.

Purpose:
--------
Generates publication-quality visualizations showing:
1. Cosine similarity across all generation positions
2. L2 distance between representations across positions
3. Visual identification of divergence points
4. Trends in representation similarity over generation


Input:
------
Requires two .npy files containing representations from different GPU runs:
- repr_file1: representations.npy from GPU 0 run
- repr_file2: representations.npy from GPU 1 run
- Optional: divergence_idx from part2 analysis

Plot Structure:
---------------
Creates a 2-subplot figure:
- Top plot: Cosine similarity across positions (y-axis: similarity, x-axis: position)
- Bottom plot: L2 distance across positions (y-axis: distance, x-axis: position)

If divergence_idx provided:
- Plots are truncated at divergence point
- Vertical red line marks divergence
- Helps visualize how similarity degrades before divergence

Methodology:
------------
1. Load representation arrays from both GPU runs
2. For each token position:
   a. Compute cosine similarity: 1 - cosine_distance(repr1[i], repr2[i])
   b. Compute L2 distance: ||repr1[i] - repr2[i]||
3. Plot both metrics across all positions
4. Mark divergence point if provided
5. Save high-resolution plot

Use Case:
---------
Use this script to:
- Visualize how GPU-specific differences evolve during generation
- Identify if representations drift gradually or diverge suddenly
- Understand the relationship between representation similarity and output divergence
- Create figures for papers/presentations

Interpretation:
---------------
High cosine similarity (close to 1.0):
- Representations are nearly identical
- GPUs producing similar outputs

Low cosine similarity (< 0.99):
- Representations are diverging
- Likely to produce different tokens soon

Increasing L2 distance:
- Representations drifting apart
- Accumulation of numerical differences

Sharp drop at divergence:
- Sudden change in representation space
- May indicate token flip causing trajectory change

Dependencies:
-------------
- numpy, matplotlib, scipy

Key Functions:
--------------
- plot_representation_comparison(): Main plotting function
  * Computes similarity metrics
  * Creates dual-subplot visualization
  * Handles divergence marking

Output:
-------
- High-resolution PNG plot (default: repr_comparison.png)
- Shows cosine similarity and L2 distance trends
- Marked divergence point (if provided)

Usage Example:
--------------
```python
plot_representation_comparison(
    'results/exp4_gpu0/question_0/representations.npy',
    'results/exp4_gpu1/question_0/representations.npy',
    output_file='comparison_q0.png',
    divergence_idx=15  # From part2 analysis
)
```

Note:
-----
These visualizations are crucial for understanding whether divergences are:
- Gradual (accumulation of small differences)
- Sudden (catastrophic divergence at specific token)
- Predictable (consistent patterns across questions)
"""

import numpy as np
import os
from datetime import datetime
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 4 Part 4: Create visualization plots comparing hidden representations from different GPU runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'repr_file1', 
        type=str, 
        help='Path to the first representations file (e.g., from GPU 1 run).'
    )
    parser.add_argument(
        'repr_file2', 
        type=str, 
        help='Path to the second representations file (e.g., from GPU 2 run).'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='repr_comparison.png', 
        help='Name for the output plot file. Default: repr_comparison.png'
    )
    parser.add_argument(
        '--divergence_idx', 
        type=int, 
        default=None, 
        help='(Optional) The token index where divergence occurred. A vertical line will be added to the plot.'
    )

    args = parser.parse_args()

    # Create a timestamped directory for the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp4_part4_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(exp_dir, args.output_file)

    plot_representation_comparison(
        args.repr_file1,
        args.repr_file2,
        output_file=output_file_path,
        divergence_idx=args.divergence_idx
    )
"""
python3 src/exp4_part4_internal_representation_plotting.py \\
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp4_gpu0/question_01/representations.npy" \\
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/exp4_gpu1/question_01/representations.npy" \\
    --output_file "exp4_q1_reps_comparison.png" \\
    --divergence_idx 52
"""