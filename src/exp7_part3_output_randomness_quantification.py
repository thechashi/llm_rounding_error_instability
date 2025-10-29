"""
Experiment 7 Part 3: Output Randomness Quantification

This script performs comprehensive statistical analysis to quantify the randomness and chaos
in the decision boundary between token predictions. It provides multiple metrics to demonstrate
that small perturbations lead to unpredictable and seemingly random logit flips.

Analysis metrics computed:
1. Flip Frequency: Percentage of adjacent grid cells with different predictions (local instability)
2. Region Fragmentation: Number of disconnected prediction regions (spatial chaos)
3. Spatial Autocorrelation (Moran's I): Measures how predictions correlate with neighbors
4. Local Entropy: Unpredictability in local neighborhoods (max entropy = random)
5. Gradient Magnitude: How rapidly logits change across the space (sensitivity)
6. Zero-Crossing Density: How often the decision boundary is crossed (complexity)
7. Chi-Square Test: Statistical test for spatial randomness

The script generates a randomness score and concludes whether the decision boundary exhibits
strong chaotic/random behavior, providing quantitative evidence of model instability.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import label
import argparse
import os
from datetime import datetime

def load_matrix(filename):
    """Load the saved matrix and axis values"""
    data = np.load(filename)
    grid = data['grid']
    e1_values = data['e1_values']
    e2_values = data['e2_values']
    return grid, e1_values, e2_values

def analyze_decision_boundary_randomness(grid, e1_values, e2_values):
    """Perform statistical analysis to show randomness near decision boundary"""
    
    # Create binary prediction map: 1 where L1 >= L2, 0 where L1 < L2
    binary_grid = (grid >= 0).astype(int)
    
    print("="*80)
    print("DECISION BOUNDARY RANDOMNESS ANALYSIS")
    print("="*80)
    
    # 1. FLIP FREQUENCY: Count how many times predictions flip in adjacent cells
    print("\n1. FLIP FREQUENCY (Local Instability)")
    print("-" * 80)
    
    # Horizontal flips
    h_flips = np.sum(np.abs(np.diff(binary_grid, axis=0)))
    # Vertical flips
    v_flips = np.sum(np.abs(np.diff(binary_grid, axis=1)))
    total_flips = h_flips + v_flips
    
    # Maximum possible flips (if completely random)
    max_h_flips = (grid.shape[0] - 1) * grid.shape[1]
    max_v_flips = grid.shape[0] * (grid.shape[1] - 1)
    max_total_flips = max_h_flips + max_v_flips
    
    flip_ratio = total_flips / max_total_flips
    
    print(f"Horizontal flips: {h_flips} / {max_h_flips} ({h_flips/max_h_flips*100:.2f}%)")
    print(f"Vertical flips: {v_flips} / {max_v_flips} ({v_flips/max_v_flips*100:.2f}%)")
    print(f"Total flips: {total_flips} / {max_total_flips} ({flip_ratio*100:.2f}%)")
    print(f"Interpretation: {flip_ratio*100:.1f}% of adjacent cells have different predictions")
    print(f"              : Higher % = more chaotic/random boundary")
    
    # 2. REGION FRAGMENTATION: Count number of disconnected regions
    print("\n2. REGION FRAGMENTATION (Spatial Chaos)")
    print("-" * 80)
    
    # Label connected components for each prediction class
    labeled_1, num_regions_1 = label(binary_grid == 1)
    labeled_0, num_regions_0 = label(binary_grid == 0)
    
    total_regions = num_regions_1 + num_regions_0
    
    print(f"Number of disconnected 'Token 1 wins' regions: {num_regions_1}")
    print(f"Number of disconnected 'Token 2 wins' regions: {num_regions_0}")
    print(f"Total disconnected regions: {total_regions}")
    print(f"Interpretation: Clean boundary would have 2 regions (one per class)")
    print(f"              : {total_regions} regions indicates high fragmentation")
    
    # 3. SPATIAL AUTOCORRELATION: Measure how predictions correlate with neighbors
    print("\n3. SPATIAL AUTOCORRELATION (Predictability)")
    print("-" * 80)
    
    # Calculate Moran's I statistic
    def morans_i(grid):
        flat = grid.flatten()
        n = len(flat)
        mean = np.mean(flat)
        
        # Create spatial weights (4-connectivity)
        W = np.zeros((n, n))
        rows, cols = grid.shape
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                # Add neighbors (up, down, left, right)
                if i > 0:
                    W[idx, (i-1)*cols + j] = 1
                if i < rows-1:
                    W[idx, (i+1)*cols + j] = 1
                if j > 0:
                    W[idx, i*cols + (j-1)] = 1
                if j < cols-1:
                    W[idx, i*cols + (j+1)] = 1
        
        W_sum = np.sum(W)
        
        numerator = n * np.sum(W * np.outer(flat - mean, flat - mean))
        denominator = W_sum * np.sum((flat - mean)**2)
        
        I = numerator / denominator if denominator != 0 else 0
        return I
    
    moran = morans_i(binary_grid)
    
    print(f"Moran's I: {moran:.4f}")
    print(f"Interpretation: Moran's I ranges from -1 to +1")
    print(f"              : +1 = perfect spatial clustering (smooth boundary)")
    print(f"              : 0 = random spatial pattern")
    print(f"              : -1 = perfect checkerboard (maximum chaos)")
    print(f"              : Your value {moran:.4f} indicates ", end="")
    if moran > 0.7:
        print("strong clustering (smooth boundary)")
    elif moran > 0.3:
        print("moderate clustering")
    elif moran > -0.3:
        print("RANDOM/CHAOTIC pattern")
    else:
        print("strong anti-clustering (checkerboard chaos)")
    
    # 4. ENTROPY: Measure unpredictability in local neighborhoods
    print("\n4. LOCAL ENTROPY (Unpredictability)")
    print("-" * 80)
    
    def local_entropy(grid, window_size=3):
        """Calculate entropy in sliding windows"""
        entropies = []
        half_w = window_size // 2
        
        for i in range(half_w, grid.shape[0] - half_w):
            for j in range(half_w, grid.shape[1] - half_w):
                window = grid[i-half_w:i+half_w+1, j-half_w:j+half_w+1]
                # Calculate entropy
                p1 = np.mean(window)
                p0 = 1 - p1
                
                if p1 > 0 and p0 > 0:
                    ent = -p1 * np.log2(p1) - p0 * np.log2(p0)
                else:
                    ent = 0
                entropies.append(ent)
        
        return np.array(entropies)
    
    entropies = local_entropy(binary_grid, window_size=5)
    mean_entropy = np.mean(entropies)
    max_entropy = 1.0  # Maximum entropy for binary variable
    
    print(f"Mean local entropy: {mean_entropy:.4f} / {max_entropy:.4f}")
    print(f"Entropy ratio: {mean_entropy/max_entropy*100:.2f}%")
    print(f"Interpretation: 0% = perfectly predictable (all same class)")
    print(f"              : 100% = maximum uncertainty (50/50 mix)")
    print(f"              : {mean_entropy/max_entropy*100:.1f}% indicates ", end="")
    if mean_entropy/max_entropy > 0.8:
        print("HIGHLY RANDOM/CHAOTIC boundary")
    elif mean_entropy/max_entropy > 0.5:
        print("moderately random boundary")
    else:
        print("relatively smooth boundary")
    
    # 5. GRADIENT ANALYSIS: How quickly logits change
    print("\n5. LOGIT GRADIENT MAGNITUDE (Sensitivity)")
    print("-" * 80)
    
    # Calculate gradients
    grad_e1 = np.gradient(grid, axis=0)
    grad_e2 = np.gradient(grid, axis=1)
    grad_magnitude = np.sqrt(grad_e1**2 + grad_e2**2)
    
    mean_grad = np.mean(grad_magnitude)
    std_grad = np.std(grad_magnitude)
    max_grad = np.max(grad_magnitude)
    
    print(f"Mean gradient magnitude: {mean_grad:.2e}")
    print(f"Std gradient magnitude: {std_grad:.2e}")
    print(f"Max gradient magnitude: {max_grad:.2e}")
    print(f"Coefficient of variation: {std_grad/mean_grad:.2f}")
    print(f"Interpretation: High std/mean ratio indicates erratic sensitivity")
    print(f"              : Logits change unpredictably across the space")
    
    # 6. ZERO-CROSSING DENSITY: How often logit difference crosses zero
    print("\n6. ZERO-CROSSING DENSITY (Boundary Complexity)")
    print("-" * 80)
    
    # Count zero crossings along rows and columns
    row_crossings = np.sum([np.sum(np.diff(np.sign(grid[i, :])) != 0) 
                            for i in range(grid.shape[0])])
    col_crossings = np.sum([np.sum(np.diff(np.sign(grid[:, j])) != 0) 
                            for j in range(grid.shape[1])])
    total_crossings = row_crossings + col_crossings
    
    # Normalize by grid size
    crossing_density = total_crossings / (grid.shape[0] + grid.shape[1])
    
    print(f"Total zero crossings: {total_crossings}")
    print(f"Crossing density: {crossing_density:.2f} crossings per line")
    print(f"Interpretation: Clean boundary would have ~1-2 crossings per line")
    print(f"              : {crossing_density:.1f} crossings indicates ", end="")
    if crossing_density > 10:
        print("EXTREMELY COMPLEX/CHAOTIC boundary")
    elif crossing_density > 5:
        print("highly complex boundary")
    elif crossing_density > 2:
        print("moderately complex boundary")
    else:
        print("relatively simple boundary")
    
    # 7. RANDOMNESS TEST: Chi-square test for spatial randomness
    print("\n7. STATISTICAL RANDOMNESS TEST")
    print("-" * 80)
    
    # Expected frequencies for random distribution
    n_total = binary_grid.size
    expected_freq = n_total / 2  # 50/50 split expected
    observed_1 = np.sum(binary_grid == 1)
    observed_0 = np.sum(binary_grid == 0)
    
    # Chi-square test for uniform distribution
    chi2_stat = ((observed_1 - expected_freq)**2 / expected_freq + 
                 (observed_0 - expected_freq)**2 / expected_freq)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    print(f"Token 1 wins: {observed_1} ({observed_1/n_total*100:.2f}%)")
    print(f"Token 2 wins: {observed_0} ({observed_0/n_total*100:.2f}%)")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Interpretation: p > 0.05 suggests distribution is consistent with random")
    print(f"              : Cannot reject randomness hypothesis" if p_value > 0.05 
          else "              : Significant deviation from randomness")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Create summary score
    randomness_indicators = []
    
    if flip_ratio > 0.3:
        randomness_indicators.append("High flip frequency")
    if total_regions > 10:
        randomness_indicators.append("High fragmentation")
    if abs(moran) < 0.3:
        randomness_indicators.append("Low spatial autocorrelation")
    if mean_entropy/max_entropy > 0.7:
        randomness_indicators.append("High local entropy")
    if crossing_density > 5:
        randomness_indicators.append("High boundary complexity")
    if p_value > 0.05:
        randomness_indicators.append("Statistically random distribution")
    
    print(f"\nRandomness indicators found: {len(randomness_indicators)}/6")
    for indicator in randomness_indicators:
        print(f"  ✓ {indicator}")
    
    if len(randomness_indicators) >= 4:
        print("\n⚠️  CONCLUSION: The decision boundary exhibits STRONG CHAOTIC/RANDOM behavior")
        print("   Small perturbations cause unpredictable logit flips in this region.")
    elif len(randomness_indicators) >= 2:
        print("\n⚠️  CONCLUSION: The decision boundary shows MODERATE randomness")
        print("   Predictions are somewhat unstable near the boundary.")
    else:
        print("\n✓ CONCLUSION: The decision boundary is relatively smooth and predictable")
    
    return {
        'flip_ratio': flip_ratio,
        'num_regions': total_regions,
        'morans_i': moran,
        'mean_entropy': mean_entropy,
        'crossing_density': crossing_density,
        'chi2_p_value': p_value,
        'randomness_score': len(randomness_indicators)
    }

def create_analysis_plots(grid, e1_values, e2_values, output_prefix, exp_dir=None):
    """Create visualization plots for the analysis"""
    binary_grid = (grid >= 0).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Original heatmap
    im1 = axes[0, 0].imshow(grid.T, extent=[e1_values[0], e1_values[-1], 
                                             e2_values[0], e2_values[-1]], 
                            origin='lower', cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_xlabel('e1')
    axes[0, 0].set_ylabel('e2')
    axes[0, 0].set_title('Logit Difference (L1 - L2)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Binary decision map
    im2 = axes[0, 1].imshow(binary_grid.T, extent=[e1_values[0], e1_values[-1], 
                                                    e2_values[0], e2_values[-1]], 
                            origin='lower', cmap='binary', aspect='auto')
    axes[0, 1].set_xlabel('e1')
    axes[0, 1].set_ylabel('e2')
    axes[0, 1].set_title('Binary Predictions (White=Token1, Black=Token2)')
    
    # Plot 3: Gradient magnitude
    grad_e1 = np.gradient(grid, axis=0)
    grad_e2 = np.gradient(grid, axis=1)
    grad_magnitude = np.sqrt(grad_e1**2 + grad_e2**2)
    
    im3 = axes[1, 0].imshow(grad_magnitude.T, extent=[e1_values[0], e1_values[-1], 
                                                       e2_values[0], e2_values[-1]], 
                            origin='lower', cmap='hot', aspect='auto')
    axes[1, 0].set_xlabel('e1')
    axes[1, 0].set_ylabel('e2')
    axes[1, 0].set_title('Gradient Magnitude (Sensitivity)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot 4: Local entropy
    def local_entropy_map(grid, window_size=5):
        half_w = window_size // 2
        entropy_map = np.zeros_like(grid, dtype=float)
        
        for i in range(half_w, grid.shape[0] - half_w):
            for j in range(half_w, grid.shape[1] - half_w):
                window = grid[i-half_w:i+half_w+1, j-half_w:j+half_w+1]
                p1 = np.mean(window)
                p0 = 1 - p1
                
                if p1 > 0 and p0 > 0:
                    entropy_map[i, j] = -p1 * np.log2(p1) - p0 * np.log2(p0)
        
        return entropy_map
    
    entropy_map = local_entropy_map(binary_grid)
    
    im4 = axes[1, 1].imshow(entropy_map.T, extent=[e1_values[0], e1_values[-1], 
                                                    e2_values[0], e2_values[-1]], 
                            origin='lower', cmap='viridis', aspect='auto')
    axes[1, 1].set_xlabel('e1')
    axes[1, 1].set_ylabel('e2')
    axes[1, 1].set_title('Local Entropy (Unpredictability)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    filename = f"{output_prefix}_analysis.png"
    if exp_dir is not None:
        filename = os.path.join(exp_dir, os.path.basename(filename))
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved analysis plots: {filename}")

if __name__ == "__main__":
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp7_part3_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved to: {exp_dir}\n")

    parser = argparse.ArgumentParser(description='Analyze randomness in decision boundary')
    parser.add_argument('filename', type=str, help='Path to .npz file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output prefix for plots')

    args = parser.parse_args()

    # Load matrix
    grid, e1_values, e2_values = load_matrix(args.filename)

    # Run analysis
    results = analyze_decision_boundary_randomness(grid, e1_values, e2_values)

    # Create plots
    output_prefix = args.output if args.output else args.filename.replace('.npz', '')
    create_analysis_plots(grid, e1_values, e2_values, output_prefix, exp_dir)

    print("\nAnalysis complete!")

# Example usage:
# python src/experiment7_part3_randomness_calc.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_2nd.npz" --output "RSV1and2"
# python analyze_randomness.py logit_diff_1st_2nd.npz --output my_analysis