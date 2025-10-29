"""
Experiment 4 Part 3: Divergence Index Representation Analysis

This script performs detailed dimension-level analysis of representations at
divergence points, quantifying how many dimensions change at different precision
levels between GPU runs.

Purpose:
--------
Provides fine-grained analysis of divergence by:
1. Comparing representations dimension-by-dimension at specific indices
2. Counting how many dimensions differ at various thresholds (1e-1 to 1e-7)
3. Computing statistical measures of differences (mean, std, max, percentiles)
4. Understanding the magnitude distribution of GPU-induced changes

Input:
------
Requires two .npy files and a specific index (typically divergence index from part2):
- repr_file1: representations.npy from GPU 0
- repr_file2: representations.npy from GPU 1
- index: Position to analyze (usually the divergence index)

Methodology:
------------
1. Load representation arrays from both GPU runs
2. Extract vectors at the specified index: vec1 = repr1[index], vec2 = repr2[index]
3. Compute element-wise absolute differences: diff = |vec1 - vec2|
4. Count dimensions where diff exceeds various thresholds:
   - > 1e-1 (large changes)
   - > 1e-2 (medium changes)
   - > 1e-3, 1e-4, 1e-5 (small changes)
   - > 1e-6, 1e-7 (tiny changes)
   - > 0 (any change at all)
5. Compute statistics: mean, std, max, min, percentiles
6. Report percentage of dimensions changed at each level

Use Case:
---------
Use this script to:
- Understand the scale of numerical differences between GPUs
- Determine if changes are large (> 1e-2) or tiny (< 1e-5)
- Identify if all dimensions change slightly or few dimensions change significantly
- Assess whether differences are meaningful or within expected numerical precision

Interpretation:
---------------
High percentage at 1e-1 threshold:
- Large differences in many dimensions
- Significant GPU-specific behavior
- Likely to cause different predictions

High percentage at 1e-7 threshold only:
- Tiny differences in many dimensions
- Within expected floating-point precision
- May not significantly affect predictions

Concentrated in few dimensions:
- Sparse pattern of changes
- Specific components affected
- May indicate particular operations sensitive to hardware

Diffuse across many dimensions:
- Widespread numerical differences
- Systematic hardware effect
- Accumulation of small errors

Dependencies:
-------------
- numpy, sys

Key Functions:
--------------
- compare_representations_at_index(): Main analysis function
  * Loads representations
  * Computes dimension-wise differences
  * Counts changes at each threshold
  * Provides comprehensive statistics

Output:
-------
Console report showing:
- Total dimensions analyzed
- Count and percentage of changes at each threshold (1e-1 to 1e-7)
- Statistical summary (mean, std, max, min, percentiles)
- Overall characterization of differences

Usage Example:
--------------
```python
compare_representations_at_index(
    'results/exp4_gpu0/question_0/representations.npy',
    'results/exp4_gpu1/question_0/representations.npy',
    index=15  # Divergence index from part2
)
```

Note:
-----
This analysis complements part2 (which finds WHEN divergence occurs) by
quantifying HOW MUCH representations differ at that point. Essential for
understanding whether divergences are due to large systematic differences or
accumulation of tiny numerical errors.
"""

import numpy as np
import sys

def compare_representations_at_index(repr_file1, repr_file2, index):
    """Compare two representations at a specific index"""
    
    # Load representations
    repr1 = np.load(repr_file1)
    repr2 = np.load(repr_file2)
    
    # Get vectors at the specified index
    vec1 = repr1[index]
    vec2 = repr2[index]
    
    # Calculate absolute differences
    diff = np.abs(vec1 - vec2)
    
    # Count changes at different precision levels
    changed_1e_1 = np.sum(diff > 1e-1)
    changed_1e_2 = np.sum(diff > 1e-2)
    changed_1e_3 = np.sum(diff > 1e-3)
    changed_1e_4 = np.sum(diff > 1e-4)
    changed_1e_5 = np.sum(diff > 1e-5)
    changed_1e_6 = np.sum(diff > 1e-6)
    changed_1e_7 = np.sum(diff > 1e-7)
    changed_any = np.sum(diff > 0)
    
    total_dims = len(vec1)
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"REPRESENTATION COMPARISON AT INDEX {index}")
    print(f"{'='*60}")
    print(f"Total dimensions: {total_dims}")
    print(f"\nValues changed at different precision levels:")
    print(f"  > 1e-1:  {changed_1e_1:6d} ({changed_1e_1/total_dims*100:6.2f}%)")
    print(f"  > 1e-2:  {changed_1e_2:6d} ({changed_1e_2/total_dims*100:6.2f}%)")
    print(f"  > 1e-3:  {changed_1e_3:6d} ({changed_1e_3/total_dims*100:6.2f}%)")
    print(f"  > 1e-4:  {changed_1e_4:6d} ({changed_1e_4/total_dims*100:6.2f}%)")
    print(f"  > 1e-5:  {changed_1e_5:6d} ({changed_1e_5/total_dims*100:6.2f}%)")
    print(f"  > 1e-6:  {changed_1e_6:6d} ({changed_1e_6/total_dims*100:6.2f}%)")
    print(f"  > 1e-7:  {changed_1e_7:6d} ({changed_1e_7/total_dims*100:6.2f}%)")
    print(f"  > 0:     {changed_any:6d} ({changed_any/total_dims*100:6.2f}%)")
    
    print(f"\nDifference statistics:")
    print(f"  Mean abs diff:    {np.mean(diff):.2e}")
    print(f"  Median abs diff:  {np.median(diff):.2e}")
    print(f"  Max abs diff:     {np.max(diff):.2e}")
    print(f"  Min abs diff:     {np.min(diff):.2e}")
    print(f"  Std of diff:      {np.std(diff):.2e}")
    
    print(f"\nVector norms:")
    print(f"  Machine 1: {np.linalg.norm(vec1):.6f}")
    print(f"  Machine 2: {np.linalg.norm(vec2):.6f}")
    
    print(f"\nSimilarity metrics:")
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    l2_dist = np.linalg.norm(vec1 - vec2)
    print(f"  Cosine similarity: {cos_sim:.10f}")
    print(f"  L2 distance:       {l2_dist:.6f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_repr.py <repr_file1.npy> <repr_file2.npy> <index>")
        print("Example: python compare_repr.py repr1.npy repr2.npy 50")
        sys.exit(1)
    
    repr_file1 = sys.argv[1]
    repr_file2 = sys.argv[2]
    index = int(sys.argv[3])
    
    compare_representations_at_index(repr_file1, repr_file2, index)

    '''
    python3 src/experiment4_part3_divergence_indx_comaprison.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A5000_2x24GB/question_01/representations.npy" "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A6000_48GB/question_01/representations.npy" 52 
    '''