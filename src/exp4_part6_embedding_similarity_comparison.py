"""
Experiment 4 Part 6: Input Embedding Comparison Between GPUs

This script compares the input embeddings extracted by experiment4_part5 from
different GPU runs, testing whether embedding matrices are loaded identically
across hardware.

Purpose:
--------
Verifies that input token embeddings are identical between GPUs by:
1. Loading embeddings extracted on GPU 0 and GPU 1 (from part5)
2. Comparing embeddings token-by-token (cosine similarity, L2 distance)
3. Identifying any differences in embedding lookups
4. Confirming that divergence is NOT due to embedding differences


If embeddings match (expected):
- Divergence must occur during forward pass
- Focus investigation on matrix operations, attention, normalization

If embeddings differ (unexpected):
- Model loading issue
- Embedding precision/rounding during weight loading
- Needs investigation of model checkpoint handling

Methodology:
------------
1. Load embedding files from both GPU runs (from part5)
2. Load metadata (tokens, prompts, indices)
3. For each question:
   a. Compare embeddings token-by-token
   b. Compute cosine similarity (should be 1.0)
   c. Compute L2 distance (should be 0.0)
   d. Identify ANY tokens where embeddings differ
   e. Plot similarity trends
4. Generate summary report

Comparison Metrics:
-------------------
For each token position:
- Cosine similarity: 1 - cosine_distance(emb1, emb2)
  * Perfect match: 1.0
  * Different: < 0.999999
- L2 distance: ||emb1 - emb2||
  * Perfect match: 0.0
  * Different: > 1e-6

Analysis:
---------
Reports:
- Number of tokens compared
- Number of tokens with ANY difference (threshold > 0)
- Tokens with significant differences (threshold > 1e-6)
- Statistics: mean/max cosine similarity and L2 distance
- List of changed tokens with their metrics

Use Case:
---------
Use this script to:
- Verify embedding matrices are loaded identically on both GPUs
- Rule out embedding differences as divergence source
- Confirm that computational differences cause divergence
- Validate model checkpoint loading consistency

Expected Results:
-----------------
TYPICAL CASE (embeddings identical):
- All cosine similarities = 1.0
- All L2 distances = 0.0
- No tokens with differences
- Conclusion: Divergence caused by forward pass computations

ATYPICAL CASE (embeddings differ):
- Some cosine similarities < 1.0
- Some L2 distances > 0
- Specific tokens with embedding differences
- Conclusion: Model loading or weight precision issue

Dependencies:
-------------
- numpy, matplotlib, scipy
- json, pathlib

Key Functions:
--------------
- load_embeddings(): Load .npy embedding files
- load_metadata(): Load token/prompt information
- compare_embeddings_simple(): Main comparison logic
  * Token-by-token comparison
  * Cosine similarity and L2 distance
  * Identify changed tokens
- Plot generation: Visualize similarity trends

Output:
-------
- Console report:
  * Number of tokens compared
  * Number of changed tokens
  * Detailed metrics for changed tokens
- Plots:
  * Cosine similarity across token positions
  * L2 distance across token positions
- Summary: Whether embeddings are identical or differ

Workflow:
---------
1. Run experiment4_GPU_comaprison.py on GPU 0 and GPU 1
2. Run experiment4_part2 to identify divergences
3. Run experiment4_part5 on GPU 0 → extract embeddings
4. Run experiment4_part5 on GPU 1 → extract embeddings
5. Run THIS script to compare the two embedding sets

Note:
-----
This analysis COMPLETES the experiment4 series by definitively establishing
whether divergence originates from:
1. Embedding differences (if this finds differences) - RARE
2. Computational differences (if embeddings identical) - EXPECTED

Most likely outcome: Embeddings identical, confirming that GPU-specific
numerical behavior occurs during forward pass computations, not during
embedding lookups.
"""

import numpy as np
import os
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_embeddings(embedding_file: Path) -> np.ndarray:
    """Load embeddings from .npy file"""
    return np.load(embedding_file)

def load_metadata(folder: Path, question_id: int) -> Dict:
    """Load metadata for a question"""
    metadata_file = folder / "embeddings_metadata.json"
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)
    return all_metadata.get(str(question_id), {})

def compare_embeddings_simple(emb1: np.ndarray, emb2: np.ndarray, question_id: int, 
                               tokens1: List[str] = None, tokens2: List[str] = None) -> Dict:
    """
    Simple comparison: for each token position, check if embeddings differ
    Only include tokens that have any difference
    """
    num_tokens = min(len(emb1), len(emb2))
    
    results = {
        'question_id': question_id,
        'num_tokens': num_tokens,
        'changed_tokens': []
    }
    
    # Store all token metrics for plotting
    all_cosine_sims = []
    all_l2_dists = []
    
    # For each token position
    for i in range(num_tokens):
        token_emb1 = emb1[i]
        token_emb2 = emb2[i]
        
        # Calculate cosine similarity and L2 distance
        cos_sim = 1 - cosine(token_emb1, token_emb2)
        l2_dist = euclidean(token_emb1, token_emb2)
        
        # Store for plotting (all tokens)
        all_cosine_sims.append(float(cos_sim))
        all_l2_dists.append(float(l2_dist))
        
        # Skip if exactly identical (cosine=1.0 and l2=0.0)
        if np.isclose(cos_sim, 1.0, atol=1e-15) and np.isclose(l2_dist, 0.0, atol=1e-15):
            continue
        
        # Check at different decimal precisions
        precision_diffs = {}
        has_any_difference = False
        
        for decimal in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
            emb1_rounded = np.round(token_emb1, decimal)
            emb2_rounded = np.round(token_emb2, decimal)
            is_different = not np.array_equal(emb1_rounded, emb2_rounded)
            
            if is_different:
                has_any_difference = True
                num_diff_elements = np.sum(emb1_rounded != emb2_rounded)
                precision_diffs[f'decimal_{decimal}'] = {
                    'different': True,
                    'num_elements_different': int(num_diff_elements),
                    'total_elements': len(token_emb1)
                }
            else:
                precision_diffs[f'decimal_{decimal}'] = {
                    'different': False
                }
        
        # Only add if there's any difference
        if has_any_difference:
            token_comparison = {
                'token_index': i,
                'cosine_similarity': float(cos_sim),
                'l2_distance': float(l2_dist),
                'precision_differences': precision_diffs
            }
            results['changed_tokens'].append(token_comparison)
    
    # Store all metrics for plotting
    results['all_cosine_similarities'] = all_cosine_sims
    results['all_l2_distances'] = all_l2_dists
    
    # Summary statistics
    summary = {}
    for decimal in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
        different_indices = []
        
        for tc in results['changed_tokens']:
            idx = tc['token_index']
            if tc['precision_differences'][f'decimal_{decimal}']['different']:
                different_indices.append(idx)
        
        num_different = len(different_indices)
        num_same = num_tokens - num_different
        
        summary[f'decimal_{decimal}'] = {
            'num_different': num_different,
            'num_same': num_same,
            'total': num_tokens,
            'percentage_different': float(num_different / num_tokens * 100),
            'different_indices': different_indices
        }
    
    results['summary'] = summary
    results['num_changed_tokens'] = len(results['changed_tokens'])
    results['num_unchanged_tokens'] = num_tokens - len(results['changed_tokens'])
    
    # Add last 3 tokens analysis
    if tokens1 and tokens2 and num_tokens >= 3:
        last_3_analysis = {
            'tokens': []
        }
        
        for i in range(num_tokens - 3, num_tokens):
            token_text = tokens1[i] if i < len(tokens1) else "N/A"
            cos_sim = all_cosine_sims[i]
            l2_dist = all_l2_dists[i]
            
            # Check precision differences for this token
            token_emb1 = emb1[i]
            token_emb2 = emb2[i]
            
            precision_diffs = {}
            for decimal in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
                emb1_rounded = np.round(token_emb1, decimal)
                emb2_rounded = np.round(token_emb2, decimal)
                is_different = not np.array_equal(emb1_rounded, emb2_rounded)
                
                if is_different:
                    num_diff_elements = np.sum(emb1_rounded != emb2_rounded)
                    precision_diffs[f'decimal_{decimal}'] = {
                        'different': True,
                        'num_elements_different': int(num_diff_elements),
                        'total_elements': len(token_emb1)
                    }
                else:
                    precision_diffs[f'decimal_{decimal}'] = {
                        'different': False
                    }
            
            last_3_analysis['tokens'].append({
                'token_index': i,
                'token_text': token_text,
                'cosine_similarity': float(cos_sim),
                'l2_distance': float(l2_dist),
                'precision_differences': precision_diffs
            })
        
        results['last_3_tokens_analysis'] = last_3_analysis
    
    return results

def plot_comparison(result: Dict, output_dir: Path):
    """
    Create a plot with cosine similarity and L2 distance for all tokens
    """
    question_id = result['question_id']
    num_tokens = result['num_tokens']
    cosine_sims = result['all_cosine_similarities']
    l2_dists = result['all_l2_distances']
    
    token_indices = list(range(num_tokens))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cosine Similarity
    ax1.plot(token_indices, cosine_sims, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(token_indices, cosine_sims, c='blue', s=20, alpha=0.5)
    ax1.set_xlabel('Token Index', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title(f'Question {question_id}: Cosine Similarity per Token', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(cosine_sims) - 0.0001, 1.0001])
    
    # Highlight changed tokens
    changed_indices = [tc['token_index'] for tc in result['changed_tokens']]
    if changed_indices:
        changed_cosines = [cosine_sims[i] for i in changed_indices]
        ax1.scatter(changed_indices, changed_cosines, c='red', s=50, marker='x', linewidths=2, 
                   label=f'Changed tokens ({len(changed_indices)})', zorder=5)
        ax1.legend()
    
    # Highlight last 3 tokens
    if num_tokens >= 3:
        last_3_indices = list(range(num_tokens - 3, num_tokens))
        last_3_cosines = [cosine_sims[i] for i in last_3_indices]
        ax1.scatter(last_3_indices, last_3_cosines, c='purple', s=100, marker='o', 
                   edgecolors='black', linewidths=2, label='Last 3 tokens', zorder=6)
        ax1.legend()
    
    # Plot 2: L2 Distance
    ax2.plot(token_indices, l2_dists, 'g-', linewidth=1, alpha=0.7)
    ax2.scatter(token_indices, l2_dists, c='green', s=20, alpha=0.5)
    ax2.set_xlabel('Token Index', fontsize=12)
    ax2.set_ylabel('L2 Distance', fontsize=12)
    ax2.set_title(f'Question {question_id}: L2 Distance per Token', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Use log scale only if there are significant variations
    max_l2 = max(l2_dists)
    min_l2 = min([d for d in l2_dists if d > 0], default=1e-10)
    if max_l2 / min_l2 > 10:
        ax2.set_yscale('log')
    
    # Highlight changed tokens
    if changed_indices:
        changed_l2 = [l2_dists[i] for i in changed_indices]
        ax2.scatter(changed_indices, changed_l2, c='red', s=50, marker='x', linewidths=2,
                   label=f'Changed tokens ({len(changed_indices)})', zorder=5)
        ax2.legend()
    
    # Highlight last 3 tokens
    if num_tokens >= 3:
        last_3_l2 = [l2_dists[i] for i in last_3_indices]
        ax2.scatter(last_3_indices, last_3_l2, c='purple', s=100, marker='o',
                   edgecolors='black', linewidths=2, label='Last 3 tokens', zorder=6)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'question_{question_id:02d}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: {output_file}")

def print_summary(result: Dict):
    """Print human-readable summary"""
    print(f"\nQuestion {result['question_id']}:")
    print(f"Total tokens: {result['num_tokens']}")
    print(f"Changed tokens: {result['num_changed_tokens']}")
    print(f"Unchanged tokens: {result['num_unchanged_tokens']}")
    
    # Print last 3 tokens
    if 'last_3_tokens_analysis' in result:
        print("\nLast 3 tokens:")
        for token_info in result['last_3_tokens_analysis']['tokens']:
            print(f"  Token {token_info['token_index']}: '{token_info['token_text']}'")
            print(f"    Cosine similarity: {token_info['cosine_similarity']:.10f}")
            print(f"    L2 distance: {token_info['l2_distance']:.6e}")
            # Show if different at 6 decimals
            if token_info['precision_differences']['decimal_6']['different']:
                num_diff = token_info['precision_differences']['decimal_6']['num_elements_different']
                print(f"    Different at 6 decimals: {num_diff}/4096 elements")
    
    print("\nSummary by precision:")
    for decimal in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
        key = f'decimal_{decimal}'
        summary = result['summary'][key]
        print(f"  {decimal} decimals: {summary['num_different']}/{summary['total']} different ({summary['percentage_different']:.2f}%)")
        if summary['num_different'] > 0 and summary['num_different'] <= 10:
            print(f"    Different at indices: {summary['different_indices']}")
    
    # Show details for changed tokens
    if result['num_changed_tokens'] > 0:
        print(f"\nChanged tokens details:")
        for tc in result['changed_tokens'][:10]:  # Show first 10
            print(f"  Token {tc['token_index']}: cos_sim={tc['cosine_similarity']:.10f}, l2_dist={tc['l2_distance']:.6e}")
        if result['num_changed_tokens'] > 10:
            print(f"  ... and {result['num_changed_tokens'] - 10} more")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simple embedding comparison with plots')
    parser.add_argument('folder1', type=str, help='First embeddings folder')
    parser.add_argument('folder2', type=str, help='Second embeddings folder')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: timestamped in ../results/)')
    parser.add_argument('--plot-dir', type=str, default=None,
                       help='Directory to save plots (default: timestamped in ../results/)')

    args = parser.parse_args()

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp4_part6_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Set output paths
    if args.output is None:
        output_file = os.path.join(exp_dir, 'simple_comparison.json')
    else:
        output_file = os.path.join(exp_dir, args.output)

    if args.plot_dir is None:
        plot_dir = Path(os.path.join(exp_dir, 'comparison_plots'))
    else:
        plot_dir = Path(os.path.join(exp_dir, args.plot_dir))

    folder1 = Path(args.folder1)
    folder2 = Path(args.folder2)

    # Create plot directory
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("SIMPLE EMBEDDING COMPARISON WITH PLOTS")
    print("="*80)
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print(f"Plot directory: {plot_dir}")
    
    # Find all embedding files
    emb_files1 = sorted(folder1.glob("question_*_embeddings.npy"))
    
    all_results = []
    
    for emb_file1 in emb_files1:
        question_id = int(emb_file1.stem.split('_')[1])
        emb_file2 = folder2 / emb_file1.name
        
        if not emb_file2.exists():
            print(f"\nQuestion {question_id}: Not found in folder 2, skipping...")
            continue
        
        print(f"\nProcessing Question {question_id}...")
        
        # Load embeddings
        emb1 = load_embeddings(emb_file1)
        emb2 = load_embeddings(emb_file2)
        
        # Load metadata to get tokens
        metadata1 = load_metadata(folder1, question_id)
        metadata2 = load_metadata(folder2, question_id)
        
        tokens1 = metadata1.get('tokens', None)
        tokens2 = metadata2.get('tokens', None)
        
        # Compare
        result = compare_embeddings_simple(emb1, emb2, question_id, tokens1, tokens2)
        all_results.append(result)
        
        # Print summary
        print_summary(result)
        
        # Create plot
        plot_comparison(result, plot_dir)
    
    # Save results (without the plotting arrays to reduce file size)
    results_to_save = []
    for r in all_results:
        r_copy = r.copy()
        # Remove large arrays from JSON output
        r_copy.pop('all_cosine_similarities', None)
        r_copy.pop('all_l2_distances', None)
        results_to_save.append(r_copy)
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Plots saved to: {plot_dir}/")
    print(f"Total plots generated: {len(all_results)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

"""
Usage:
python src/experiment4_part6_comp
are_input_embs.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/embeddings_till_divergence_A5000" "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/embeddings_till_divergence_A6000" --output exp4_input_embedding_comparison.json  --plot-dir exp4_input_embs_comparison_plots
"""