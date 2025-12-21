"""
LPT to Unembedding Vector Angle Distribution Analysis

This script analyzes the distribution of angles between Last Position Token (LPT) 
representations (after final normalization) and unembedding vectors from the lm_head.

The analysis is performed separately for:
- Top-1 tokens (most likely predicted token)
- Top-2 tokens
- Top-3 tokens
- Top-10 tokens
- Top-100 tokens

For each question generated with ~1000 tokens from experiment4 data.

Dependencies:
-------------
- torch, numpy, matplotlib, seaborn
- pathlib, json
- Llama-3.1-8B-Instruct model for unembedding weights

Input:
------
- experiment4 results directory with:
  * representations.npy: [num_tokens, hidden_size]
  * top_10_logits.npy: [num_tokens, 10]
  * words.json: top_10_token_ids

Output:
-------
- Angular distribution plots (histograms, KDE plots)
- Statistical summaries (mean, median, std, percentiles)
- Comparison plots across different rank categories
- Per-question and aggregated analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine
import warnings
import argparse

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_model_for_unembedding(model_path: str, device: str = "cuda:0"):
    """
    Load model to extract unembedding weights (lm_head)
    
    Args:
        model_path: Path to the Llama model
        device: Device to load model on
        
    Returns:
        unembedding_weights: [vocab_size, hidden_size] tensor
        tokenizer: Tokenizer for token ID mapping
    """
    print("Loading model for unembedding weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Extract unembedding weights from lm_head
    # Shape: [vocab_size, hidden_size]
    unembedding_weights = model.lm_head.weight.data.cpu().float()
    
    print(f"Unembedding weights shape: {unembedding_weights.shape}")
    print(f"Vocab size: {unembedding_weights.shape[0]}")
    print(f"Hidden size: {unembedding_weights.shape[1]}")
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    return unembedding_weights, tokenizer


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Ensure vectors are 1D
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Compute dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def compute_angle_degrees(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute angle between two vectors in degrees
    
    Returns:
        Angle in degrees [0, 180]
    """
    cos_sim = compute_cosine_similarity(vec1, vec2)
    
    # Clip to valid range for arccos to handle numerical errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Convert to angle in degrees
    angle_rad = np.arccos(cos_sim)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def load_question_data(question_dir: Path) -> Dict[str, Any]:
    """
    Load all data for a single question
    
    Returns:
        Dictionary with representations, token_ids, metadata
    """
    # Load representations (after final norm)
    representations = np.load(question_dir / "representations.npy")
    
    # Load words.json for token IDs
    with open(question_dir / "words.json", 'r') as f:
        words_data = json.load(f)
    
    # Load metadata
    with open(question_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Extract top-10 token IDs for each generation step
    top_10_token_ids = np.array(words_data['top_10_token_ids'])
    
    return {
        'representations': representations,  # [num_tokens, hidden_size]
        'top_10_token_ids': top_10_token_ids,  # [num_tokens, 10]
        'metadata': metadata,
        'num_tokens': representations.shape[0]
    }


def compute_angles_for_question(
    question_data: Dict[str, Any],
    unembedding_weights: torch.Tensor,
    rank_categories: List[int] = [1, 2, 3, 10, 100]
) -> Dict[int, np.ndarray]:
    """
    Compute angles between LPT representations and unembedding vectors
    for different rank categories
    
    Args:
        question_data: Data for one question
        unembedding_weights: [vocab_size, hidden_size] unembedding matrix
        rank_categories: List of ranks to analyze (1=top-1, 2=top-2, etc.)
        
    Returns:
        Dictionary mapping rank -> array of angles [num_tokens]
    """
    representations = question_data['representations']
    top_10_token_ids = question_data['top_10_token_ids']
    num_tokens = representations.shape[0]
    
    # Convert unembedding weights to numpy
    unembedding_np = unembedding_weights.numpy()
    vocab_size = unembedding_np.shape[0]
    
    # Initialize results dictionary
    angles_by_rank = {rank: [] for rank in rank_categories}
    
    # For each generation step
    for step in range(num_tokens):
        lpt_rep = representations[step]  # [hidden_size]
        
        # Get top-k token IDs for this step
        token_ids_step = top_10_token_ids[step]  # [10]
        
        # Compute angles for each rank category
        for rank in rank_categories:
            if rank <= 10:
                # Use the token at this rank from top-10
                token_id = token_ids_step[rank - 1]
                unembedding_vec = unembedding_np[token_id]
                
                angle = compute_angle_degrees(lpt_rep, unembedding_vec)
                angles_by_rank[rank].append(angle)
            
            elif rank == 100:
                # For top-100, we need to get all top-100 token IDs
                # We'll compute logits and get top-100
                # Since we only have top-10 stored, we'll use top-10 as proxy
                # OR we can recompute from representations
                
                # Option: Use average of angles to top-10 as proxy for top-100
                # OR skip this for now and only analyze what we have
                
                # Let's compute the angle to the 10th ranked token as a proxy
                if len(token_ids_step) >= 10:
                    token_id = token_ids_step[9]  # 10th token (0-indexed)
                    unembedding_vec = unembedding_np[token_id]
                    angle = compute_angle_degrees(lpt_rep, unembedding_vec)
                    angles_by_rank[rank].append(angle)
    
    # Convert lists to numpy arrays
    for rank in rank_categories:
        angles_by_rank[rank] = np.array(angles_by_rank[rank])
    
    return angles_by_rank


def compute_angles_with_actual_top100(
    question_data: Dict[str, Any],
    unembedding_weights: torch.Tensor
) -> np.ndarray:
    """
    Compute angles to actual top-100 tokens by computing full logits
    
    Args:
        question_data: Data for one question
        unembedding_weights: [vocab_size, hidden_size] unembedding matrix
        
    Returns:
        Array of angles to 100th ranked token [num_tokens]
    """
    representations = question_data['representations']
    num_tokens = representations.shape[0]
    
    # Convert to torch tensors
    reps_torch = torch.from_numpy(representations).float()  # [num_tokens, hidden_size]
    unembedding_torch = unembedding_weights.float()  # [vocab_size, hidden_size]
    
    # Compute logits: [num_tokens, vocab_size]
    logits = torch.matmul(reps_torch, unembedding_torch.T)
    
    # Get top-100 token IDs for each step
    top_100_values, top_100_indices = torch.topk(logits, k=100, dim=1)
    
    # Get the 100th ranked token (index 99)
    top_100_token_ids = top_100_indices[:, 99].numpy()  # [num_tokens]
    
    # Compute angles
    angles = []
    unembedding_np = unembedding_weights.numpy()
    
    for step in range(num_tokens):
        lpt_rep = representations[step]
        token_id = top_100_token_ids[step]
        unembedding_vec = unembedding_np[token_id]
        
        angle = compute_angle_degrees(lpt_rep, unembedding_vec)
        angles.append(angle)
    
    return np.array(angles)


def plot_angle_distributions(
    angles_by_rank: Dict[int, np.ndarray],
    question_id: int,
    output_dir: Path,
    title_suffix: str = ""
):
    """
    Create comprehensive visualizations of angle distributions
    
    Args:
        angles_by_rank: Dictionary mapping rank -> angles array
        question_id: Question identifier
        output_dir: Directory to save plots
        title_suffix: Additional text for plot titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ranks = sorted(angles_by_rank.keys())
    
    # ============================================================
    # 1. Overlaid Histograms
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, rank in enumerate(ranks):
        angles = angles_by_rank[rank]
        color = colors[i % len(colors)]
        
        ax.hist(
            angles, 
            bins=50, 
            alpha=0.5, 
            label=f'Top-{rank}',
            color=color,
            edgecolor='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Angle Distribution: LPT → Unembedding Vectors (Question {question_id})\n{title_suffix}',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'q{question_id}_histogram_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 2. Separate Subplots for Each Rank
    # ============================================================
    fig, axes = plt.subplots(len(ranks), 1, figsize=(14, 4 * len(ranks)))
    
    if len(ranks) == 1:
        axes = [axes]
    
    for i, rank in enumerate(ranks):
        angles = angles_by_rank[rank]
        color = colors[i % len(colors)]
        
        axes[i].hist(
            angles,
            bins=50,
            alpha=0.7,
            color=color,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add statistics
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        std_angle = np.std(angles)
        
        axes[i].axvline(mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.2f}°')
        axes[i].axvline(median_angle, color='green', linestyle='--', linewidth=2, label=f'Median: {median_angle:.2f}°')
        
        axes[i].set_xlabel('Angle (degrees)', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'Top-{rank} Token (std: {std_angle:.2f}°)', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Angle Distributions by Rank (Question {question_id})\n{title_suffix}',
        fontsize=14,
        fontweight='bold',
        y=1.001
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'q{question_id}_histogram_separate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 3. KDE Plots
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, rank in enumerate(ranks):
        angles = angles_by_rank[rank]
        color = colors[i % len(colors)]
        
        sns.kdeplot(
            data=angles,
            label=f'Top-{rank}',
            color=color,
            linewidth=2.5,
            ax=ax
        )
    
    ax.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(
        f'KDE: Angle Distribution (Question {question_id})\n{title_suffix}',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'q{question_id}_kde.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 4. Box Plot Comparison
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data_for_boxplot = [angles_by_rank[rank] for rank in ranks]
    labels_for_boxplot = [f'Top-{rank}' for rank in ranks]
    
    bp = ax.boxplot(
        data_for_boxplot,
        labels=labels_for_boxplot,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Angle Distribution Comparison (Question {question_id})\n{title_suffix}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'q{question_id}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 5. Violin Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for seaborn
    import pandas as pd
    
    data_list = []
    for rank in ranks:
        angles = angles_by_rank[rank]
        for angle in angles:
            data_list.append({'Rank': f'Top-{rank}', 'Angle': angle})
    
    df = pd.DataFrame(data_list)
    
    sns.violinplot(
        data=df,
        x='Rank',
        y='Angle',
        palette=colors[:len(ranks)],
        ax=ax
    )
    
    ax.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Angle Distribution (Violin Plot, Question {question_id})\n{title_suffix}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'q{question_id}_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots for Question {question_id}")


def compute_statistics(angles_by_rank: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    """
    Compute statistical summaries for angle distributions
    
    Returns:
        Dictionary mapping rank -> statistics dictionary
    """
    stats = {}
    
    for rank, angles in angles_by_rank.items():
        stats[rank] = {
            'mean': np.mean(angles),
            'median': np.median(angles),
            'std': np.std(angles),
            'min': np.min(angles),
            'max': np.max(angles),
            'q25': np.percentile(angles, 25),
            'q75': np.percentile(angles, 75),
            'q95': np.percentile(angles, 95),
            'q99': np.percentile(angles, 99),
            'count': len(angles)
        }
    
    return stats


def save_statistics(stats_by_question: Dict[int, Dict[int, Dict[str, float]]], output_dir: Path):
    """
    Save statistics to JSON file and create summary table
    """
    # Save full statistics
    stats_file = output_dir / "angle_statistics.json"
    
    # Convert to serializable format
    stats_serializable = {}
    for q_id, stats_by_rank in stats_by_question.items():
        stats_serializable[f"question_{q_id}"] = {
            f"top_{rank}": stats for rank, stats in stats_by_rank.items()
        }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\nSaved statistics to: {stats_file}")
    
    # Create summary table
    print("\n" + "="*100)
    print("STATISTICAL SUMMARY: ANGLES BETWEEN LPT AND UNEMBEDDING VECTORS")
    print("="*100)
    
    for q_id in sorted(stats_by_question.keys()):
        print(f"\nQuestion {q_id}:")
        print("-" * 100)
        print(f"{'Rank':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Q25':<10} {'Q75':<10} {'Q95':<10}")
        print("-" * 100)
        
        for rank in sorted(stats_by_question[q_id].keys()):
            stats = stats_by_question[q_id][rank]
            print(f"Top-{rank:<5} {stats['mean']:<10.2f} {stats['median']:<10.2f} {stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} {stats['max']:<10.2f} {stats['q25']:<10.2f} "
                  f"{stats['q75']:<10.2f} {stats['q95']:<10.2f}")


def plot_aggregated_distributions(
    all_angles_by_rank: Dict[int, List[np.ndarray]],
    output_dir: Path
):
    """
    Create aggregated plots across all questions
    
    Args:
        all_angles_by_rank: Dictionary mapping rank -> list of angle arrays (one per question)
        output_dir: Directory to save plots
    """
    print("\nCreating aggregated plots across all questions...")
    
    # Concatenate angles from all questions for each rank
    aggregated_angles = {}
    for rank, angle_lists in all_angles_by_rank.items():
        aggregated_angles[rank] = np.concatenate(angle_lists)
    
    ranks = sorted(aggregated_angles.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # ============================================================
    # 1. Aggregated Histogram
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for i, rank in enumerate(ranks):
        angles = aggregated_angles[rank]
        color = colors[i % len(colors)]
        
        ax.hist(
            angles,
            bins=100,
            alpha=0.5,
            label=f'Top-{rank} (n={len(angles)})',
            color=color,
            edgecolor='black',
            linewidth=0.3
        )
    
    ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title(
        'Aggregated Angle Distribution: LPT → Unembedding Vectors (All Questions)',
        fontsize=16,
        fontweight='bold'
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 2. Aggregated KDE
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for i, rank in enumerate(ranks):
        angles = aggregated_angles[rank]
        color = colors[i % len(colors)]
        
        sns.kdeplot(
            data=angles,
            label=f'Top-{rank}',
            color=color,
            linewidth=3,
            ax=ax
        )
    
    ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title(
        'Aggregated KDE: Angle Distribution (All Questions)',
        fontsize=16,
        fontweight='bold'
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_kde.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 3. Aggregated Box Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    data_for_boxplot = [aggregated_angles[rank] for rank in ranks]
    labels_for_boxplot = [f'Top-{rank}' for rank in ranks]
    
    bp = ax.boxplot(
        data_for_boxplot,
        labels=labels_for_boxplot,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False  # Hide outliers for cleaner visualization
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Angle (degrees)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token Rank', fontsize=14, fontweight='bold')
    ax.set_title(
        'Aggregated Angle Distribution Comparison (All Questions)',
        fontsize=16,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # 4. Statistics Summary
    # ============================================================
    print("\n" + "="*100)
    print("AGGREGATED STATISTICS (ALL QUESTIONS)")
    print("="*100)
    print(f"{'Rank':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Q25':<10} {'Q75':<10} {'Q95':<10} {'Count':<10}")
    print("-" * 100)
    
    for rank in ranks:
        angles = aggregated_angles[rank]
        print(f"Top-{rank:<5} {np.mean(angles):<10.2f} {np.median(angles):<10.2f} {np.std(angles):<10.2f} "
              f"{np.min(angles):<10.2f} {np.max(angles):<10.2f} {np.percentile(angles, 25):<10.2f} "
              f"{np.percentile(angles, 75):<10.2f} {np.percentile(angles, 95):<10.2f} {len(angles):<10}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="LPT to Unembedding Vector Angle Distribution Analysis")
    parser.add_argument('--model_path', type=str, default="/home/chashi/Research/Llama-3.1-8B-Instruct", help='Path to the Llama model')
    parser.add_argument('--exp4_dir', type=Path, default="/home/chashi/Research/llm_rounding_error_instability/exp4_generation_results_A6000_48GB", help='Path to the experiment 4 results directory')
    parser.add_argument('--output_dir', type=Path, default=Path("../results/exp4_part7_lpt_unembedding_angle_analysis"), help='Output directory for angle analysis')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to load model on')
    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================
    RANK_CATEGORIES = [1, 2, 3, 10, 100]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("LPT TO UNEMBEDDING VECTOR ANGLE ANALYSIS")
    print("="*100)
    print(f"Experiment directory: {args.exp4_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rank categories: {RANK_CATEGORIES}")
    print(f"Device: {args.device}")
    print("="*100)
    
    # ============================================================
    # Load Model for Unembedding Weights
    # ============================================================
    unembedding_weights, tokenizer = load_model_for_unembedding(args.model_path, args.device)
    
    # ============================================================
    # Find All Question Directories
    # ============================================================
    question_dirs = sorted([d for d in args.exp4_dir.iterdir() if d.is_dir() and d.name.startswith("question_")])
    
    if not question_dirs:
        print(f"\nERROR: No question directories found in {args.exp4_dir}")
        print("Please ensure the experiment directory contains question_XX subdirectories.")
        return
    
    print(f"\nFound {len(question_dirs)} question directories")
    
    # ============================================================
    # Process Each Question
    # ============================================================
    all_angles_by_rank = {rank: [] for rank in RANK_CATEGORIES}
    stats_by_question = {}
    
    for question_dir in question_dirs:
        question_id = int(question_dir.name.split("_")[1])
        
        print(f"\n{'='*80}")
        print(f"Processing Question {question_id}")
        print(f"{'='*80}")
        
        # Load question data
        question_data = load_question_data(question_dir)
        num_tokens = question_data['num_tokens']
        
        print(f"Number of tokens generated: {num_tokens}")
        print(f"Representation shape: {question_data['representations'].shape}")
        
        # Compute angles for ranks 1-10
        print("Computing angles for top-1, top-2, top-3, top-10...")
        angles_by_rank = compute_angles_for_question(
            question_data,
            unembedding_weights,
            rank_categories=[1, 2, 3, 10]
        )
        
        # Compute angles for top-100 (requires full logit computation)
        if 100 in RANK_CATEGORIES:
            print("Computing angles for top-100 (this may take a moment)...")
            angles_top100 = compute_angles_with_actual_top100(
                question_data,
                unembedding_weights
            )
            angles_by_rank[100] = angles_top100
        
        # Compute statistics
        stats = compute_statistics(angles_by_rank)
        stats_by_question[question_id] = stats
        
        # Print statistics for this question
        print("\nStatistics for this question:")
        for rank in sorted(angles_by_rank.keys()):
            print(f"  Top-{rank}: Mean={stats[rank]['mean']:.2f}°, "
                  f"Median={stats[rank]['median']:.2f}°, "
                  f"Std={stats[rank]['std']:.2f}°")
        
        # Store angles for aggregated analysis
        for rank in RANK_CATEGORIES:
            if rank in angles_by_rank:
                all_angles_by_rank[rank].append(angles_by_rank[rank])
        
        # Create plots for this question
        question_output_dir = args.output_dir / f"question_{question_id:02d}"
        plot_angle_distributions(
            angles_by_rank,
            question_id,
            question_output_dir,
            title_suffix=question_data['metadata'].get('input_text', '')[:100]
        )
    
    # ============================================================
    # Save Statistics
    # ============================================================
    save_statistics(stats_by_question, args.output_dir)
    
    # ============================================================
    # Create Aggregated Plots
    # ============================================================
    plot_aggregated_distributions(all_angles_by_rank, args.output_dir)
    
    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"Results saved to: {args.output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - Per-question plots: {args.output_dir}/question_XX/")
    print(f"  - Aggregated plots: {args.output_dir}/aggregated_*.png")
    print(f"  - Statistics: {args.output_dir}/angle_statistics.json")
    print("="*100)


if __name__ == "__main__":
    main()