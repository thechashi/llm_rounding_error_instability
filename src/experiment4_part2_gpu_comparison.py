import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import pandas as pd

def load_question_data(question_dir: Path) -> Dict:
    """Load all data for a single question"""
    data = {
        'representations': np.load(question_dir / "representations.npy"),
        'top_10_logits': np.load(question_dir / "top_10_logits.npy"),
        'top_10_probs': np.load(question_dir / "top_10_probs.npy"),
    }
    
    with open(question_dir / "words.json", 'r') as f:
        words_data = json.load(f)
        data['top_5_words'] = words_data['top_5_words']
        data['generated_words'] = words_data['generated_words']
        data['top_10_token_ids'] = np.array(words_data['top_10_token_ids'])
    
    with open(question_dir / "metadata.json", 'r') as f:
        data['metadata'] = json.load(f)
    
    return data

def compare_representations(repr1: np.ndarray, repr2: np.ndarray) -> Dict:
    """Compare two sets of representations"""
    num_tokens = min(len(repr1), len(repr2))
    
    # Calculate metrics for each token position
    cosine_sims = []
    l2_distances = []
    l1_distances = []
    
    for i in range(num_tokens):
        # Cosine similarity (using 1 - cosine_distance)
        cos_sim = 1 - cosine(repr1[i], repr2[i])
        cosine_sims.append(cos_sim)
        
        # L2 (Euclidean) distance
        l2_dist = euclidean(repr1[i], repr2[i])
        l2_distances.append(l2_dist)
        
        # L1 (Manhattan) distance
        l1_dist = np.sum(np.abs(repr1[i] - repr2[i]))
        l1_distances.append(l1_dist)
    
    cosine_sims = np.array(cosine_sims)
    l2_distances = np.array(l2_distances)
    l1_distances = np.array(l1_distances)
    
    # Calculate summary statistics
    results = {
        'num_tokens_compared': num_tokens,
        'cosine_similarity': {
            'mean': float(np.mean(cosine_sims)),
            'std': float(np.std(cosine_sims)),
            'min': float(np.min(cosine_sims)),
            'max': float(np.max(cosine_sims)),
            'median': float(np.median(cosine_sims)),
            'q25': float(np.percentile(cosine_sims, 25)),
            'q75': float(np.percentile(cosine_sims, 75))
        },
        'l2_distance': {
            'mean': float(np.mean(l2_distances)),
            'std': float(np.std(l2_distances)),
            'min': float(np.min(l2_distances)),
            'max': float(np.max(l2_distances)),
            'median': float(np.median(l2_distances)),
            'q25': float(np.percentile(l2_distances, 25)),
            'q75': float(np.percentile(l2_distances, 75))
        },
        'l1_distance': {
            'mean': float(np.mean(l1_distances)),
            'std': float(np.std(l1_distances)),
            'min': float(np.min(l1_distances)),
            'max': float(np.max(l1_distances)),
            'median': float(np.median(l1_distances)),
            'q25': float(np.percentile(l1_distances, 25)),
            'q75': float(np.percentile(l1_distances, 75))
        },
        'per_token_cosine_sim': cosine_sims.tolist(),
        'per_token_l2_dist': l2_distances.tolist()
    }
    
    return results

def compare_logits_and_probs(logits1: np.ndarray, logits2: np.ndarray,
                              probs1: np.ndarray, probs2: np.ndarray) -> Dict:
    """Compare logits and probabilities"""
    num_tokens = min(len(logits1), len(logits2))
    
    # Calculate differences for each token
    logit_l2_dists = []
    logit_l1_dists = []
    prob_l2_dists = []
    prob_l1_dists = []
    logit_correlations = []
    prob_correlations = []
    
    for i in range(num_tokens):
        # Logit comparisons
        logit_l2 = euclidean(logits1[i], logits2[i])
        logit_l1 = np.sum(np.abs(logits1[i] - logits2[i]))
        logit_l2_dists.append(logit_l2)
        logit_l1_dists.append(logit_l1)
        
        # Probability comparisons
        prob_l2 = euclidean(probs1[i], probs2[i])
        prob_l1 = np.sum(np.abs(probs1[i] - probs2[i]))
        prob_l2_dists.append(prob_l2)
        prob_l1_dists.append(prob_l1)
        
        # Correlations (Pearson)
        if len(logits1[i]) > 1:
            logit_corr, _ = pearsonr(logits1[i], logits2[i])
            prob_corr, _ = pearsonr(probs1[i], probs2[i])
            logit_correlations.append(logit_corr)
            prob_correlations.append(prob_corr)
    
    results = {
        'num_tokens_compared': num_tokens,
        'logits': {
            'l2_distance': {
                'mean': float(np.mean(logit_l2_dists)),
                'std': float(np.std(logit_l2_dists)),
                'min': float(np.min(logit_l2_dists)),
                'max': float(np.max(logit_l2_dists))
            },
            'l1_distance': {
                'mean': float(np.mean(logit_l1_dists)),
                'std': float(np.std(logit_l1_dists)),
                'min': float(np.min(logit_l1_dists)),
                'max': float(np.max(logit_l1_dists))
            },
            'correlation': {
                'mean': float(np.mean(logit_correlations)),
                'std': float(np.std(logit_correlations)),
                'min': float(np.min(logit_correlations)),
                'max': float(np.max(logit_correlations))
            }
        },
        'probabilities': {
            'l2_distance': {
                'mean': float(np.mean(prob_l2_dists)),
                'std': float(np.std(prob_l2_dists)),
                'min': float(np.min(prob_l2_dists)),
                'max': float(np.max(prob_l2_dists))
            },
            'l1_distance': {
                'mean': float(np.mean(prob_l1_dists)),
                'std': float(np.std(prob_l1_dists)),
                'min': float(np.min(prob_l1_dists)),
                'max': float(np.max(prob_l1_dists))
            },
            'correlation': {
                'mean': float(np.mean(prob_correlations)),
                'std': float(np.std(prob_correlations)),
                'min': float(np.min(prob_correlations)),
                'max': float(np.max(prob_correlations))
            }
        }
    }
    
    return results

def compare_words(words1_data: Dict, words2_data: Dict) -> Dict:
    """Compare generated words and top words"""
    generated1 = words1_data['generated_words']
    generated2 = words2_data['generated_words']
    top5_words1 = words1_data['top_5_words']
    top5_words2 = words2_data['top_5_words']
    token_ids1 = words1_data['top_10_token_ids']
    token_ids2 = words2_data['top_10_token_ids']
    
    num_tokens = min(len(generated1), len(generated2))
    
    # Compare generated words (top-1)
    generated_matches = sum(1 for i in range(num_tokens) if generated1[i] == generated2[i])
    generated_mismatch_rate = (num_tokens - generated_matches) / num_tokens if num_tokens > 0 else 0
    
    # Compare top-5 words overlap
    top5_overlaps = []
    for i in range(num_tokens):
        overlap = len(set(top5_words1[i]) & set(top5_words2[i]))
        top5_overlaps.append(overlap)
    
    # Compare top-10 token IDs
    top10_id_matches = []
    for i in range(num_tokens):
        matches = sum(1 for j in range(10) if token_ids1[i][j] == token_ids2[i][j])
        top10_id_matches.append(matches)
    
    # Find positions where generated words differ
    mismatch_positions = [i for i in range(num_tokens) if generated1[i] != generated2[i]]
    
    results = {
        'num_tokens_compared': num_tokens,
        'generated_words': {
            'total_matches': generated_matches,
            'total_mismatches': num_tokens - generated_matches,
            'match_rate': generated_matches / num_tokens if num_tokens > 0 else 0,
            'mismatch_rate': generated_mismatch_rate,
            'mismatch_positions': mismatch_positions[:50]  # First 50 mismatches
        },
        'top_5_words_overlap': {
            'mean_overlap': float(np.mean(top5_overlaps)),
            'std_overlap': float(np.std(top5_overlaps)),
            'min_overlap': int(np.min(top5_overlaps)),
            'max_overlap': int(np.max(top5_overlaps))
        },
        'top_10_token_ids': {
            'mean_matches': float(np.mean(top10_id_matches)),
            'std_matches': float(np.std(top10_id_matches)),
            'min_matches': int(np.min(top10_id_matches)),
            'max_matches': int(np.max(top10_id_matches)),
            'all_10_match_rate': sum(1 for m in top10_id_matches if m == 10) / num_tokens if num_tokens > 0 else 0
        }
    }
    
    return results

def compare_question(question_dir1: Path, question_dir2: Path, question_id: int) -> Dict:
    """Compare all data for a single question"""
    print(f"\nComparing Question {question_id}...")
    
    # Load data from both machines
    data1 = load_question_data(question_dir1)
    data2 = load_question_data(question_dir2)
    
    # Compare representations
    print("  - Comparing representations...")
    repr_comparison = compare_representations(data1['representations'], data2['representations'])
    
    # Compare logits and probabilities
    print("  - Comparing logits and probabilities...")
    logits_probs_comparison = compare_logits_and_probs(
        data1['top_10_logits'], data2['top_10_logits'],
        data1['top_10_probs'], data2['top_10_probs']
    )
    
    # Compare words
    print("  - Comparing words...")
    words_comparison = compare_words(
        {'generated_words': data1['generated_words'],
         'top_5_words': data1['top_5_words'],
         'top_10_token_ids': data1['top_10_token_ids']},
        {'generated_words': data2['generated_words'],
         'top_5_words': data2['top_5_words'],
         'top_10_token_ids': data2['top_10_token_ids']}
    )
    
    return {
        'question_id': question_id,
        'question_text': data1['metadata']['input_text'],
        'representations': repr_comparison,
        'logits_and_probs': logits_probs_comparison,
        'words': words_comparison
    }

def aggregate_results(all_comparisons: List[Dict]) -> Dict:
    """Aggregate statistics across all questions"""
    
    # Aggregate representation metrics
    all_cosine_means = [c['representations']['cosine_similarity']['mean'] for c in all_comparisons]
    all_l2_means = [c['representations']['l2_distance']['mean'] for c in all_comparisons]
    all_l1_means = [c['representations']['l1_distance']['mean'] for c in all_comparisons]
    
    # Aggregate word match rates
    all_match_rates = [c['words']['generated_words']['match_rate'] for c in all_comparisons]
    all_mismatch_rates = [c['words']['generated_words']['mismatch_rate'] for c in all_comparisons]
    
    # Aggregate logit correlations
    all_logit_corrs = [c['logits_and_probs']['logits']['correlation']['mean'] for c in all_comparisons]
    all_prob_corrs = [c['logits_and_probs']['probabilities']['correlation']['mean'] for c in all_comparisons]
    
    return {
        'num_questions': len(all_comparisons),
        'representations': {
            'cosine_similarity': {
                'overall_mean': float(np.mean(all_cosine_means)),
                'overall_std': float(np.std(all_cosine_means)),
                'overall_min': float(np.min(all_cosine_means)),
                'overall_max': float(np.max(all_cosine_means))
            },
            'l2_distance': {
                'overall_mean': float(np.mean(all_l2_means)),
                'overall_std': float(np.std(all_l2_means)),
                'overall_min': float(np.min(all_l2_means)),
                'overall_max': float(np.max(all_l2_means))
            },
            'l1_distance': {
                'overall_mean': float(np.mean(all_l1_means)),
                'overall_std': float(np.std(all_l1_means)),
                'overall_min': float(np.min(all_l1_means)),
                'overall_max': float(np.max(all_l1_means))
            }
        },
        'word_matching': {
            'match_rate': {
                'overall_mean': float(np.mean(all_match_rates)),
                'overall_std': float(np.std(all_match_rates)),
                'overall_min': float(np.min(all_match_rates)),
                'overall_max': float(np.max(all_match_rates))
            },
            'mismatch_rate': {
                'overall_mean': float(np.mean(all_mismatch_rates)),
                'overall_std': float(np.std(all_mismatch_rates)),
                'overall_min': float(np.min(all_mismatch_rates)),
                'overall_max': float(np.max(all_mismatch_rates))
            }
        },
        'correlations': {
            'logits': {
                'overall_mean': float(np.mean(all_logit_corrs)),
                'overall_std': float(np.std(all_logit_corrs))
            },
            'probabilities': {
                'overall_mean': float(np.mean(all_prob_corrs)),
                'overall_std': float(np.std(all_prob_corrs))
            }
        }
    }

def print_summary(comparison: Dict):
    """Print a human-readable summary of comparisons"""
    print("\n" + "="*80)
    print(f"COMPARISON SUMMARY FOR QUESTION {comparison['question_id']}")
    print("="*80)
    print(f"Question: {comparison['question_text'][:100]}...")
    print()
    
    # Representations
    repr = comparison['representations']
    print("REPRESENTATIONS:")
    print(f"  Cosine Similarity:  mean={repr['cosine_similarity']['mean']:.6f}, std={repr['cosine_similarity']['std']:.6f}")
    print(f"  L2 Distance:        mean={repr['l2_distance']['mean']:.6f}, std={repr['l2_distance']['std']:.6f}")
    print(f"  L1 Distance:        mean={repr['l1_distance']['mean']:.6f}, std={repr['l1_distance']['std']:.6f}")
    print()
    
    # Words
    words = comparison['words']
    print("WORDS:")
    print(f"  Generated Word Match Rate:  {words['generated_words']['match_rate']*100:.2f}%")
    print(f"  Total Matches:              {words['generated_words']['total_matches']}")
    print(f"  Total Mismatches:           {words['generated_words']['total_mismatches']}")
    print(f"  Top-5 Words Overlap:        mean={words['top_5_words_overlap']['mean_overlap']:.2f}/5")
    print(f"  Top-10 IDs Match:           mean={words['top_10_token_ids']['mean_matches']:.2f}/10")
    print()
    
    # Logits & Probs
    lp = comparison['logits_and_probs']
    print("LOGITS & PROBABILITIES:")
    print(f"  Logit Correlation:     mean={lp['logits']['correlation']['mean']:.6f}")
    print(f"  Prob Correlation:      mean={lp['probabilities']['correlation']['mean']:.6f}")
    print(f"  Logit L2 Distance:     mean={lp['logits']['l2_distance']['mean']:.6f}")
    print(f"  Prob L2 Distance:      mean={lp['probabilities']['l2_distance']['mean']:.6f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare generation results from two machines')
    parser.add_argument('folder1', type=str, help='Path to first results folder')
    parser.add_argument('folder2', type=str, help='Path to second results folder')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                       help='Output JSON file for detailed results')
    parser.add_argument('--num-questions', type=int, default=10,
                       help='Number of questions to compare')
    
    args = parser.parse_args()
    
    folder1 = Path(args.folder1)
    folder2 = Path(args.folder2)
    
    if not folder1.exists():
        raise FileNotFoundError(f"Folder 1 not found: {folder1}")
    if not folder2.exists():
        raise FileNotFoundError(f"Folder 2 not found: {folder2}")
    
    print("="*80)
    print("COMPARING GENERATION RESULTS FROM TWO MACHINES")
    print("="*80)
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print(f"Number of questions: {args.num_questions}")
    
    # Compare all questions
    all_comparisons = []
    for i in range(1, args.num_questions + 1):
        question_dir1 = folder1 / f"question_{i:02d}"
        question_dir2 = folder2 / f"question_{i:02d}"
        
        if not question_dir1.exists():
            print(f"\nWarning: Question {i} not found in folder 1, skipping...")
            continue
        if not question_dir2.exists():
            print(f"\nWarning: Question {i} not found in folder 2, skipping...")
            continue
        
        comparison = compare_question(question_dir1, question_dir2, i)
        all_comparisons.append(comparison)
        print_summary(comparison)
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS ALL QUESTIONS")
    print("="*80)
    
    aggregate = aggregate_results(all_comparisons)
    
    print("\nREPRESENTATIONS (Averaged across all questions):")
    print(f"  Cosine Similarity:  {aggregate['representations']['cosine_similarity']['overall_mean']:.6f} ± {aggregate['representations']['cosine_similarity']['overall_std']:.6f}")
    print(f"  L2 Distance:        {aggregate['representations']['l2_distance']['overall_mean']:.6f} ± {aggregate['representations']['l2_distance']['overall_std']:.6f}")
    print(f"  L1 Distance:        {aggregate['representations']['l1_distance']['overall_mean']:.6f} ± {aggregate['representations']['l1_distance']['overall_std']:.6f}")
    
    print("\nWORD MATCHING (Averaged across all questions):")
    print(f"  Match Rate:         {aggregate['word_matching']['match_rate']['overall_mean']*100:.2f}% ± {aggregate['word_matching']['match_rate']['overall_std']*100:.2f}%")
    print(f"  Mismatch Rate:      {aggregate['word_matching']['mismatch_rate']['overall_mean']*100:.2f}% ± {aggregate['word_matching']['mismatch_rate']['overall_std']*100:.2f}%")
    
    print("\nCORRELATIONS (Averaged across all questions):")
    print(f"  Logit Correlation:  {aggregate['correlations']['logits']['overall_mean']:.6f} ± {aggregate['correlations']['logits']['overall_std']:.6f}")
    print(f"  Prob Correlation:   {aggregate['correlations']['probabilities']['overall_mean']:.6f} ± {aggregate['correlations']['probabilities']['overall_std']:.6f}")
    
    # Save detailed results
    output_data = {
        'per_question_comparisons': all_comparisons,
        'aggregate_statistics': aggregate
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {args.output}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

'''
python src/experiment4_part2_comaprison_GPUS.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results" "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results" --output exp4_comparison.json --num-questions 5
'''