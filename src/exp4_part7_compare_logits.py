import numpy as np
import json
from pathlib import Path
import pandas as pd
import os
from datetime import datetime

def load_question_data_at_indices(question_dir: Path, indices: list):
    """Load specific data at given indices for a single question"""
    data = {}
    
    # Load the necessary arrays
    data['top_10_logits'] = np.load(question_dir / "top_10_logits.npy")
    data['top_10_probs'] = np.load(question_dir / "top_10_probs.npy")
    
    with open(question_dir / "words.json", 'r') as f:
        words_data = json.load(f)
        data['top_5_words'] = words_data['top_5_words']
        data['generated_words'] = words_data['generated_words']
        data['top_10_token_ids'] = np.array(words_data['top_10_token_ids'])
    
    # Extract data only at specified indices
    result = {}
    for idx in indices:
        if 0 <= idx < len(data['generated_words']):
            result[idx] = {
                'top_5_logits': data['top_10_logits'][idx][:5],  # First 5 of top 10
                'top_5_probs': data['top_10_probs'][idx][:5],
                'top_5_words': data['top_5_words'][idx],
                'generated_word': data['generated_words'][idx],
                'top_5_token_ids': data['top_10_token_ids'][idx][:5]
            }
    
    return result

def find_divergence_index(generated1: list, generated2: list):
    """Find the first index where generated words diverge"""
    num_tokens = min(len(generated1), len(generated2))
    
    for i in range(num_tokens):
        if generated1[i] != generated2[i]:
            return i
    
    return None  # No divergence found

def compare_top5_at_divergence(folder1: Path, folder2: Path, question_id: int):
    """Compare top-5 logits and words at divergence and previous indices"""
    
    # Construct question directories
    question_dir1 = folder1 / f"question_{question_id:02d}"
    question_dir2 = folder2 / f"question_{question_id:02d}"
    
    if not question_dir1.exists() or not question_dir2.exists():
        print(f"Question {question_id} not found in one or both folders")
        return None
    
    # Load generated words to find divergence
    with open(question_dir1 / "words.json", 'r') as f:
        words1 = json.load(f)['generated_words']
    
    with open(question_dir2 / "words.json", 'r') as f:
        words2 = json.load(f)['generated_words']
    
    # Find divergence index
    div_idx = find_divergence_index(words1, words2)
    
    if div_idx is None:
        print(f"No divergence found for question {question_id}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Question {question_id}: Divergence at index {div_idx}")
    print(f"{'='*80}")
    
    # Determine indices to analyze
    indices_to_analyze = []
    if div_idx > 0:
        indices_to_analyze.append(div_idx - 1)  # Previous index
    indices_to_analyze.append(div_idx)  # Divergence index
    
    # Load data at these indices
    data1 = load_question_data_at_indices(question_dir1, indices_to_analyze)
    data2 = load_question_data_at_indices(question_dir2, indices_to_analyze)
    
    results = {}
    
    for idx in indices_to_analyze:
        if idx not in data1 or idx not in data2:
            continue
            
        is_divergence = (idx == div_idx)
        index_type = "DIVERGENCE INDEX" if is_divergence else "PREVIOUS INDEX"
        
        print(f"\n{'-'*60}")
        print(f"{index_type}: Position {idx}")
        print(f"{'-'*60}")
        
        # Compare generated words
        print(f"\nGenerated Words:")
        print(f"  GPU 1: '{data1[idx]['generated_word']}'")
        print(f"  GPU 2: '{data2[idx]['generated_word']}'")
        print(f"  Match: {data1[idx]['generated_word'] == data2[idx]['generated_word']}")
        
        # Compare top-5 logits
        print(f"\nTop-5 Logits Comparison:")
        logit_diff = data1[idx]['top_5_logits'] - data2[idx]['top_5_logits']
        
        comparison_df = pd.DataFrame({
            'Rank': range(1, 6),
            'GPU1_Word': data1[idx]['top_5_words'],
            'GPU1_Logit': data1[idx]['top_5_logits'],
            'GPU2_Word': data2[idx]['top_5_words'],
            'GPU2_Logit': data2[idx]['top_5_logits'],
            'Logit_Diff': logit_diff,
            'GPU1_Prob': data1[idx]['top_5_probs'],
            'GPU2_Prob': data2[idx]['top_5_probs'],
        })
        
        print(comparison_df.to_string(index=False))
        
        # Analyze word overlap in top-5
        words_set1 = set(data1[idx]['top_5_words'])
        words_set2 = set(data2[idx]['top_5_words'])
        overlap = words_set1 & words_set2
        only_gpu1 = words_set1 - words_set2
        only_gpu2 = words_set2 - words_set1
        
        print(f"\nTop-5 Words Analysis:")
        print(f"  Overlap: {len(overlap)}/5 words - {overlap if overlap else 'None'}")
        print(f"  Only in GPU1: {only_gpu1 if only_gpu1 else 'None'}")
        print(f"  Only in GPU2: {only_gpu2 if only_gpu2 else 'None'}")
        
        # Calculate metrics
        logit_l2_dist = np.linalg.norm(data1[idx]['top_5_logits'] - data2[idx]['top_5_logits'])
        prob_l2_dist = np.linalg.norm(data1[idx]['top_5_probs'] - data2[idx]['top_5_probs'])
        max_logit_diff = np.max(np.abs(logit_diff))
        
        print(f"\nNumerical Metrics:")
        print(f"  Logit L2 Distance: {logit_l2_dist:.6f}")
        print(f"  Prob L2 Distance: {prob_l2_dist:.6f}")
        print(f"  Max Logit Difference: {max_logit_diff:.6f}")
        print(f"  Mean Logit Difference: {np.mean(np.abs(logit_diff)):.6f}")
        
        # Store results
        results[index_type] = {
            'index': idx,
            'gpu1_word': data1[idx]['generated_word'],
            'gpu2_word': data2[idx]['generated_word'],
            'top5_overlap': len(overlap),
            'logit_l2_dist': logit_l2_dist,
            'prob_l2_dist': prob_l2_dist,
            'max_logit_diff': max_logit_diff,
            'comparison_df': comparison_df
        }
    
    return results

def analyze_all_questions(folder1: Path, folder2: Path, num_questions: int = 10, save_dir: Path = None):
    """Analyze divergence for all questions and create summary"""
    
    all_results = {}
    divergence_summary = []
    
    for q_id in range(1, num_questions + 1):
        result = compare_top5_at_divergence(folder1, folder2, q_id)
        if result:
            all_results[q_id] = result
            
            # Collect summary statistics
            if 'DIVERGENCE INDEX' in result:
                div_data = result['DIVERGENCE INDEX']
                divergence_summary.append({
                    'Question': q_id,
                    'Divergence_Index': div_data['index'],
                    'Top5_Overlap': div_data['top5_overlap'],
                    'Logit_L2_Dist': div_data['logit_l2_dist'],
                    'Max_Logit_Diff': div_data['max_logit_diff'],
                    'GPU1_Word': div_data['gpu1_word'],
                    'GPU2_Word': div_data['gpu2_word']
                })
    
    # Print summary
    if divergence_summary:
        print(f"\n{'='*80}")
        print("DIVERGENCE SUMMARY ACROSS ALL QUESTIONS")
        print(f"{'='*80}")
        
        summary_df = pd.DataFrame(divergence_summary)
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*80}")
        print(f"Questions with divergence: {len(divergence_summary)}/{num_questions}")
        print(f"Average divergence index: {summary_df['Divergence_Index'].mean():.2f}")
        print(f"Average top-5 overlap at divergence: {summary_df['Top5_Overlap'].mean():.2f}/5")
        print(f"Average logit L2 distance: {summary_df['Logit_L2_Dist'].mean():.6f}")
        print(f"Average max logit difference: {summary_df['Max_Logit_Diff'].mean():.6f}")
    
    return all_results, summary_df if divergence_summary else None

if __name__ == "__main__":
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(f"../results/exp7_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set your folder paths here
    gpu1_folder = Path("/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/old/old/exp4_generation_results_A5000_2x24GB")  # Replace with your GPU0 folder
    gpu2_folder = Path("/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/results/old/old/exp4_generation_results_A6000_48GB")  # Replace with your GPU1 folder
    
    # Analyze all questions
    print("\n\nALL QUESTIONS ANALYSIS")
    all_results, summary = analyze_all_questions(
        gpu1_folder, 
        gpu2_folder, 
        num_questions=10,
        save_dir=exp_dir
    )