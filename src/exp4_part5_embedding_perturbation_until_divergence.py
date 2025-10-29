"""
Experiment 4 Part 5: Input Embedding Analysis Until Divergence

This script analyzes the INPUT EMBEDDINGS (not hidden representations) for tokens
leading up to divergence points, testing if the embeddings themselves differ
between GPU runs or if differences only emerge during forward propagation.

Purpose:
--------
Distinguishes between two sources of GPU-specific differences:
1. EMBEDDING DIFFERENCES: Do input token embeddings differ between GPUs?
2. COMPUTATIONAL DIFFERENCES: Do identical embeddings produce different hidden
   states due to different forward pass computations?

This helps determine whether divergence is due to:
- Embedding matrix storage/retrieval differences (unlikely but possible)
- Forward pass computation differences (more likely - matmul, attention, etc.)


This helps separate:
- Model weight loading issues (if embeddings differ)
- Computation issues (if embeddings identical but outputs differ)

Methodology:
------------
1. Load Llama model on current GPU
2. For each question from part2 that showed divergence:
   a. Load metadata to get prompt and generated tokens
   b. Reconstruct input sequence up to divergence point
   c. Convert tokens to embeddings using model's embedding layer
   d. Save embeddings to disk
3. Run this script on GPU 0 and GPU 1 separately
4. Use part6 to compare the saved embeddings

Test Design:
------------
1. Both GPUs load the same model checkpoint
2. Extract embeddings for SAME token IDs
3. Compare resulting embedding vectors
4. Expected: Embeddings should be IDENTICAL (deterministic lookup)
5. If different: Model loading issue or embedding matrix differences
6. If identical: Divergence caused by forward pass computations

Use Case:
---------
Use this script to:
- Verify that embedding matrices are loaded identically on different GPUs
- Rule out embedding differences as source of divergence
- Focus investigation on computational differences if embeddings match
- Identify model loading issues if embeddings differ

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, json, pathlib

Key Functions:
--------------
- load_model(): Load model on current GPU
- load_question_metadata(): Get prompt and generated tokens
- extract_and_save_embeddings(): Main extraction logic
  * Tokenizes input sequence
  * Extracts embeddings from embedding layer
  * Saves to disk for comparison

Output:
-------
- Directory: results/exp4_embeddings_gpu{N}_TIMESTAMP/
- Per-question files: question_{id}_embeddings.npy
- Metadata: embeddings_metadata.json (prompts, tokens, indices)

Workflow:
---------
1. Run experiment4_GPU_comaprison.py on GPU 0 and GPU 1
2. Run experiment4_part2 to identify divergences
3. Run THIS script on GPU 0 → saves embeddings
4. Run THIS script on GPU 1 → saves embeddings
5. Run part6 to compare the two embedding sets

Note:
-----
If embeddings are IDENTICAL between GPUs (expected):
- Divergence must be caused by forward pass computations
- Focus on matmul, attention, normalization operations

If embeddings DIFFER between GPUs (unexpected):
- Model loading or weight precision issues
- Embedding matrix storage differences
- Needs further investigation of model checkpoint loading
"""

import torch
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path: str):
    """Load model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer

def load_question_metadata(question_dir: Path) -> Dict:
    """Load metadata and generated words for a question"""
    with open(question_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(question_dir / "words.json", 'r') as f:
        words_data = json.load(f)
    
    return {
        'input_text': metadata['input_text'],
        'generated_words': words_data['generated_words'],
        'full_text': metadata['full_generated_text']
    }

def get_text_till_divergence(question_data: Dict, divergence_idx: int) -> str:
    """
    Construct text till divergence_idx + 1
    
    Args:
        question_data: Dictionary with 'input_text' and 'generated_words'
        divergence_idx: The index where divergence occurred (in generated tokens)
    
    Returns:
        Text containing input + generated words up to and including divergence_idx
    """
    input_text = question_data['input_text']
    generated_words = question_data['generated_words']
    
    # Get generated words till divergence_idx + 1 (inclusive)
    words_till_divergence = generated_words[:divergence_idx + 1]
    
    # Concatenate
    text_till_divergence = input_text + ''.join(words_till_divergence)
    
    return text_till_divergence

def extract_input_embeddings(
    model,
    tokenizer,
    text: str,
    question_id: int
) -> Tuple[np.ndarray, Dict]:
    """
    Extract input embeddings for the given text
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to get embeddings for
        question_id: Question identifier
    
    Returns:
        Tuple of (embeddings_array, metadata_dict)
        - embeddings_array: [num_tokens, embedding_dim] with float64 precision
        - metadata_dict: Information about the extraction
    """
    device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    input_ids = inputs['input_ids'].to(device)
    num_tokens = input_ids.shape[1]
    
    print(f"  Question {question_id}: {num_tokens} tokens")
    
    # Get embeddings from the embedding layer
    with torch.no_grad():
        # Access the embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embed_layer = model.transformer.wte
        else:
            raise AttributeError("Could not find embedding layer in model")
        
        # Get embeddings [1, num_tokens, embedding_dim]
        embeddings = embed_layer(input_ids)
        
        # Convert to numpy with maximum precision (float64)
        embeddings_np = embeddings.squeeze(0).cpu().to(torch.float64).numpy()
    
    # Decode tokens for verification
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    
    metadata = {
        'question_id': question_id,
        'num_tokens': num_tokens,
        'embedding_dim': embeddings_np.shape[1],
        'text': text,
        'tokens': tokens,
        'token_ids': input_ids[0].cpu().tolist()
    }
    
    return embeddings_np, metadata

def load_divergence_indices_from_json(comparison_json_path: Path) -> Dict[int, int]:
    """
    Load divergence indices from the comparison results JSON
    
    Args:
        comparison_json_path: Path to the comparison results JSON file
    
    Returns:
        Dictionary mapping question_id -> divergence_index
    """
    with open(comparison_json_path, 'r') as f:
        comparison_data = json.load(f)
    
    divergence_dict = {}
    
    for question_comparison in comparison_data['per_question_comparisons']:
        question_id = question_comparison['question_id']
        
        if 'divergence_analysis' in question_comparison:
            divergence_idx = question_comparison['divergence_analysis']['divergence_index']
            divergence_dict[question_id] = divergence_idx
            print(f"Question {question_id}: Divergence at index {divergence_idx}")
        else:
            print(f"Question {question_id}: No divergence found (skipping)")
    
    return divergence_dict

def load_divergence_indices_from_dict(divergence_dict_path: Path) -> Dict[int, int]:
    """
    Load divergence indices from a simple JSON dict file
    Format: {"1": 52, "2": 30, ...}
    """
    with open(divergence_dict_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to integers
    divergence_dict = {int(k): int(v) for k, v in data.items()}
    
    print("Loaded divergence indices:")
    for qid, div_idx in sorted(divergence_dict.items()):
        print(f"  Question {qid}: Divergence at index {div_idx}")
    
    return divergence_dict

def save_embeddings_dataset(
    all_embeddings: Dict[int, np.ndarray],
    all_metadata: Dict[int, Dict],
    output_dir: Path
):
    """
    Save embeddings dataset with maximum precision
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual embedding files with maximum precision
    for question_id, embeddings in all_embeddings.items():
        embedding_file = output_dir / f"question_{question_id:02d}_embeddings.npy"
        np.save(embedding_file, embeddings)
        print(f"Saved: {embedding_file} - Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    
    # Save all metadata
    metadata_file = output_dir / "embeddings_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved metadata: {metadata_file}")
    
    # Create a summary
    summary = {
        'num_questions': len(all_embeddings),
        'questions': {}
    }
    
    for question_id, embeddings in all_embeddings.items():
        summary['questions'][question_id] = {
            'shape': list(embeddings.shape),
            'num_tokens': int(embeddings.shape[0]),
            'embedding_dim': int(embeddings.shape[1]),
            'dtype': str(embeddings.dtype),
            'file': f"question_{question_id:02d}_embeddings.npy"
        }
    
    summary_file = output_dir / "embeddings_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_file}")
    
    # Also save a combined file
    combined_file = output_dir / "all_embeddings.npz"
    np.savez_compressed(
        combined_file,
        **{f"question_{qid:02d}": emb for qid, emb in all_embeddings.items()}
    )
    print(f"Saved combined file: {combined_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract input embeddings till divergence index + 1 for all questions'
    )
    parser.add_argument(
        'results_folder',
        type=str,
        help='Path to results folder (from first script)'
    )
    parser.add_argument(
        'divergence_source',
        type=str,
        help='Path to divergence indices (comparison JSON or simple dict JSON)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for embeddings (default: timestamped in ../results/)'
    )
    parser.add_argument(
        '--source-type',
        type=str,
        choices=['comparison', 'dict'],
        default='comparison',
        help='Type of divergence source: "comparison" (full comparison JSON) or "dict" (simple dict)'
    )

    args = parser.parse_args()

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp4_part5_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(exp_dir)
    else:
        output_dir = Path(os.path.join(exp_dir, args.output_dir))

    results_folder = Path(args.results_folder)
    divergence_source = Path(args.divergence_source)
    
    # Validate inputs
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")
    if not divergence_source.exists():
        raise FileNotFoundError(f"Divergence source not found: {divergence_source}")
    
    print("="*80)
    print("EXTRACTING INPUT EMBEDDINGS TILL DIVERGENCE INDEX")
    print("="*80)
    print(f"Results Folder: {results_folder}")
    print(f"Divergence Source: {divergence_source}")
    print(f"Source Type: {args.source_type}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_model(args.model_path)
    print()
    
    # Load divergence indices
    print("="*80)
    print("LOADING DIVERGENCE INDICES")
    print("="*80)
    
    if args.source_type == 'comparison':
        divergence_dict = load_divergence_indices_from_json(divergence_source)
    else:  # dict
        divergence_dict = load_divergence_indices_from_dict(divergence_source)
    
    print(f"Found {len(divergence_dict)} questions with divergence")
    print()
    
    # Process each question
    print("="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    
    all_embeddings = {}
    all_metadata = {}
    
    for question_id, divergence_idx in sorted(divergence_dict.items()):
        # Load question data
        question_dir = results_folder / f"question_{question_id:02d}"
        if not question_dir.exists():
            print(f"Question {question_id}: Skipping (folder not found)")
            continue
        
        print(f"\nProcessing Question {question_id} (divergence at index {divergence_idx})...")
        
        try:
            # Load question data
            question_data = load_question_metadata(question_dir)
            
            # Get text till divergence + 1
            text_till_divergence = get_text_till_divergence(question_data, divergence_idx)
            
            print(f"  Input text length: {len(question_data['input_text'])} chars")
            print(f"  Total text length: {len(text_till_divergence)} chars")
            print(f"  Generated words included: {divergence_idx + 1}")
            
            # Extract embeddings
            embeddings, metadata = extract_input_embeddings(
                model, tokenizer, text_till_divergence, question_id
            )
            
            # Add divergence info to metadata
            metadata['divergence_index'] = divergence_idx
            metadata['num_generated_words_included'] = divergence_idx + 1
            
            # Store
            all_embeddings[question_id] = embeddings
            all_metadata[question_id] = metadata
            
            print(f"  ✓ Extracted embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
            
        except Exception as e:
            print(f"  ✗ Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all embeddings
    print("\n" + "="*80)
    print("SAVING EMBEDDINGS DATASET")
    print("="*80)
    save_embeddings_dataset(all_embeddings, all_metadata, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Processed {len(all_embeddings)} questions")
    print(f"Embeddings saved to: {output_dir.absolute()}")
    print("\nDataset structure:")
    print(f"  - Individual files: question_XX_embeddings.npy (float64 precision)")
    print(f"  - Metadata: embeddings_metadata.json")
    print(f"  - Summary: embeddings_summary.json")
    print(f"  - Combined: all_embeddings.npz")
    print()

if __name__ == "__main__":
    main()

"""
Example usage:

Option 1: Using the full comparison JSON from the second script
python src/experiment4_part5_input_embeddings_till_div.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A5000_2x24GB" \
    "exp4_comparison.json" \
    --model-path "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct" \
    --output-dir "embeddings_till_divergence_A5000" \
    --source-type comparison

Option 2: Using a simple divergence dict JSON
First create a file like divergence_indices.json:
{
    "1": 52,
    "2": 30,
    "3": 45,
    ...
}

Then run:
python src/experiment4_part5_input_embeddings_till_div.py \
    "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/exp4_generation_results_A5000_2x24GB" \
    "divergence_indices.json" \
    --model-path "/path/to/Llama-3.1-8B-Instruct" \
    --output-dir "embeddings_till_divergence_A5000" \
    --source-type dict
"""