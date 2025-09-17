import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Any, Tuple, Optional
import re
import os

warnings.filterwarnings('ignore')

def load_model(model_path: str):
    """Load model and tokenizer with proper configuration"""
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
        torch_dtype=torch.float16,
        trust_remote_code=True,
        output_hidden_states=True,
        low_cpu_mem_usage=True
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using fitz (PyMuPDF)"""
    print(f"Extracting text from PDF using fitz: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    text = ""
    
    try:
        doc = fitz.open(pdf_path)
        print(f"PDF opened successfully. Number of pages: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text + " "
            
            if page_num % 10 == 0:
                print(f"Processed page {page_num + 1}/{len(doc)}")
        
        doc.close()
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise
    
    # Clean up the extracted text
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    
    print(f"Text extraction completed.")
    print(f"Total characters: {len(text)}")
    print(f"First 200 characters: {text[:200]}...")
    
    return text

def split_text_into_words(text: str) -> List[str]:
    """Split text into words while preserving sentence structure"""
    words = text.split()
    
    # Filter out very short or problematic tokens
    filtered_words = []
    for word in words:
        if re.search(r'[a-zA-Z]', word) or word in ['.', ',', '!', '?', ';', ':']:
            filtered_words.append(word)
    
    print(f"Total words after filtering: {len(filtered_words)}")
    return filtered_words

def get_model_predictions_and_analysis(model, tokenizer, context: str, target_word: str, top_k: int = 2):
    """
    Get model predictions and analyze target word using context-aware tokenization
    """
    device = next(model.parameters()).device
    
    # Step 1: Tokenize just the context
    context_inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=20000,
        add_special_tokens=True
    )
    context_token_ids = context_inputs['input_ids'][0].tolist()
    context_length = len(context_token_ids)
    
    # Step 2: Tokenize context + target_word
    context_with_target = context + " " + target_word
    full_inputs = tokenizer(
        context_with_target,
        return_tensors="pt",
        truncation=True,
        max_length=20000,
        add_special_tokens=True
    )
    full_token_ids = full_inputs['input_ids'][0].tolist()
    full_length = len(full_token_ids)
    
    # Step 3: Extract the new tokens added by target_word
    if full_length > context_length:
        target_token_ids = full_token_ids[context_length:]
        true_target_token_id = target_token_ids[0]  # First token of the target word
        
        print(f"DEBUG: Context tokens: {context_length}")
        print(f"DEBUG: Full tokens: {full_length}")
        print(f"DEBUG: New tokens from target word: {target_token_ids}")
        print(f"DEBUG: Target word '{target_word}' -> Token ID: {true_target_token_id}")
        print(f"DEBUG: Decoded back: '{tokenizer.decode([true_target_token_id])}'")
    else:
        print(f"WARNING: Target word '{target_word}' didn't add any new tokens!")
        true_target_token_id = -1
    
    # Step 4: Run model inference on just the context
    context_inputs = {k: v.to(device) for k, v in context_inputs.items()}
    
    with torch.no_grad():
        # Single forward pass through the model
        outputs = model(**context_inputs, output_hidden_states=True)
        
        # Get logits for the next token (last position)
        next_token_logits = outputs.logits[0, -1, :].cpu().detach()
        
        # Get the last hidden state (this is our "last pseudo token")
        last_hidden_state = outputs.hidden_states[-1][0, -1, :].cpu().detach().float()
        
        # Convert to probabilities using double precision to avoid underflow
        next_token_logits_double = next_token_logits.double()
        next_token_probs_double = F.softmax(next_token_logits_double, dim=0)
        
        # Convert back to float for storage
        next_token_probs = next_token_probs_double.float()
        next_token_logits = next_token_logits.float()
        
        # Get top k predictions
        top_logits, top_indices = torch.topk(next_token_logits, top_k)
        top_probs = next_token_probs[top_indices]
        
        # Decode top tokens
        top_tokens = []
        for idx in top_indices:
            token = tokenizer.decode([idx], skip_special_tokens=True).strip()
            top_tokens.append(token)
        
        # Step 5: Look up the true target token ID in our predictions
        if true_target_token_id >= 0 and true_target_token_id < len(next_token_logits):
            target_logit = next_token_logits[true_target_token_id].item()
            target_prob = next_token_probs[true_target_token_id].item()
        else:
            target_logit, target_prob = float('-inf'), 0.0
        
        # Step 6: Check if target token ID matches any of the top predictions
        target_in_top_k = False
        matching_top_index = -1
        if true_target_token_id >= 0:
            for i, top_idx in enumerate(top_indices):
                if top_idx.item() == true_target_token_id:
                    target_in_top_k = True
                    matching_top_index = i
                    break
        
        # Debug output
        print(f'Predicted token IDs: {[idx.item() for idx in top_indices]}')
        print(f'True target token ID: {true_target_token_id}')
        print(f'Match with top-{matching_top_index + 1 if target_in_top_k else "None"}')
        
        return {
            'top_tokens': top_tokens,
            'top_indices': top_indices.numpy(),
            'top_logits': top_logits.numpy(),
            'top_probs': top_probs.numpy(),
            'target_word': target_word,
            'target_logit': target_logit,
            'target_prob': target_prob,
            'target_token_id': true_target_token_id,
            'target_in_top_k': target_in_top_k,
            'matching_top_index': matching_top_index,
            'last_hidden_state': last_hidden_state  # Added this for cosine similarity calculation
        }

def get_unembedding_vector(model, token_id: int) -> Optional[torch.Tensor]:
    """Get the unembedding vector (lm_head weight) for a token"""
    try:
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
            if model.lm_head.weight.is_meta:
                return None
            
            lm_head_weights = model.lm_head.weight.detach().cpu().float()
            
            if token_id < 0 or token_id >= lm_head_weights.size(0):
                return None
                
            return lm_head_weights[token_id]
        else:
            return None
            
    except Exception as e:
        print(f"Error accessing unembedding vector for token {token_id}: {e}")
        return None

def calculate_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> Optional[float]:
    """Calculate cosine similarity between two vectors"""
    try:
        if vec1 is None or vec2 is None:
            return None
        
        if vec1.size() != vec2.size():
            return None
            
        cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return cosine_sim
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return None

def analyze_pdf_document(pdf_path: str, model_path: str, output_csv: str = "pdf_analysis_results.csv"):
    """
    Main function to analyze PDF document word by word with context-aware tokenization
    """
    
    print("=" * 60)
    print("PDF WORD-BY-WORD ANALYSIS WITH CONTEXT-AWARE TOKENIZATION")
    print("=" * 60)
    
    # Step 1: Load the language model
    print("\n1. Loading language model...")
    model, tokenizer = load_model(model_path)
    
    # Step 2: Extract text from PDF
    print("\n2. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    # Step 3: Split into words
    print("\n3. Splitting text into words...")
    words = split_text_into_words(text)
    
    if len(words) < 2:
        print("Error: Not enough words in the document for analysis.")
        return None
    
    # Step 4: Prepare for analysis
    print(f"\n4. Starting word-by-word analysis...")
    print(f"   Total word positions to analyze: {len(words) - 1}")
    
    results = []
    
    # Check if we can access unembedding vectors
    test_unembedding = get_unembedding_vector(model, 0)
    can_compute_similarities = test_unembedding is not None
    
    if not can_compute_similarities:
        print("Warning: Cannot access unembedding vectors. Cosine similarities will be None.")
    
    # Step 5: Process each word position
    for i in range(len(words) - 1):
        current_position = i + 1
        current_context = " ".join(words[:i + 1])
        original_next_word = words[i + 1]
        
        print(f"\nPosition {current_position}/{len(words) - 1}")
        last_word = current_context.split()[-1] if current_context.split() else ""
        print(f"Context last word: '{last_word}' | Original next: '{original_next_word}'")
        
        try:
            # Get model predictions and analyze target word
            analysis = get_model_predictions_and_analysis(
                model, tokenizer, current_context, original_next_word, top_k=2
            )
            
            # Extract top predictions
            first_pred = analysis['top_tokens'][0] if len(analysis['top_tokens']) > 0 else ""
            second_pred = analysis['top_tokens'][1] if len(analysis['top_tokens']) > 1 else ""
            
            first_pred_token_id = analysis['top_indices'][0] if len(analysis['top_indices']) > 0 else -1
            second_pred_token_id = analysis['top_indices'][1] if len(analysis['top_indices']) > 1 else -1
            
            # Get metrics for predictions
            first_pred_logit = float(analysis['top_logits'][0]) if len(analysis['top_logits']) > 0 else 0.0
            first_pred_prob = float(analysis['top_probs'][0]) if len(analysis['top_probs']) > 0 else 0.0
            
            second_pred_logit = float(analysis['top_logits'][1]) if len(analysis['top_logits']) > 1 else 0.0
            second_pred_prob = float(analysis['top_probs'][1]) if len(analysis['top_probs']) > 1 else 0.0
            
            # Get metrics for original word (from context-aware tokenization)
            original_logit = analysis['target_logit']
            original_prob = analysis['target_prob']
            original_token_id = analysis['target_token_id']
            
            # Get the last hidden state (pseudo token)
            last_hidden_state = analysis['last_hidden_state']
            
            # Calculate cosine similarities with original word's unembedding vector
            first_pred_original_cosine_sim = None
            second_pred_original_cosine_sim = None
            
            # NEW: Calculate cosine similarities with last pseudo token
            last_pseudo_first_cosine_sim = None
            last_pseudo_second_cosine_sim = None
            last_pseudo_original_cosine_sim = None
            
            if can_compute_similarities and original_token_id >= 0:
                original_unembedding = get_unembedding_vector(model, original_token_id)
                
                if original_unembedding is not None:
                    # Calculate cosine similarities between predictions and original
                    if first_pred_token_id >= 0:
                        first_pred_unembedding = get_unembedding_vector(model, first_pred_token_id)
                        first_pred_original_cosine_sim = calculate_cosine_similarity(
                            first_pred_unembedding, original_unembedding
                        )
                        
                        # NEW: Calculate cosine similarity between last pseudo token and first prediction
                        last_pseudo_first_cosine_sim = calculate_cosine_similarity(
                            last_hidden_state, first_pred_unembedding
                        )
                    
                    if second_pred_token_id >= 0:
                        second_pred_unembedding = get_unembedding_vector(model, second_pred_token_id)
                        second_pred_original_cosine_sim = calculate_cosine_similarity(
                            second_pred_unembedding, original_unembedding
                        )
                        
                        # NEW: Calculate cosine similarity between last pseudo token and second prediction
                        last_pseudo_second_cosine_sim = calculate_cosine_similarity(
                            last_hidden_state, second_pred_unembedding
                        )
                    
                    # NEW: Calculate cosine similarity between last pseudo token and original
                    last_pseudo_original_cosine_sim = calculate_cosine_similarity(
                        last_hidden_state, original_unembedding
                    )
            
            # Check if prediction matches original (by token ID, not string)
            prediction_match = (first_pred_token_id == original_token_id) and (original_token_id >= 0)
            
            # Store results
            result_row = {
                'position': current_position,
                'context_last_word': last_word,
                'first_prediction': first_pred,
                'first_pred_logit': first_pred_logit,
                'first_pred_prob': first_pred_prob,
                'first_pred_original_cosine_sim': first_pred_original_cosine_sim,
                'second_prediction': second_pred,
                'second_pred_logit': second_pred_logit,
                'second_pred_prob': second_pred_prob,
                'second_pred_original_cosine_sim': second_pred_original_cosine_sim,
                'original_next_word': original_next_word,
                'original_logit': original_logit,
                'original_prob': original_prob,
                'prediction_match': prediction_match,
                # NEW: Added three new columns for last pseudo token cosine similarities
                'last_pseudo_first_cosine_sim': last_pseudo_first_cosine_sim,
                'last_pseudo_second_cosine_sim': last_pseudo_second_cosine_sim,
                'last_pseudo_original_cosine_sim': last_pseudo_original_cosine_sim
            }
            
            results.append(result_row)
            
            # Enhanced display
            print(f"Predicted: '{first_pred}' (ID: {first_pred_token_id}, logit: {first_pred_logit:.4f}, prob: {first_pred_prob:.6f})")
            print(f"Original: '{original_next_word}' (ID: {original_token_id}, logit: {original_logit:.4f}, prob: {original_prob:.6f})")
            print(f"Token ID Match: {prediction_match}")
            
            # Verify consistency for exact matches
            if prediction_match and analysis['target_in_top_k']:
                if abs(original_logit - first_pred_logit) < 0.001:
                    print("✓ Logits match perfectly - token IDs are identical")
                else:
                    print(f"ERROR: Same token ID but different logits!")
            
            if first_pred_original_cosine_sim is not None:
                print(f"First pred cosine sim with original: {first_pred_original_cosine_sim:.4f}")
            if second_pred_original_cosine_sim is not None:
                print(f"Second pred cosine sim with original: {second_pred_original_cosine_sim:.4f}")
            
            # NEW: Display last pseudo token cosine similarities
            if last_pseudo_first_cosine_sim is not None:
                print(f"Last pseudo token cosine sim with first pred: {last_pseudo_first_cosine_sim:.4f}")
            if last_pseudo_second_cosine_sim is not None:
                print(f"Last pseudo token cosine sim with second pred: {last_pseudo_second_cosine_sim:.4f}")
            if last_pseudo_original_cosine_sim is not None:
                print(f"Last pseudo token cosine sim with original: {last_pseudo_original_cosine_sim:.4f}")
            
            # Save checkpoint every 25 positions
            if current_position % 25 == 0:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_file = f"checkpoint_{output_csv}"
                checkpoint_df.to_csv(checkpoint_file, index=False)
                print(f"Checkpoint saved: {checkpoint_file}")
            
            # Save main file every 100 positions
            if current_position % 100 == 0:
                main_df = pd.DataFrame(results)
                main_df.to_csv(output_csv, index=False)
                print(f"Main file updated: {output_csv}")
            
        except Exception as e:
            print(f"Error at position {current_position}: {str(e)}")
            # Add error entry to results
            error_row = {
                'position': current_position,
                'context_last_word': last_word,
                'original_next_word': original_next_word,
                'first_prediction': '',
                'first_pred_logit': 0.0,
                'first_pred_prob': 0.0,
                'first_pred_original_cosine_sim': None,
                'second_prediction': '',
                'second_pred_logit': 0.0,
                'second_pred_prob': 0.0,
                'second_pred_original_cosine_sim': None,
                'original_logit': float('-inf'),
                'original_prob': 0.0,
                'prediction_match': False,
                'last_pseudo_first_cosine_sim': None,
                'last_pseudo_second_cosine_sim': None,
                'last_pseudo_original_cosine_sim': None,
                'error': str(e)
            }
            results.append(error_row)
    
    # Step 6: Save final results
    print(f"\n6. Saving final results...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    # Step 7: Print summary statistics
    print(f"\nANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Total positions analyzed: {len(results)}")
    
    if 'prediction_match' in df_results.columns:
        valid_predictions = df_results.dropna(subset=['prediction_match'])
        accuracy = valid_predictions['prediction_match'].mean() if len(valid_predictions) > 0 else 0.0
        print(f"Token ID match accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        # Additional statistics
        if 'original_prob' in df_results.columns:
            valid_probs = df_results[df_results['original_prob'] > 0]
            if len(valid_probs) > 0:
                avg_original_prob = valid_probs['original_prob'].mean()
                median_original_prob = valid_probs['original_prob'].median()
                print(f"Average probability of original words: {avg_original_prob:.6f}")
                print(f"Median probability of original words: {median_original_prob:.6f}")
                print(f"Min probability: {valid_probs['original_prob'].min():.2e}")
                print(f"Max probability: {valid_probs['original_prob'].max():.6f}")
        
        # Cosine similarity statistics
        if 'first_pred_original_cosine_sim' in df_results.columns:
            valid_first_cosines = df_results.dropna(subset=['first_pred_original_cosine_sim'])
            if len(valid_first_cosines) > 0:
                avg_first_cosine = valid_first_cosines['first_pred_original_cosine_sim'].mean()
                print(f"Average first prediction cosine similarity with original: {avg_first_cosine:.4f}")
        
        if 'second_pred_original_cosine_sim' in df_results.columns:
            valid_second_cosines = df_results.dropna(subset=['second_pred_original_cosine_sim'])
            if len(valid_second_cosines) > 0:
                avg_second_cosine = valid_second_cosines['second_pred_original_cosine_sim'].mean()
                print(f"Average second prediction cosine similarity with original: {avg_second_cosine:.4f}")
        
        # NEW: Last pseudo token cosine similarity statistics
        if 'last_pseudo_first_cosine_sim' in df_results.columns:
            valid_pseudo_first = df_results.dropna(subset=['last_pseudo_first_cosine_sim'])
            if len(valid_pseudo_first) > 0:
                avg_pseudo_first = valid_pseudo_first['last_pseudo_first_cosine_sim'].mean()
                print(f"Average last pseudo token cosine similarity with first pred: {avg_pseudo_first:.4f}")
        
        if 'last_pseudo_second_cosine_sim' in df_results.columns:
            valid_pseudo_second = df_results.dropna(subset=['last_pseudo_second_cosine_sim'])
            if len(valid_pseudo_second) > 0:
                avg_pseudo_second = valid_pseudo_second['last_pseudo_second_cosine_sim'].mean()
                print(f"Average last pseudo token cosine similarity with second pred: {avg_pseudo_second:.4f}")
        
        if 'last_pseudo_original_cosine_sim' in df_results.columns:
            valid_pseudo_original = df_results.dropna(subset=['last_pseudo_original_cosine_sim'])
            if len(valid_pseudo_original) > 0:
                avg_pseudo_original = valid_pseudo_original['last_pseudo_original_cosine_sim'].mean()
                print(f"Average last pseudo token cosine similarity with original: {avg_pseudo_original:.4f}")
    
    print(f"Results saved to: {output_csv}")
    print(f"Data shape: {df_results.shape}")
    
    return df_results

def display_sample_results(df: pd.DataFrame, n_samples: int = 5):
    """Display sample results for quick inspection"""
    print(f"\nSample Results (first {n_samples} rows):")
    print("-" * 160)
    
    display_columns = [
        'position', 'context_last_word', 'first_prediction', 'first_pred_prob',
        'first_pred_original_cosine_sim', 'second_pred_original_cosine_sim',
        'original_next_word', 'original_prob', 'prediction_match',
        'last_pseudo_first_cosine_sim', 'last_pseudo_second_cosine_sim', 'last_pseudo_original_cosine_sim'
    ]
    
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head(n_samples).to_string(index=False, float_format='%.6f'))

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "../data/attention_is_all_you_need.pdf"
    MODEL_PATH = "../../models/Llama-3.1-8B-Instruct"
    OUTPUT_CSV = "../results/pdf_word_analysis_old_paper.csv"
    
    print("PDF Word Analysis Script with Context-Aware Tokenization")
    print(f"PDF: {PDF_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_CSV}")
    
    try:
        # Run the analysis
        results_df = analyze_pdf_document(PDF_PATH, MODEL_PATH, OUTPUT_CSV)
        
        if results_df is not None:
            # Display sample results
            display_sample_results(results_df)
            
            print(f"\nAnalysis completed successfully!")
            print(f"Full results available in: {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your file paths and try again.")
