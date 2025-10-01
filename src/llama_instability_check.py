import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load LLaMA model and tokenizer
# -----------------------------
def load_llama_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Load LLaMA model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

# -----------------------------
# Get embeddings + logits + ALL hidden states (before and after norm)
# -----------------------------
def get_embeddings_logits_and_all_hidden(model, tokenizer, input_text):
    """Get embeddings, logits, and ALL token hidden states (before and after RMSNorm)"""
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Token embeddings
        embeddings = model.model.embed_tokens(inputs["input_ids"])
        
        # Forward pass through the model with hidden states
        outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask"), output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :]  # last-token logits
        
        # Hidden states from last layer BEFORE the final norm - ALL tokens
        hidden_before_norm_all = outputs.hidden_states[-1][0, :, :]  # [seq_len, hidden_dim]
        
        # Apply the final RMSNorm to get the normalized hidden states - ALL tokens
        hidden_after_norm_all = model.model.norm(outputs.hidden_states[-1])[0, :, :]  # [seq_len, hidden_dim]
        
        # Also get just the last token for backward compatibility
        hidden_before_norm_last = outputs.hidden_states[-1][0, -1, :]
        hidden_after_norm_last = model.model.norm(outputs.hidden_states[-1])[0, -1, :]
    
    return embeddings, logits, hidden_before_norm_all, hidden_after_norm_all, hidden_before_norm_last, hidden_after_norm_last, inputs["input_ids"]

# -----------------------------
# Apply perturbation to ALL tokens
# -----------------------------
def perturb_embeddings_all(embeddings, perturbation_vector):
    """Apply perturbation to all tokens in the sequence"""
    perturbed = embeddings.clone()
    # Add perturbation to all tokens: embeddings shape is [batch, seq_len, hidden_dim]
    # perturbation_vector shape is [hidden_dim]
    perturbed[0, :, :] += perturbation_vector  # Add to all tokens in the sequence
    perturbed[0, :, :] -= perturbation_vector  # Subtract to introduce rounding errors
    diff = perturbed - embeddings
    print('Perturbation vetor: ', perturbation_vector)
    print('Original embedding: ', embeddings)
    print('Perturb embedding: ', perturbed)
    print('Diff: ', diff)
    diff_norm = torch.norm(diff).item()
    
    # Compute per-token statistics
    # diff shape: [1, seq_len, hidden_dim]
    per_token_norms = torch.norm(diff[0], dim=1)  # [seq_len] - L2 norm for each token
    
    # Compute cosine similarity for each token between original and perturbed
    orig_tokens = embeddings[0]  # [seq_len, hidden_dim]
    pert_tokens = perturbed[0]   # [seq_len, hidden_dim]
    per_token_cosine = F.cosine_similarity(orig_tokens, pert_tokens, dim=1)  # [seq_len]
    
    # Statistics
    stats = {
        'diff_norm_total': diff_norm,
        'diff_mean': torch.mean(diff).item(),
        'diff_std': torch.std(diff).item(),
        'per_token_norm_mean': torch.mean(per_token_norms).item(),
        'per_token_norm_std': torch.std(per_token_norms).item(),
        'per_token_norm_min': torch.min(per_token_norms).item(),
        'per_token_norm_max': torch.max(per_token_norms).item(),
        'per_token_cosine_mean': torch.mean(per_token_cosine).item(),
        'per_token_cosine_std': torch.std(per_token_cosine).item(),
        'per_token_cosine_min': torch.min(per_token_cosine).item(),
        'per_token_cosine_max': torch.max(per_token_cosine).item(),
    }
    
    return perturbed, diff_norm, diff, stats

# -----------------------------
# Compute per-token EMBEDDING changes (rounding errors)
# -----------------------------
def compute_per_token_embedding_changes(orig_embeddings, pert_embeddings, tokenizer, input_ids):
    """
    Compute L2 distance and cosine similarity for each token's EMBEDDING after add+subtract
    This shows the rounding errors introduced by floating point operations
    Returns: list of dicts with per-token embedding metrics
    """
    seq_len = orig_embeddings.shape[1]
    per_token_metrics = []
    
    tokens_text = [tokenizer.decode([token_id.item()]) for token_id in input_ids[0]]
    
    for i in range(seq_len):
        orig_vec = orig_embeddings[0, i, :]
        pert_vec = pert_embeddings[0, i, :]
        
        l2_dist = torch.norm(orig_vec - pert_vec).item()
        cos_sim = F.cosine_similarity(orig_vec.unsqueeze(0), pert_vec.unsqueeze(0)).item()
        
        orig_norm = torch.norm(orig_vec).item()
        pert_norm = torch.norm(pert_vec).item()
        
        per_token_metrics.append({
            'token_idx': i,
            'token_text': tokens_text[i],
            'l2_distance': l2_dist,
            'cosine_similarity': cos_sim,
            'orig_norm': orig_norm,
            'pert_norm': pert_norm
        })
    
    return per_token_metrics

# -----------------------------
# Compute per-token hidden state changes
# -----------------------------
def compute_per_token_hidden_changes(orig_hidden_all, pert_hidden_all, tokenizer, input_ids):
    """
    Compute L2 distance and cosine similarity for each token's hidden state
    Returns: list of dicts with per-token metrics
    """
    seq_len = orig_hidden_all.shape[0]
    per_token_metrics = []
    
    tokens_text = [tokenizer.decode([token_id.item()]) for token_id in input_ids[0]]
    
    for i in range(seq_len):
        orig_vec = orig_hidden_all[i]
        pert_vec = pert_hidden_all[i]
        
        l2_dist = torch.norm(orig_vec - pert_vec).item()
        cos_sim = F.cosine_similarity(orig_vec.unsqueeze(0), pert_vec.unsqueeze(0)).item()
        
        orig_norm = torch.norm(orig_vec).item()
        pert_norm = torch.norm(pert_vec).item()
        
        per_token_metrics.append({
            'token_idx': i,
            'token_text': tokens_text[i],
            'l2_distance': l2_dist,
            'cosine_similarity': cos_sim,
            'orig_norm': orig_norm,
            'pert_norm': pert_norm
        })
    
    return per_token_metrics

# -----------------------------
# Compute L2 distance and cosine similarity
# -----------------------------
def compute_similarity_metrics(vec1, vec2):
    """Compute L2 distance and cosine similarity between two vectors"""
    l2_distance = torch.norm(vec1 - vec2).item()
    cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    return l2_distance, cosine_sim

# -----------------------------
# Main test function
# -----------------------------
def test_perturbation_effects(model, tokenizer, input_text, perturbation_powers=None):
    """Test perturbation effects on LLaMA model"""
    if perturbation_powers is None:
        perturbation_powers = list(range(-10, 6))  # 2^-10 to 2^5
    
    print(f"\nTesting input: '{input_text}'")
    device = next(model.parameters()).device
    
    # Get original states
    (original_embeddings, original_logits, 
     orig_hidden_before_all, orig_hidden_after_all,
     orig_hidden_before_last, orig_hidden_after_last, 
     input_ids) = get_embeddings_logits_and_all_hidden(model, tokenizer, input_text)
    
    original_token_id = torch.argmax(original_logits).item()
    original_token = tokenizer.decode([original_token_id])
    original_prob = F.softmax(original_logits, dim=-1)[original_token_id].item()
    
    # Print norms for last token
    norm_before = torch.norm(orig_hidden_before_last).item()
    norm_after = torch.norm(orig_hidden_after_last).item()
    print(f"Original prediction: '{original_token}' (prob={original_prob:.4f})")
    print(f"Last token hidden state norm BEFORE RMSNorm: {norm_before:.4f}")
    print(f"Last token hidden state norm AFTER RMSNorm: {norm_after:.4f}")
    
    embed_dim = original_embeddings.shape[-1]
    results = []
    
    for power in perturbation_powers:
        magnitude = 2.0 ** power
        
        perturbation_types = [
            # ("random_uniform", torch.rand(embed_dim, device=device) - 0.5),
            # ("random_normal", torch.randn(embed_dim, device=device)),
            ("constant_positive", torch.ones(embed_dim, device=device)),
            # ("single_dimension", torch.zeros(embed_dim, device=device))
        ]
        # perturbation_types[3][1][0] = 1.0
        
        for pert_type, base_vec in perturbation_types:
            vec = base_vec * magnitude
            
            perturbed_embeddings, diff_norm, diff, diff_stats = perturb_embeddings_all(original_embeddings, vec)
            
            with torch.no_grad():
                attention_mask = None
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                pert_outputs = model(
                    inputs_embeds=perturbed_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                pert_logits = pert_outputs.logits[0, -1, :]
                
                # Get perturbed hidden states (before and after norm) - ALL tokens
                pert_hidden_before_all = pert_outputs.hidden_states[-1][0, :, :]
                pert_hidden_after_all = model.model.norm(pert_outputs.hidden_states[-1])[0, :, :]
                
                # Last token only
                pert_hidden_before_last = pert_outputs.hidden_states[-1][0, -1, :]
                pert_hidden_after_last = model.model.norm(pert_outputs.hidden_states[-1])[0, -1, :]
            
            new_token_id = torch.argmax(pert_logits).item()
            new_token = tokenizer.decode([new_token_id])
            new_prob = F.softmax(pert_logits, dim=-1)[new_token_id].item()
            
            token_changed = (new_token_id != original_token_id)
            logit_diff = torch.norm(pert_logits - original_logits).item()
            
            # Compute metrics for last token BEFORE norm
            l2_dist_before, cos_sim_before = compute_similarity_metrics(orig_hidden_before_last, pert_hidden_before_last)
            
            # Compute metrics for last token AFTER norm
            l2_dist_after, cos_sim_after = compute_similarity_metrics(orig_hidden_after_last, pert_hidden_after_last)
            
            # Get norms for perturbed last token
            pert_norm_before = torch.norm(pert_hidden_before_last).item()
            pert_norm_after = torch.norm(pert_hidden_after_last).item()
            
            # Compute per-token EMBEDDING changes (rounding errors from add+subtract)
            per_token_embedding_changes = compute_per_token_embedding_changes(
                original_embeddings, perturbed_embeddings, tokenizer, input_ids
            )
            
            # Compute per-token hidden state changes (BEFORE and AFTER norm)
            per_token_before_norm = compute_per_token_hidden_changes(
                orig_hidden_before_all, pert_hidden_before_all, tokenizer, input_ids
            )
            per_token_after_norm = compute_per_token_hidden_changes(
                orig_hidden_after_all, pert_hidden_after_all, tokenizer, input_ids
            )
            
            results.append({
                "power": power,
                "magnitude": magnitude,
                "perturbation_type": pert_type,
                "original_token": str(original_token),
                "new_token": str(new_token),
                "token_changed": bool(token_changed),
                "original_prob": float(original_prob),
                "new_prob": float(new_prob),
                "logit_diff": float(logit_diff),
                "embedding_diff": float(diff_norm),
                # Diff statistics
                "diff_mean": float(diff_stats['diff_mean']),
                "diff_std": float(diff_stats['diff_std']),
                "per_token_norm_mean": float(diff_stats['per_token_norm_mean']),
                "per_token_norm_std": float(diff_stats['per_token_norm_std']),
                "per_token_norm_min": float(diff_stats['per_token_norm_min']),
                "per_token_norm_max": float(diff_stats['per_token_norm_max']),
                "per_token_cosine_mean": float(diff_stats['per_token_cosine_mean']),
                "per_token_cosine_std": float(diff_stats['per_token_cosine_std']),
                "per_token_cosine_min": float(diff_stats['per_token_cosine_min']),
                "per_token_cosine_max": float(diff_stats['per_token_cosine_max']),
                # Last token - Before norm metrics
                "hidden_l2_before_norm": float(l2_dist_before),
                "hidden_cosine_before_norm": float(cos_sim_before),
                "hidden_norm_before_orig": float(norm_before),
                "hidden_norm_before_pert": float(pert_norm_before),
                # Last token - After norm metrics
                "hidden_l2_after_norm": float(l2_dist_after),
                "hidden_cosine_after_norm": float(cos_sim_after),
                "hidden_norm_after_orig": float(norm_after),
                "hidden_norm_after_pert": float(pert_norm_after),
            })
            
        # if token_changed:
            print(f"\n  2^{power} {pert_type}: '{original_token}' → '{new_token}' (prob={new_prob:.4f})")
            print(f"    LAST TOKEN - BEFORE norm: L2={l2_dist_before:.4f}, cosine={cos_sim_before:.4f}, norm: {norm_before:.4f}→{pert_norm_before:.4f}")
            print(f"    LAST TOKEN - AFTER norm:  L2={l2_dist_after:.4f}, cosine={cos_sim_after:.4f}, norm: {norm_after:.4f}→{pert_norm_after:.4f}")
            
            print(f"\n    EMBEDDING DIFF STATS (Overall):")
            print(f"      Total diff norm: {diff_stats['diff_norm_total']:.4f}, Mean: {diff_stats['diff_mean']:.6f}, Std: {diff_stats['diff_std']:.4f}")
            print(f"      Per-token norm: mean={diff_stats['per_token_norm_mean']:.4f}, std={diff_stats['per_token_norm_std']:.4f}")
            print(f"      Per-token cosine: mean={diff_stats['per_token_cosine_mean']:.4f}, std={diff_stats['per_token_cosine_std']:.4f}")
            
            print(f"\n    PER-TOKEN EMBEDDING CHANGES (Rounding Errors from Add+Subtract):")
            for tok_metrics in per_token_embedding_changes:
                print(f"      Token {tok_metrics['token_idx']} '{tok_metrics['token_text']}': "
                        f"L2={tok_metrics['l2_distance']:.6f}, cosine={tok_metrics['cosine_similarity']:.6f}, "
                        f"norm: {tok_metrics['orig_norm']:.4f}→{tok_metrics['pert_norm']:.4f}")
            
            print(f"\n    PER-TOKEN HIDDEN STATE CHANGES (BEFORE RMSNorm):")
            for tok_metrics in per_token_before_norm:
                print(f"      Token {tok_metrics['token_idx']} '{tok_metrics['token_text']}': "
                        f"L2={tok_metrics['l2_distance']:.4f}, cosine={tok_metrics['cosine_similarity']:.4f}, "
                        f"norm: {tok_metrics['orig_norm']:.2f}→{tok_metrics['pert_norm']:.2f}")
            
            print(f"\n    PER-TOKEN HIDDEN STATE CHANGES (AFTER RMSNorm):")
            for tok_metrics in per_token_after_norm:
                print(f"      Token {tok_metrics['token_idx']} '{tok_metrics['token_text']}': "
                        f"L2={tok_metrics['l2_distance']:.4f}, cosine={tok_metrics['cosine_similarity']:.4f}, "
                        f"norm: {tok_metrics['orig_norm']:.2f}→{tok_metrics['pert_norm']:.2f}")
    
    return pd.DataFrame(results)

# -----------------------------
# Test multiple inputs
# -----------------------------
def run_comprehensive_test():
    """Run comprehensive tests"""
    print("Loading LLaMA 3.1 model...")
    model, tokenizer = load_llama_model()
    
    test_cases = [
        "The capital of France is",
        # "To solve this math problem, we need to",
        # "In conclusion, the main finding was",
        # "The quick brown fox jumps over",
        # "Machine learning is a subset of"
    ]
    
    all_results = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Test case {i+1}: '{test_case}'")
        print('='*80)
        
        results_df = test_perturbation_effects(model, tokenizer, test_case)
        all_results[f"test_case_{i+1}"] = results_df
        
        # Summary statistics
        total_changes = results_df['token_changed'].sum()
        total_tests = len(results_df)
        change_rate = total_changes / total_tests * 100
        
        print(f"\n{'='*80}")
        print(f"Summary for '{test_case}':")
        print(f"  Total perturbations tested: {total_tests}")
        print(f"  Perturbations causing token change: {total_changes}")
        print(f"  Change rate: {change_rate:.2f}%")
        
        changed_results = results_df[results_df['token_changed'] == True]
        if len(changed_results) > 0:
            min_magnitude = changed_results['magnitude'].min()
            most_sensitive = changed_results[changed_results['magnitude'] == min_magnitude].iloc[0]
            print(f"  Most sensitive perturbation: {most_sensitive['perturbation_type']} at magnitude {min_magnitude}")
    
    return all_results

# -----------------------------
# Analysis functions
# -----------------------------
def analyze_perturbation_sensitivity(results_dict):
    """Analyze perturbation sensitivity"""
    all_data = []
    
    for test_name, df in results_dict.items():
        df_copy = df.copy()
        df_copy['test_case'] = test_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("\nPerturbation Type Analysis:")
    print("="*50)
    
    type_analysis = combined_df.groupby('perturbation_type').agg({
        'token_changed': ['count', 'sum', 'mean'],
        'logit_diff': 'mean',
        'embedding_diff': 'mean',
        'hidden_l2_before_norm': 'mean',
        'hidden_cosine_before_norm': 'mean',
        'hidden_l2_after_norm': 'mean',
        'hidden_cosine_after_norm': 'mean'
    }).round(4)
    
    print(type_analysis)
    
    print("\nMagnitude Analysis:")
    print("="*50)
    
    magnitude_analysis = combined_df.groupby('power').agg({
        'token_changed': ['count', 'sum', 'mean'],
        'logit_diff': 'mean',
        'hidden_l2_before_norm': 'mean',
        'hidden_l2_after_norm': 'mean'
    }).round(4)
    
    print(magnitude_analysis)
    
    # Analyze changed tokens only
    print("\nMetrics for Changed Tokens:")
    print("="*50)
    changed_only = combined_df[combined_df['token_changed'] == True]
    if len(changed_only) > 0:
        print("BEFORE RMSNorm:")
        print(f"  Average L2 distance: {changed_only['hidden_l2_before_norm'].mean():.4f}")
        print(f"  Average Cosine similarity: {changed_only['hidden_cosine_before_norm'].mean():.4f}")
        print("\nAFTER RMSNorm:")
        print(f"  Average L2 distance: {changed_only['hidden_l2_after_norm'].mean():.4f}")
        print(f"  Average Cosine similarity: {changed_only['hidden_cosine_after_norm'].mean():.4f}")
    
    return combined_df

if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Analyze results
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    combined_analysis = analyze_perturbation_sensitivity(results)
    
    # Save results
    print("\nSaving results to CSV files...")
    for test_name, df in results.items():
        filename = f"llama_perturbation_{test_name}.csv"
        df_to_save = df.copy()
        
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                df_to_save[col] = df_to_save[col].astype(str)
        
        try:
            df_to_save.to_csv(filename, index=False, escapechar='\\')
            print(f"  Saved {filename}")
        except Exception as e:
            print(f"  Error saving {filename}: {e}")
            pickle_filename = filename.replace('.csv', '.pkl')
            df.to_pickle(pickle_filename)
            print(f"  Saved as pickle: {pickle_filename}")
    
    # Save combined analysis
    try:
        combined_to_save = combined_analysis.copy()
        for col in combined_to_save.columns:
            if combined_to_save[col].dtype == 'object':
                combined_to_save[col] = combined_to_save[col].astype(str)
        
        combined_to_save.to_csv("llama_perturbation_combined_analysis.csv", index=False, escapechar='\\')
        print("  Saved llama_perturbation_combined_analysis.csv")
    except Exception as e:
        print(f"  Error saving combined analysis: {e}")
        combined_analysis.to_pickle("llama_perturbation_combined_analysis.pkl")
        print("  Saved as pickle: llama_perturbation_combined_analysis.pkl")
    
    print("\nDone!")