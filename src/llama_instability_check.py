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
    
    # Force everything to CPU for consistency with original code
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

# -----------------------------
# Get embeddings + logits (LLaMA version)
# -----------------------------
def get_embeddings_and_logits(model, tokenizer, input_text):
    """Get embeddings and logits from LLaMA model"""
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Token embeddings - LLaMA uses model.model.embed_tokens
        embeddings = model.model.embed_tokens(inputs["input_ids"])
        
        # Forward pass through the model
        outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits[0, -1, :]  # last-token logits
    
    return embeddings, logits, inputs["input_ids"]

# -----------------------------
# Apply perturbation to ALL tokens (same as original)
# -----------------------------
def perturb_embeddings_all(embeddings, perturbation_vector):
    """Apply perturbation to all tokens in the sequence"""
    perturbed = embeddings.clone()
    perturbed[0] += perturbation_vector  # add to all tokens in sequence
    diff = perturbed - embeddings
    diff_norm = torch.norm(diff).item()
    return perturbed, diff_norm, diff

# -----------------------------
# Main test function (adapted for LLaMA)
# -----------------------------
def test_perturbation_effects(model, tokenizer, input_text, perturbation_powers=None):
    """Test perturbation effects on LLaMA model"""
    if perturbation_powers is None:
        perturbation_powers = list(range(-10, 6))  # 2^-10 to 2^5
    
    print(f"\nTesting input: '{input_text}'")
    device = next(model.parameters()).device
    
    # Original embeddings + logits
    original_embeddings, original_logits, input_ids = get_embeddings_and_logits(model, tokenizer, input_text)
    original_token_id = torch.argmax(original_logits).item()
    original_token = tokenizer.decode([original_token_id])
    original_prob = F.softmax(original_logits, dim=-1)[original_token_id].item()
    print(f"Original prediction: '{original_token}' (prob={original_prob:.4f})")
    
    embed_dim = original_embeddings.shape[-1]
    results = []
    
    for power in perturbation_powers:
        magnitude = 2.0 ** power
        
        # Same perturbation types as original
        perturbation_types = [
            ("random_uniform", torch.rand(embed_dim, device=device) - 0.5),
            ("random_normal", torch.randn(embed_dim, device=device)),
            ("constant_positive", torch.ones(embed_dim, device=device)),
            ("single_dimension", torch.zeros(embed_dim, device=device))
        ]
        perturbation_types[3][1][0] = 1.0  # single dim perturbation
        
        for pert_type, base_vec in perturbation_types:
            vec = base_vec * magnitude
            
            # Apply perturbation to ALL token embeddings
            perturbed_embeddings, diff_norm, diff = perturb_embeddings_all(original_embeddings, vec)
            
            with torch.no_grad():
                # Forward pass with perturbed embeddings
                attention_mask = None
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                pert_outputs = model(
                    inputs_embeds=perturbed_embeddings,
                    attention_mask=attention_mask
                )
                pert_logits = pert_outputs.logits[0, -1, :]
            
            new_token_id = torch.argmax(pert_logits).item()
            new_token = tokenizer.decode([new_token_id])
            new_prob = F.softmax(pert_logits, dim=-1)[new_token_id].item()
            
            token_changed = (new_token_id != original_token_id)
            logit_diff = torch.norm(pert_logits - original_logits).item()
            
            results.append({
                "power": power,
                "magnitude": magnitude,
                "perturbation_type": pert_type,
                "original_token": str(original_token),  # Convert to string
                "new_token": str(new_token),  # Convert to string
                "token_changed": bool(token_changed),  # Ensure boolean
                "original_prob": float(original_prob),  # Ensure float
                "new_prob": float(new_prob),  # Ensure float
                "logit_diff": float(logit_diff),  # Ensure float
                "embedding_diff": float(diff_norm),  # Ensure float
            })
            
            if token_changed:
                print(f"  2^{power} {pert_type}: '{original_token}' → '{new_token}' (prob={new_prob:.4f})")
    
    return pd.DataFrame(results)

# -----------------------------
# Test multiple inputs
# -----------------------------
def run_comprehensive_test():
    """Run the same tests as the original GPT-OSS code"""
    print("Loading LLaMA 3.1 model...")
    model, tokenizer = load_llama_model()
    
    # Test cases (similar to what would be tested)
    test_cases = [
        "The capital of France is",
        "To solve this math problem, we need to",
        "In conclusion, the main finding was",
        "The quick brown fox jumps over",
        "Machine learning is a subset of"
    ]
    
    all_results = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}: '{test_case}'")
        print('='*60)
        
        results_df = test_perturbation_effects(model, tokenizer, test_case)
        all_results[f"test_case_{i+1}"] = results_df
        
        # Summary statistics
        total_changes = results_df['token_changed'].sum()
        total_tests = len(results_df)
        change_rate = total_changes / total_tests * 100
        
        print(f"\nSummary for '{test_case}':")
        print(f"  Total perturbations tested: {total_tests}")
        print(f"  Perturbations causing token change: {total_changes}")
        print(f"  Change rate: {change_rate:.2f}%")
        
        # Show most sensitive perturbations
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
    """Analyze which perturbation types are most effective"""
    all_data = []
    
    for test_name, df in results_dict.items():
        df_copy = df.copy()
        df_copy['test_case'] = test_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Analyze by perturbation type
    print("\nPerturbation Type Analysis:")
    print("="*50)
    
    type_analysis = combined_df.groupby('perturbation_type').agg({
        'token_changed': ['count', 'sum', 'mean'],
        'logit_diff': 'mean',
        'embedding_diff': 'mean'
    }).round(4)
    
    print(type_analysis)
    
    # Analyze by magnitude
    print("\nMagnitude Analysis:")
    print("="*50)
    
    magnitude_analysis = combined_df.groupby('power').agg({
        'token_changed': ['count', 'sum', 'mean'],
        'logit_diff': 'mean'
    }).round(4)
    
    print(magnitude_analysis)
    
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
        # Create a copy and convert problematic columns to strings
        df_to_save = df.copy()
        
        # Convert any tensor columns or objects that might cause issues
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                df_to_save[col] = df_to_save[col].astype(str)
        
        try:
            df_to_save.to_csv(filename, index=False, escapechar='\\')
            print(f"  Saved {filename}")
        except Exception as e:
            print(f"  Error saving {filename}: {e}")
            # Save as pickle instead
            pickle_filename = filename.replace('.csv', '.pkl')
            df.to_pickle(pickle_filename)
            print(f"  Saved as pickle: {pickle_filename}")
    
    # Save combined analysis
    try:
        # Convert problematic columns
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