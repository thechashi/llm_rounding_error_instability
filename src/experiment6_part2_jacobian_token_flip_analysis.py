import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 

def load_model(model_path):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,  # Changed to float32
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "22GB", 1: "22GB"}
    )
    model.eval()
    return model, tokenizer

def model_forward_last_hidden(model, flattened_last_token_embedding, original_shape, full_embeddings, last_token_idx):
    """Forward pass for Jacobian computation"""
    last_token_embedding = flattened_last_token_embedding.view(original_shape)
    modified_embeddings = full_embeddings.clone()
    modified_embeddings[0, last_token_idx, :] = last_token_embedding
    outputs = model(inputs_embeds=modified_embeddings, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
    return last_hidden_state  # Already float32

def compute_jacobian(model, full_embeddings, last_token_idx):
    """Compute Jacobian"""
    device = next(model.parameters()).device
    last_token_embedding = full_embeddings[0, last_token_idx, :].clone().float()
    original_shape = last_token_embedding.shape
    flattened_embedding = last_token_embedding.flatten().detach().requires_grad_(True)
    
    print(f"Computing Jacobian (in float32)...")
    print(f"Flattened embedding dtype: {flattened_embedding.dtype}")
    
    forward_fn = partial(
        model_forward_last_hidden,
        model,
        original_shape=original_shape,
        full_embeddings=full_embeddings.detach(),
        last_token_idx=last_token_idx
    )
    
    jacobian = torch.autograd.functional.jacobian(forward_fn, flattened_embedding, vectorize=True)
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Jacobian dtype: {jacobian.dtype}")
    return jacobian

def perform_svd(jacobian):
    """Perform SVD"""
    print(f"Performing SVD on Jacobian with dtype: {jacobian.dtype}")
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    print(f"\nSVD Results:")
    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")
    print(f"  Top 10 singular values: {S[:10].cpu().numpy()}")
    return U, S, Vt

def test_token_flip_along_direction(model, tokenizer, full_embeddings, last_token_idx, direction, epsilon_range):
    """
    Test token prediction flip along a direction in float32
    epsilon_range: list of epsilon values to test
    """
    device = next(model.parameters()).device
    
    # Ensure direction matches embeddings dtype (float32)
    direction = direction.to(full_embeddings.dtype).to(device)
    
    results = []
    
    # Get original prediction
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=full_embeddings, output_hidden_states=True)
        logits_orig = outputs_orig.logits[0, last_token_idx, :]
        orig_token_id = torch.argmax(logits_orig).item()
        orig_token = tokenizer.decode([orig_token_id])
    
    print(f"Original prediction: '{orig_token}' (ID: {orig_token_id})")
    
    for epsilon in tqdm(epsilon_range):
        # Perturb embedding in float32
        perturbed_embeddings = full_embeddings.clone()
        perturbed_embeddings[0, last_token_idx, :] += epsilon * direction
        
        # Get prediction
        with torch.no_grad():
            outputs = model(inputs_embeds=perturbed_embeddings, output_hidden_states=True)
            logits = outputs.logits[0, last_token_idx, :]
            pred_token_id = torch.argmax(logits).item()
            pred_token = tokenizer.decode([pred_token_id])
            
            probs = F.softmax(logits, dim=-1)
            pred_prob = probs[pred_token_id].item()
        
        token_changed = (pred_token_id != orig_token_id)
        
        results.append({
            'epsilon': epsilon,
            'pred_token_id': pred_token_id,
            'pred_token': pred_token,
            'pred_prob': pred_prob,
            'token_changed': token_changed
        })
        
        if token_changed:
            print(f"  ε={epsilon:.2e}: FLIP! '{orig_token}' → '{pred_token}' (p={pred_prob:.4f})")
    
    return results

if __name__ == "__main__":
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    device = next(model.parameters()).device
    
    # Load optimized state
    print("Loading optimized state...")
    state = torch.load("optimized_state_geometric.pt")
    full_embeddings = state['full_embeddings'].to(device).float()  # Ensure float32
    last_token_idx = state['last_token_idx']
    input_text = state['input_text']
    
    print(f"Input text: '{input_text}'")
    print(f"Last token index: {last_token_idx}")
    print(f"Embeddings dtype: {full_embeddings.dtype}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Compute Jacobian and SVD
    print("\n" + "="*80)
    print("COMPUTING JACOBIAN AND SVD")
    print("="*80)
    jacobian = compute_jacobian(model, full_embeddings, last_token_idx)
    U, S, Vt = perform_svd(jacobian)
    
    # Generate epsilon range appropriate for float32 precision
    epsilon_min = -1e-6
    epsilon_max = 1e-6
    epsilon_step = 1e-9
    epsilon_range = np.arange(epsilon_min, epsilon_max + epsilon_step, epsilon_step)
    
    print(f"\nEpsilon range: [{epsilon_min}, {epsilon_max}] with step {epsilon_step}")
    print(f"Total steps: {len(epsilon_range)}")
    
    # Test along top 5 singular directions
    print("\n" + "="*80)
    print("TESTING TOKEN FLIPS ALONG TOP 5 SINGULAR DIRECTIONS")
    print("="*80)
    
    all_results = {}
    
    for k in range(1, 6):
        print(f"\n--- Singular Direction {k} (Singular Value: {S[k-1].item():.6f}) ---")
        
        direction = Vt[k-1, :].to(device)
        
        results = test_token_flip_along_direction(
            model, tokenizer, full_embeddings, last_token_idx, 
            direction, epsilon_range
        )
        
        all_results[f'direction_{k}'] = results
        
        # Save results for this direction
        df = pd.DataFrame(results)
        df.to_csv(f"token_flip_direction_{k}_geom_float32.csv", index=False)
        print(f"Saved results to token_flip_direction_{k}_geom_float32.csv")
        
        # Summary
        flips = df[df['token_changed'] == True]
        if len(flips) > 0:
            first_flip = flips.iloc[0]
            print(f"First flip at ε={first_flip['epsilon']:.2e}")
        else:
            print("No token flip observed in this range")
    
    print("\nDone!")