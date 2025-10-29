"""
Lipschitz Constant Computation for Llama Models

This script rigorously computes the Lipschitz constant of the Llama model's
transformation from input embeddings to final hidden states using Jacobian
analysis and Singular Value Decomposition (SVD).

Purpose:
--------
Quantifies the theoretical upper bound on how much output can change relative to
input perturbations by:
1. Computing the Jacobian matrix (∂h/∂x) where h = hidden state, x = embedding
2. Performing SVD to find the largest singular value
3. The largest singular value = Lipschitz constant = maximum amplification factor


Relationship:
-------------
This file provides the FOUNDATIONAL METHODOLOGY for Lipschitz analysis used
throughout the project:
1. Jacobian computation via torch.autograd.functional.jacobian
2. SVD-based singular value extraction
3. Lipschitz constant = largest singular value

archive_lipschitz_constant_computation.py and
llama_model_lipschitz_computation_part2.py appear nearly identical, suggesting
part2 may be a backup or variant. The key techniques developed here are
foundational and reused in:
- experiment2 series: Average Lipschitz analysis
- experiment5: Layer-wise Lipschitz analysis
- experiment6-7: Jacobian-based sensitivity analysis

Theoretical Background:
-----------------------
The Lipschitz constant L quantifies the maximum rate of change:
  ||f(x + Δx) - f(x)|| ≤ L * ||Δx||

For neural networks, L = largest singular value of the Jacobian matrix.
This provides a rigorous upper bound on how much small input perturbations
can affect outputs.

Test Methodology:
-----------------
1. Load Llama model in float32 (for numerical precision)
2. Extract input embeddings for a test prompt
3. Compute Jacobian: ∂(last hidden state)/∂(last token embedding)
4. Perform SVD on Jacobian: J = U Σ Vᵀ
5. Extract largest singular value σ_max from Σ
6. σ_max = Lipschitz constant

Use Case:
---------
Use this script to:
- Compute theoretical upper bound on output sensitivity
- Understand how perturbations propagate through the model
- Validate that empirical instability (from instability_check.py) is bounded
  by the Lipschitz constant
- Establish baseline for more sophisticated experiments

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, pandas
- Llama-3.1-8B-Instruct model
- Uses float32 for numerical precision in Jacobian computation

Key Functions:
--------------
- load_llama_model(): Load model in float32 precision
- model_forward_last_hidden(): Forward pass for Jacobian computation
- compute_jacobian(): Compute Jacobian matrix using torch.autograd
- perform_svd(): SVD analysis to extract singular values
- test_lipschitz_constant(): Full pipeline for Lipschitz constant computation

Output:
-------
- Jacobian matrix shape and properties
- Singular values (particularly σ_max)
- Lipschitz constant = σ_max
- Comparison to empirical perturbation effects
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
        torch_dtype=torch.float32,  # Use float32 for better numerical precision in Jacobian
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

# -----------------------------
# Forward function for Jacobian computation
# -----------------------------
def model_forward_last_hidden(model, flattened_last_token_embedding, original_shape, full_embeddings, last_token_idx):
    """
    Forward pass that takes ONLY the last token embedding as input for Jacobian.
    Returns the last token's final hidden state (before norm).
    
    Args:
        flattened_last_token_embedding: Flattened last token embedding [embed_dim]
        original_shape: Original shape of last token embedding
        full_embeddings: Full sequence embeddings (kept fixed except last token)
        last_token_idx: Index of the last token
    """
    # Restore the last token embedding shape
    last_token_embedding = flattened_last_token_embedding.view(original_shape)
    
    # Replace the last token in full embeddings
    modified_embeddings = full_embeddings.clone()
    modified_embeddings[0, last_token_idx, :] = last_token_embedding
    
    # Forward pass
    outputs = model(inputs_embeds=modified_embeddings, output_hidden_states=True)
    
    # Get last token's hidden state from last layer (before RMSNorm)
    last_hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
    
    return last_hidden_state

# -----------------------------
# Compute Jacobian
# -----------------------------
def compute_jacobian(model, full_embeddings, last_token_idx):
    """
    Compute Jacobian of last token's final hidden state w.r.t last token's input embedding.
    
    Returns:
        jacobian: Shape [hidden_dim_output, hidden_dim_input]
    """
    device = next(model.parameters()).device
    
    # Extract last token embedding
    last_token_embedding = full_embeddings[0, last_token_idx, :].clone()
    original_shape = last_token_embedding.shape
    
    # Flatten for Jacobian computation
    flattened_embedding = last_token_embedding.flatten()
    flattened_embedding = flattened_embedding.detach().requires_grad_(True)
    
    print(f"Computing Jacobian for embedding shape: {original_shape}")
    print(f"Flattened embedding shape: {flattened_embedding.shape}")
    
    # Create partial function
    from functools import partial
    forward_fn = partial(
        model_forward_last_hidden,
        model,
        original_shape=original_shape,
        full_embeddings=full_embeddings.detach(),
        last_token_idx=last_token_idx
    )
    
    # Compute Jacobian
    jacobian = torch.autograd.functional.jacobian(
        forward_fn,
        flattened_embedding,
        vectorize=True
    )
    
    print(f"Jacobian shape: {jacobian.shape}")
    return jacobian, last_token_embedding

# -----------------------------
# Perform SVD
# -----------------------------
def perform_svd(jacobian):
    """
    Perform SVD on Jacobian.
    Returns U, S, Vt where:
    - U: left singular vectors (output space directions)
    - S: singular values
    - Vt: right singular vectors (input space directions)
    """
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    print(f"\nSVD Results:")
    print(f"  U shape (output directions): {U.shape}")
    print(f"  S shape (singular values): {S.shape}")
    print(f"  Vt shape (input directions): {Vt.shape}")
    print(f"  Top 10 singular values: {S[:10].cpu().numpy()}")
    return U, S, Vt

# -----------------------------
# Compute Lipschitz constant along a direction
# -----------------------------
def compute_lipschitz_along_direction(model, full_embeddings, last_token_idx, direction, epsilon):
    """
    Compute Lipschitz constant: ||f(x + ε·δx) - f(x)|| / ε
    
    Args:
        model: LLaMA model
        full_embeddings: Full sequence embeddings
        last_token_idx: Index of last token
        direction: Direction vector in input space (should be normalized)
        epsilon: Perturbation magnitude
    
    Returns:
        lipschitz_constant: ||f(x + ε·δx) - f(x)|| / ε
        numerator: ||f(x + ε·δx) - f(x)||
        orig_token_id: Original predicted token ID
        pert_token_id: Perturbed predicted token ID
        orig_prob: Original prediction probability
        pert_prob: Perturbed prediction probability
        logit_diff: L2 norm of logit difference
        logits_orig: Original logits
        logits_pert: Perturbed logits
    """
    device = next(model.parameters()).device
    
    # Get f(x) - original output and logits
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=full_embeddings, output_hidden_states=True)
        fx = outputs_orig.hidden_states[-1][0, last_token_idx, :]
        logits_orig = outputs_orig.logits[0, last_token_idx, :]
        
        orig_token_id = torch.argmax(logits_orig).item()
        orig_prob = F.softmax(logits_orig, dim=-1)[orig_token_id].item()
    
    # Perturb input: x + ε·δx
    perturbed_embeddings = full_embeddings.clone()
    perturbed_embeddings[0, last_token_idx, :] += epsilon * direction
    
    # Get f(x + ε·δx) - perturbed output and logits
    with torch.no_grad():
        outputs_pert = model(inputs_embeds=perturbed_embeddings, output_hidden_states=True)
        fx_pert = outputs_pert.hidden_states[-1][0, last_token_idx, :]
        logits_pert = outputs_pert.logits[0, last_token_idx, :]
        
        pert_token_id = torch.argmax(logits_pert).item()
        pert_prob = F.softmax(logits_pert, dim=-1)[pert_token_id].item()
    
    # Compute Lipschitz constant
    numerator = torch.norm(fx_pert - fx).item()
    lipschitz = numerator / epsilon
    
    # Compute logit difference
    logit_diff = torch.norm(logits_pert - logits_orig).item()
    
    return lipschitz, numerator, orig_token_id, pert_token_id, orig_prob, pert_prob, logit_diff, logits_orig, logits_pert

# -----------------------------
# Main analysis function
# -----------------------------
def analyze_lipschitz_constants(model, tokenizer, input_text, top_k=5, epsilon_powers=None, analyze_dims=None):
    """
    Analyze Lipschitz constants along top-k singular directions.
    
    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        input_text: Input text
        top_k: Number of top singular directions to analyze (1-5)
        epsilon_powers: List of powers for epsilon values (e.g., [-1, -2, ..., -18] for 10^-1 to 10^-18)
        analyze_dims: List of dimensions to analyze (e.g., [50, 100, 200, 500, 1000, 2000, 4000, 4096])
    """
    if epsilon_powers is None:
        epsilon_powers = list(range(-1, -19, -1))  # 10^-1 to 10^-18
    
    if analyze_dims is None:
        analyze_dims = [50, 100, 200, 500, 1000, 2000, 4000, 4096]
    
    device = next(model.parameters()).device
    
    # Tokenize and get embeddings
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        full_embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    
    print(f"Input text: '{input_text}'")
    print(f"Sequence length: {inputs['input_ids'].shape[1]}")
    print(f"Last token index: {last_token_idx}")
    
    # Compute Jacobian
    print("\n" + "="*80)
    print("COMPUTING JACOBIAN")
    print("="*80)
    jacobian, last_token_embedding = compute_jacobian(model, full_embeddings, last_token_idx)
    
    # Perform SVD
    print("\n" + "="*80)
    print("PERFORMING SVD")
    print("="*80)
    U, S, Vt = perform_svd(jacobian)
    
    # Store original prediction once (same for all directions/epsilons)
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=full_embeddings, output_hidden_states=True)
        logits_orig = outputs_orig.logits[0, last_token_idx, :]
        orig_token_id = torch.argmax(logits_orig).item()
        orig_token = tokenizer.decode([orig_token_id])
        orig_prob = F.softmax(logits_orig, dim=-1)[orig_token_id].item()
    
    print(f"Original prediction: '{orig_token}' (token_id={orig_token_id}, prob={orig_prob:.4f})")
    
    # Analyze Lipschitz constants for top-k singular directions
    print("\n" + "="*80)
    print("COMPUTING LIPSCHITZ CONSTANTS FOR TOP-K SINGULAR DIRECTIONS")
    print("="*80)
    
    results_topk = []
    
    for k in range(1, top_k + 1):
        # Get k-th right singular vector (input direction)
        direction = Vt[k-1, :]  # Shape: [hidden_dim]
        
        print(f"\n--- Singular Direction {k} (Singular Value: {S[k-1].item():.6f}) ---")
        
        for power in epsilon_powers:
            epsilon = 10.0 ** power
            
            lipschitz, numerator, orig_tok_id, pert_tok_id, orig_p, pert_p, logit_diff, logits_orig_val, logits_pert_val = compute_lipschitz_along_direction(
                model, full_embeddings, last_token_idx, direction, epsilon
            )
            
            # Decode tokens
            orig_token_str = tokenizer.decode([orig_tok_id])
            pert_token_str = tokenizer.decode([pert_tok_id])
            token_changed = (orig_tok_id != pert_tok_id)
            
            # Get the logit values for the predicted tokens
            orig_logit_val = logits_orig_val[orig_tok_id].item()
            pert_logit_val = logits_pert_val[pert_tok_id].item()
            
            results_topk.append({
                'analysis_type': 'top_k_singular',
                'dimension': k,
                'singular_value': S[k-1].item(),
                'epsilon_power': power,
                'epsilon': epsilon,
                'lipschitz_constant': lipschitz,
                'output_diff_norm': numerator,
                'logit_diff': logit_diff,
                'orig_token_id': orig_tok_id,
                'orig_token': orig_token_str,
                'orig_prob': orig_p,
                'orig_logit': orig_logit_val,
                'pert_token_id': pert_tok_id,
                'pert_token': pert_token_str,
                'pert_prob': pert_p,
                'pert_logit': pert_logit_val,
                'token_changed': token_changed
            })
            
            # Print with token information
            if token_changed:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}) → '{pert_token_str}' (p={pert_p:.4f}) ✓")
            else:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}→{pert_p:.4f})")
    
    # Analyze Lipschitz constants for specific dimensions
    print("\n" + "="*80)
    print("COMPUTING LIPSCHITZ CONSTANTS FOR SPECIFIC DIMENSIONS")
    print("="*80)
    
    results_dims = []
    
    for dim_idx in analyze_dims:
        if dim_idx > len(S):
            print(f"\nSkipping dimension {dim_idx} (exceeds available singular values: {len(S)})")
            continue
            
        # Get the direction for this dimension
        direction = Vt[dim_idx-1, :]  # -1 because we're 0-indexed
        
        print(f"\n--- Dimension {dim_idx} (Singular Value: {S[dim_idx-1].item():.6f}) ---")
        
        for power in epsilon_powers:
            epsilon = 10.0 ** power
            
            lipschitz, numerator, orig_tok_id, pert_tok_id, orig_p, pert_p, logit_diff, logits_orig_val, logits_pert_val = compute_lipschitz_along_direction(
                model, full_embeddings, last_token_idx, direction, epsilon
            )
            
            # Decode tokens
            orig_token_str = tokenizer.decode([orig_tok_id])
            pert_token_str = tokenizer.decode([pert_tok_id])
            token_changed = (orig_tok_id != pert_tok_id)
            
            # Get the logit values for the predicted tokens
            orig_logit_val = logits_orig_val[orig_tok_id].item()
            pert_logit_val = logits_pert_val[pert_tok_id].item()
            
            results_dims.append({
                'analysis_type': 'specific_dimension',
                'dimension': dim_idx,
                'singular_value': S[dim_idx-1].item(),
                'epsilon_power': power,
                'epsilon': epsilon,
                'lipschitz_constant': lipschitz,
                'output_diff_norm': numerator,
                'logit_diff': logit_diff,
                'orig_token_id': orig_tok_id,
                'orig_token': orig_token_str,
                'orig_prob': orig_p,
                'orig_logit': orig_logit_val,
                'pert_token_id': pert_tok_id,
                'pert_token': pert_token_str,
                'pert_prob': pert_p,
                'pert_logit': pert_logit_val,
                'token_changed': token_changed
            })
            
            # Print with token information
            if token_changed:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}) → '{pert_token_str}' (p={pert_p:.4f}) ✓")
            else:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}→{pert_p:.4f})")
    
    # Combine all results
    all_results = results_topk + results_dims
    
    return pd.DataFrame(all_results), jacobian, U, S, Vt

# -----------------------------
# Run analysis
# -----------------------------
if __name__ == "__main__":
    print("Loading LLaMA 3.1 model...")
    model, tokenizer = load_llama_model()
    
    # Test with different inputs
    test_inputs = [
        "The capital of France is",
        # "Machine learning is a subset of",
        # "To solve this problem, we need to"
    ]
    
    all_results = {}
    
    for i, input_text in enumerate(test_inputs):
        print("\n" + "="*80)
        print(f"TEST CASE {i+1}: '{input_text}'")
        print("="*80)
        
        results_df, jacobian, U, S, Vt = analyze_lipschitz_constants(
            model, tokenizer, input_text, 
            top_k=5,
            epsilon_powers=list(range(-1, -19, -1)),
            analyze_dims=[50, 100, 200, 500, 1000, 2000, 4000, 4096]
        )
        
        all_results[f"test_case_{i+1}"] = {
            'results': results_df,
            'jacobian': jacobian,
            'U': U,
            'S': S,
            'Vt': Vt
        }
        
        # Save results
        filename = f"lipschitz_analysis_test_{i+1}_complete.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nSaved results to {filename}")
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    for test_name, data in all_results.items():
        df = data['results']
        print(f"\n{test_name}:")
        print(f"  Top 5 singular values: {data['S'][:5].cpu().numpy()}")
        
        # Summary for top-k analysis
        topk_df = df[df['analysis_type'] == 'top_k_singular']
        if len(topk_df) > 0:
            print(f"\n  Top-K Singular Directions Analysis:")
            for k in range(1, 6):
                subset = topk_df[topk_df['dimension'] == k]
                if len(subset) > 0:
                    print(f"    Direction {k} (σ={data['S'][k-1].item():.4f}):")
                    print(f"      Lipschitz range: [{subset['lipschitz_constant'].min():.6f}, {subset['lipschitz_constant'].max():.6f}]")
                    print(f"      Mean Lipschitz: {subset['lipschitz_constant'].mean():.6f}")
        
        # Summary for specific dimensions analysis
        dims_df = df[df['analysis_type'] == 'specific_dimension']
        if len(dims_df) > 0:
            print(f"\n  Specific Dimensions Analysis:")
            for dim in [50, 100, 200, 500, 1000, 2000, 4000, 4096]:
                subset = dims_df[dims_df['dimension'] == dim]
                if len(subset) > 0:
                    print(f"    Dimension {dim} (σ={subset['singular_value'].iloc[0]:.6f}):")
                    print(f"      Lipschitz range: [{subset['lipschitz_constant'].min():.6f}, {subset['lipschitz_constant'].max():.6f}]")
                    print(f"      Mean Lipschitz: {subset['lipschitz_constant'].mean():.6f}")
    
    print("\nDone!")