"""
Lipschitz Constant Computation for GPT-OSS Model (CPU-Only Version)

This script rigorously computes the Lipschitz constant of the GPT-OSS model's
transformation from input embeddings to final hidden states using Jacobian
analysis and Singular Value Decomposition (SVD).

This version runs entirely on CPU to avoid GPU OOM errors.
All computations are done in bfloat16 to match the model's dtype.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os

warnings.filterwarnings('ignore')

# Force CPU usage - set before any CUDA operations
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# SINGLE CONTROL VARIABLE FOR PRECISION
# Use bfloat16 to match GPT-OSS model's internal precision
PRECISION = torch.float32 #torch.bfloat16

# -----------------------------
# Load GPT-OSS model and tokenizer
# -----------------------------
def load_gpt_model():
    """Load GPT-OSS model and tokenizer - CPU only"""
    model_id = "openai/gpt-oss-20b"
    
    print(f"Loading model: {model_id}")
    print("NOTE: This will run on CPU and may be slow")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Force everything to CPU to avoid GPU OOM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"},
        torch_dtype=torch.bfloat16,  # Load as bfloat16 first
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Then convert to desired precision if different
    if PRECISION != torch.bfloat16:
        print(f"Converting model from bfloat16 to {PRECISION}...")
        model = model.to(dtype=PRECISION)
    
    # Double-check everything is on CPU
    model = model.cpu()
    
    print(f"Model loaded successfully on CPU")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer

# -----------------------------
# Forward function for Jacobian computation
# -----------------------------
def model_forward_last_hidden(model, flattened_last_token_embedding, original_shape, 
                              full_embeddings, last_token_idx):
    """
    Forward pass that takes ONLY the last token embedding as input for Jacobian.
    Returns the last token's final hidden state (before layer norm).
    
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
    
    # Get last token's hidden state from last layer (before final layer norm)
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
        last_token_embedding: The original last token embedding
    """
    dtype = full_embeddings.dtype
    
    # Extract last token embedding - keep in bfloat16
    last_token_embedding = full_embeddings[0, last_token_idx, :].clone()
    original_shape = last_token_embedding.shape
    
    # Flatten for Jacobian computation - keep in bfloat16
    flattened_embedding = last_token_embedding.flatten()
    flattened_embedding = flattened_embedding.detach().requires_grad_(True)
    
    print(f"\nComputing Jacobian:")
    print(f"  Embedding shape: {original_shape}")
    print(f"  Flattened shape: {flattened_embedding.shape}")
    print(f"  Dtype: {flattened_embedding.dtype}")
    
    # Create partial function
    from functools import partial
    
    forward_fn = partial(
        model_forward_last_hidden,
        model,
        original_shape=original_shape,
        full_embeddings=full_embeddings.detach(),  # Keep in bfloat16
        last_token_idx=last_token_idx
    )
    
    # Compute Jacobian using torch.autograd
    print("  Computing Jacobian matrix (this may take a while on CPU)...")
    print("  NOTE: Using bfloat16 for consistency with model dtype")
    
    try:
        jacobian = torch.autograd.functional.jacobian(
            forward_fn,
            flattened_embedding,
            vectorize=True
        )
        print("  ✓ Jacobian computed with vectorize=True")
    except RuntimeError as e:
        print(f"  ⚠ Error with vectorize=True: {e}")
        print("  Trying with vectorize=False (slower but more compatible)...")
        jacobian = torch.autograd.functional.jacobian(
            forward_fn,
            flattened_embedding,
            vectorize=False
        )
        print("  ✓ Jacobian computed with vectorize=False")
    
    print(f"  Jacobian shape: {jacobian.shape}")
    print(f"  Jacobian dtype: {jacobian.dtype}")
    # Convert to float32 only for norm calculation (more accurate)
    print(f"  Jacobian norm: {torch.norm(jacobian.float()).item():.6f}")
    
    return jacobian, last_token_embedding

# -----------------------------
# Perform SVD
# -----------------------------
def perform_svd(jacobian):
    """
    Perform SVD on Jacobian matrix.
    Returns U, S, Vt where:
    - U: left singular vectors (output space directions)
    - S: singular values (sorted in descending order)
    - Vt: right singular vectors (input space directions)
    
    The largest singular value S[0] = Lipschitz constant
    """
    print("\nPerforming SVD on Jacobian...")
    print("  Converting to float32 for SVD stability...")
    
    # Convert to float32 for SVD (more stable and accurate)
    jacobian_float = jacobian.float()
    
    U, S, Vt = torch.linalg.svd(jacobian_float, full_matrices=False)
    
    print(f"  U shape (output directions): {U.shape}")
    print(f"  S shape (singular values): {S.shape}")
    print(f"  Vt shape (input directions): {Vt.shape}")
    
    print(f"\n  Top 10 singular values:")
    for i, sv in enumerate(S[:10].cpu().numpy()):
        print(f"    σ_{i+1}: {sv:.6f}")
    
    print(f"\n  *** LIPSCHITZ CONSTANT = {S[0].item():.6f} ***")
    print(f"      (This is the largest singular value σ_max)")
    
    return U, S, Vt

# -----------------------------
# Compute empirical Lipschitz constant along a direction
# -----------------------------
def compute_lipschitz_along_direction(model, tokenizer, full_embeddings, last_token_idx, 
                                     direction, epsilon):
    """
    Empirically compute Lipschitz constant: ||f(x + ε·δx) - f(x)|| / ε
    
    Args:
        model: GPT-OSS model
        tokenizer: Tokenizer
        full_embeddings: Full sequence embeddings
        last_token_idx: Index of last token
        direction: Direction vector in input space (should be normalized)
        epsilon: Perturbation magnitude
    
    Returns:
        (tuple): All analysis values
    """
    dtype = full_embeddings.dtype
    
    # Ensure direction matches dtype (convert from float32 SVD back to bfloat16)
    direction = direction.to(dtype)
    
    # Get f(x) - original output
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=full_embeddings, output_hidden_states=True)
        fx = outputs_orig.hidden_states[-1][0, last_token_idx, :]
        logits_orig = outputs_orig.logits[0, last_token_idx, :]
        
        orig_token_id = torch.argmax(logits_orig).item()
        orig_prob = F.softmax(logits_orig.float(), dim=-1)[orig_token_id].item()
        orig_token = tokenizer.decode([orig_token_id])
        orig_logit = logits_orig[orig_token_id].item()
    
    # Perturb input: x + ε·δx
    perturbed_embeddings = full_embeddings.clone()
    perturbed_embeddings[0, last_token_idx, :] += epsilon * direction
    
    # Get f(x + ε·δx) - perturbed output
    with torch.no_grad():
        outputs_pert = model(inputs_embeds=perturbed_embeddings, output_hidden_states=True)
        fx_pert = outputs_pert.hidden_states[-1][0, last_token_idx, :]
        logits_pert = outputs_pert.logits[0, last_token_idx, :]
        
        pert_token_id = torch.argmax(logits_pert).item()
        pert_prob = F.softmax(logits_pert.float(), dim=-1)[pert_token_id].item()
        pert_token = tokenizer.decode([pert_token_id])
        pert_logit = logits_pert[pert_token_id].item()
    
    # Compute Lipschitz constant (convert to float for numerical stability)
    numerator = torch.norm(fx_pert.float() - fx.float()).item()
    lipschitz = numerator / epsilon
    
    # Compute logit difference
    logit_diff = torch.norm(logits_pert.float() - logits_orig.float()).item()
    
    token_changed = (orig_token_id != pert_token_id)
    
    return (
        lipschitz, numerator,
        orig_token, orig_token_id, orig_prob, orig_logit,
        pert_token, pert_token_id, pert_prob, pert_logit,
        token_changed, logit_diff
    )

# -----------------------------
# Main analysis function
# -----------------------------
def analyze_lipschitz_constants(model, tokenizer, input_text, 
                                top_k=5, 
                                epsilon_powers=None,
                                analyze_dims=None):
    """
    Complete Lipschitz constant analysis.
    
    Args:
        model: GPT-OSS model
        tokenizer: Tokenizer
        input_text: Input text prompt
        top_k: Number of top singular directions to analyze (1-5)
        epsilon_powers: List of powers for epsilon values (e.g., [-1, -2, ..., -10])
        analyze_dims: List of specific dimensions to analyze
    
    Returns:
        results_df: DataFrame with all results
        jacobian: Jacobian matrix
        U, S, Vt: SVD components
    """
    if epsilon_powers is None:
        epsilon_powers = list(range(-1, -11, -1))  # 10^-1 to 10^-10
    
    if analyze_dims is None:
        # GPT-OSS has 2880 hidden dim
        analyze_dims = [10, 50, 100, 200, 500, 1000, 2000, 2880]
    
    dtype = next(model.parameters()).dtype
    
    # Tokenize and get embeddings
    print(f"\nTokenizing input: '{input_text}'")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    # Ensure inputs are on CPU
    inputs = {k: v.cpu() for k, v in inputs.items()}
    
    # Get embeddings - keep in bfloat16
    with torch.no_grad():
        full_embeddings = model.model.embed_tokens(inputs["input_ids"]).to(dtype)
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    
    print(f"Sequence length: {inputs['input_ids'].shape[1]}")
    print(f"Last token index: {last_token_idx}")
    print(f"Embedding dimension: {full_embeddings.shape[-1]}")
    print(f"Embedding dtype: {full_embeddings.dtype}")
    print(f"Embedding device: {full_embeddings.device}")
    
    # Compute Jacobian
    print("\n" + "="*80)
    print("STEP 1: COMPUTING JACOBIAN MATRIX")
    print("="*80)
    jacobian, last_token_embedding = compute_jacobian(model, full_embeddings, last_token_idx)
    
    # Perform SVD
    print("\n" + "="*80)
    print("STEP 2: PERFORMING SVD TO FIND LIPSCHITZ CONSTANT")
    print("="*80)
    U, S, Vt = perform_svd(jacobian)
    
    # Store original prediction
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=full_embeddings)
        logits_orig = outputs_orig.logits[0, last_token_idx, :]
        orig_token_id = torch.argmax(logits_orig).item()
        orig_token = tokenizer.decode([orig_token_id])
        orig_prob = F.softmax(logits_orig.float(), dim=-1)[orig_token_id].item()
    
    print(f"\nOriginal prediction: '{orig_token}' (token_id={orig_token_id}, prob={orig_prob:.4f})")
    
    # Analyze top-k singular directions
    print("\n" + "="*80)
    print("STEP 3: EMPIRICAL VERIFICATION ALONG TOP-K SINGULAR DIRECTIONS")
    print("="*80)
    
    results = []
    
    for k in range(1, min(top_k + 1, len(S) + 1)):
        # Get k-th right singular vector (input direction)
        direction = Vt[k-1, :]
        sigma_k = S[k-1].item()
        
        print(f"\n--- Singular Direction {k} (σ_{k} = {sigma_k:.6f}) ---")
        
        for power in epsilon_powers:
            epsilon = 10.0 ** power
            
            (
                lipschitz, numerator,
                orig_token_str, orig_token_id, orig_p, orig_logit_val,
                pert_token_str, pert_token_id, pert_p, pert_logit_val,
                token_changed, logit_diff
            ) = compute_lipschitz_along_direction(
                model, tokenizer, full_embeddings, last_token_idx, direction, epsilon
            )
            
            results.append({
                'analysis_type': 'top_k_singular',
                'dimension': k,
                'singular_value': sigma_k,
                'epsilon_power': power,
                'epsilon': epsilon,
                'lipschitz_empirical': lipschitz,
                'lipschitz_theoretical': sigma_k,
                'ratio_empirical_to_theoretical': lipschitz / sigma_k,
                'output_diff_norm': numerator,
                'logit_diff': logit_diff,
                'orig_token_id': orig_token_id,
                'orig_token': orig_token_str,
                'orig_prob': orig_p,
                'orig_logit': orig_logit_val,
                'pert_token_id': pert_token_id,
                'pert_token': pert_token_str,
                'pert_prob': pert_p,
                'pert_logit': pert_logit_val,
                'token_changed': token_changed
            })
            
            # Print results - NEW FORMAT
            if token_changed:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}) → '{pert_token_str}' (p={pert_p:.4f}) ✓")
            else:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}→{pert_p:.4f})")
    
    # Analyze specific dimensions
    print("\n" + "="*80)
    print("STEP 4: EMPIRICAL VERIFICATION FOR SPECIFIC DIMENSIONS")
    print("="*80)
    
    for dim_idx in analyze_dims:
        if dim_idx > len(S):
            print(f"\nSkipping dimension {dim_idx} (exceeds available singular values: {len(S)})")
            continue
        
        direction = Vt[dim_idx-1, :]
        sigma = S[dim_idx-1].item()
        
        print(f"\n--- Dimension {dim_idx} (σ = {sigma:.6f}) ---")
        
        for power in epsilon_powers:
            epsilon = 10.0 ** power
            
            (
                lipschitz, numerator,
                orig_token_str, orig_token_id, orig_p, orig_logit_val,
                pert_token_str, pert_token_id, pert_p, pert_logit_val,
                token_changed, logit_diff
            ) = compute_lipschitz_along_direction(
                model, tokenizer, full_embeddings, last_token_idx, direction, epsilon
            )
            
            results.append({
                'analysis_type': 'specific_dimension',
                'dimension': dim_idx,
                'singular_value': sigma,
                'epsilon_power': power,
                'epsilon': epsilon,
                'lipschitz_empirical': lipschitz,
                'lipschitz_theoretical': sigma,
                'ratio_empirical_to_theoretical': lipschitz / sigma,
                'output_diff_norm': numerator,
                'logit_diff': logit_diff,
                'orig_token_id': orig_token_id,
                'orig_token': orig_token_str,
                'orig_prob': orig_p,
                'orig_logit': orig_logit_val,
                'pert_token_id': pert_token_id,
                'pert_token': pert_token_str,
                'pert_prob': pert_p,
                'pert_logit': pert_logit_val,
                'token_changed': token_changed
            })
            
            # Print results - NEW FORMAT
            if token_changed:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}) → '{pert_token_str}' (p={pert_p:.4f}) ✓")
            else:
                print(f"  ε=10^{power:3d} ({epsilon:.2e}): Lipschitz={lipschitz:.6f}, ||Δf||={numerator:.6e}, logit: {orig_logit_val:.4f}→{pert_logit_val:.4f}, ||Δlogits||={logit_diff:.6e} | '{orig_token_str}' (p={orig_p:.4f}→{pert_p:.4f})")
    
    return pd.DataFrame(results), jacobian, U, S, Vt

# -----------------------------
# Run analysis
# -----------------------------
if __name__ == "__main__":
    print(f"Using precision: {PRECISION}")
    print("Running on CPU only (GPU disabled)")
    print("Loading GPT-OSS model (this will take time and use significant RAM)...")
    
    model, tokenizer = load_gpt_model()
    
    # Test inputs
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
            epsilon_powers=list(range(-1, -11, -1)),
            analyze_dims=[10, 50, 100, 200, 500, 1000, 2000, 2880]
        )
        
        all_results[f"test_case_{i+1}"] = {
            'results': results_df,
            'jacobian': jacobian,
            'U': U,
            'S': S,
            'Vt': Vt
        }
        
        # Save results
        filename = f"gpt_oss_lipschitz_test_{i+1}_complete.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n✓ Saved results to {filename}")
    
    # Summary analysis
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for test_name, data in all_results.items():
        df = data['results']
        S_vals = data['S']
        
        print(f"\n{test_name}:")
        print(f"  LIPSCHITZ CONSTANT: {S_vals[0].item():.6f}")
        print(f"  Top 5 singular values: {S_vals[:5].cpu().numpy()}")
        
        # Analysis for top-k
        topk_df = df[df['analysis_type'] == 'top_k_singular']
        if len(topk_df) > 0:
            print(f"\n  Top-K Analysis:")
            for k in range(1, 6):
                subset = topk_df[topk_df['dimension'] == k]
                if len(subset) > 0:
                    print(f"    Direction {k} (σ={S_vals[k-1].item():.4f}):")
                    print(f"      Empirical Lipschitz range: [{subset['lipschitz_empirical'].min():.6f}, "
                          f"{subset['lipschitz_empirical'].max():.6f}]")
                    print(f"      Mean ratio (empirical/theoretical): {subset['ratio_empirical_to_theoretical'].mean():.4f}")
                    print(f"      Token changes: {subset['token_changed'].sum()} / {len(subset)}")
        
        # Analysis for specific dimensions
        dims_df = df[df['analysis_type'] == 'specific_dimension']
        if len(dims_df) > 0:
            print(f"\n  Specific Dimensions Analysis:")
            unique_dims = sorted(dims_df['dimension'].unique())
            for dim in unique_dims:
                subset = dims_df[dims_df['dimension'] == dim]
                if len(subset) > 0:
                    print(f"    Dimension {dim} (σ={subset['singular_value'].iloc[0]:.6f}):")
                    print(f"      Lipschitz range: [{subset['lipschitz_empirical'].min():.6f}, "
                          f"{subset['lipschitz_empirical'].max():.6f}]")
                    print(f"      Token changes: {subset['token_changed'].sum()} / {len(subset)}")
    
    print(f"\n✓ Analysis complete using precision: {PRECISION}")
    print("Done!") 
# """
# Lipschitz Constant Computation for GPT-OSS Model (CPU-Only Version)

# This script rigorously computes the Lipschitz constant of the GPT-OSS model's
# transformation from input embeddings to final hidden states using Jacobian
# analysis and Singular Value Decomposition (SVD).

# This version runs entirely on CPU to avoid GPU OOM errors.
# All computations are done in bfloat16 to match the model's dtype.
# """

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import warnings
# import os

# warnings.filterwarnings('ignore')

# # Force CPU usage - set before any CUDA operations
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# # SINGLE CONTROL VARIABLE FOR PRECISION
# # Use bfloat16 to match GPT-OSS model's internal precision
# PRECISION = torch.bfloat16

# # -----------------------------
# # Load GPT-OSS model and tokenizer
# # -----------------------------
# def load_gpt_model():
#     """Load GPT-OSS model and tokenizer - CPU only"""
#     model_id = "openai/gpt-oss-20b"
    
#     print(f"Loading model: {model_id}")
#     print("NOTE: This will run on CPU and may be slow")
    
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # Force everything to CPU to avoid GPU OOM
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map={"": "cpu"},  # Force ALL layers to CPU
#         torch_dtype=PRECISION,
#         trust_remote_code=True,
#         low_cpu_mem_usage=True
#     )
    
#     # Double-check everything is on CPU
#     model = model.cpu()
    
#     print(f"Model loaded successfully on CPU")
#     print(f"Model device: {next(model.parameters()).device}")
#     print(f"Model dtype: {next(model.parameters()).dtype}")
    
#     return model, tokenizer

# # -----------------------------
# # Forward function for Jacobian computation
# # -----------------------------
# def model_forward_last_hidden(model, flattened_last_token_embedding, original_shape, 
#                               full_embeddings, last_token_idx):
#     """
#     Forward pass that takes ONLY the last token embedding as input for Jacobian.
#     Returns the last token's final hidden state (before layer norm).
    
#     Args:
#         flattened_last_token_embedding: Flattened last token embedding [embed_dim]
#         original_shape: Original shape of last token embedding
#         full_embeddings: Full sequence embeddings (kept fixed except last token)
#         last_token_idx: Index of the last token
#     """
#     # Restore the last token embedding shape
#     last_token_embedding = flattened_last_token_embedding.view(original_shape)
    
#     # Replace the last token in full embeddings
#     modified_embeddings = full_embeddings.clone()
#     modified_embeddings[0, last_token_idx, :] = last_token_embedding
    
#     # Forward pass
#     outputs = model(inputs_embeds=modified_embeddings, output_hidden_states=True)
    
#     # Get last token's hidden state from last layer (before final layer norm)
#     last_hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
    
#     return last_hidden_state

# # -----------------------------
# # Compute Jacobian
# # -----------------------------
# def compute_jacobian(model, full_embeddings, last_token_idx):
#     """
#     Compute Jacobian of last token's final hidden state w.r.t last token's input embedding.
    
#     Returns:
#         jacobian: Shape [hidden_dim_output, hidden_dim_input]
#         last_token_embedding: The original last token embedding
#     """
#     dtype = full_embeddings.dtype
    
#     # Extract last token embedding - keep in bfloat16
#     last_token_embedding = full_embeddings[0, last_token_idx, :].clone()
#     original_shape = last_token_embedding.shape
    
#     # Flatten for Jacobian computation - keep in bfloat16
#     flattened_embedding = last_token_embedding.flatten()
#     flattened_embedding = flattened_embedding.detach().requires_grad_(True)
    
#     print(f"\nComputing Jacobian:")
#     print(f"  Embedding shape: {original_shape}")
#     print(f"  Flattened shape: {flattened_embedding.shape}")
#     print(f"  Dtype: {flattened_embedding.dtype}")
    
#     # Create partial function
#     from functools import partial
    
#     forward_fn = partial(
#         model_forward_last_hidden,
#         model,
#         original_shape=original_shape,
#         full_embeddings=full_embeddings.detach(),  # Keep in bfloat16
#         last_token_idx=last_token_idx
#     )
    
#     # Compute Jacobian using torch.autograd
#     print("  Computing Jacobian matrix (this may take a while on CPU)...")
#     print("  NOTE: Using bfloat16 for consistency with model dtype")
    
#     try:
#         jacobian = torch.autograd.functional.jacobian(
#             forward_fn,
#             flattened_embedding,
#             vectorize=True
#         )
#         print("  ✓ Jacobian computed with vectorize=True")
#     except RuntimeError as e:
#         print(f"  ⚠ Error with vectorize=True: {e}")
#         print("  Trying with vectorize=False (slower but more compatible)...")
#         jacobian = torch.autograd.functional.jacobian(
#             forward_fn,
#             flattened_embedding,
#             vectorize=False
#         )
#         print("  ✓ Jacobian computed with vectorize=False")
    
#     print(f"  Jacobian shape: {jacobian.shape}")
#     print(f"  Jacobian dtype: {jacobian.dtype}")
#     # Convert to float32 only for norm calculation (more accurate)
#     print(f"  Jacobian norm: {torch.norm(jacobian.float()).item():.6f}")
    
#     return jacobian, last_token_embedding

# # -----------------------------
# # Perform SVD
# # -----------------------------
# def perform_svd(jacobian):
#     """
#     Perform SVD on Jacobian matrix.
#     Returns U, S, Vt where:
#     - U: left singular vectors (output space directions)
#     - S: singular values (sorted in descending order)
#     - Vt: right singular vectors (input space directions)
    
#     The largest singular value S[0] = Lipschitz constant
#     """
#     print("\nPerforming SVD on Jacobian...")
#     print("  Converting to float32 for SVD stability...")
    
#     # Convert to float32 for SVD (more stable and accurate)
#     jacobian_float = jacobian.float()
    
#     U, S, Vt = torch.linalg.svd(jacobian_float, full_matrices=False)
    
#     print(f"  U shape (output directions): {U.shape}")
#     print(f"  S shape (singular values): {S.shape}")
#     print(f"  Vt shape (input directions): {Vt.shape}")
    
#     print(f"\n  Top 10 singular values:")
#     for i, sv in enumerate(S[:10].cpu().numpy()):
#         print(f"    σ_{i+1}: {sv:.6f}")
    
#     print(f"\n  *** LIPSCHITZ CONSTANT = {S[0].item():.6f} ***")
#     print(f"      (This is the largest singular value σ_max)")
    
#     return U, S, Vt

# # -----------------------------
# # Compute empirical Lipschitz constant along a direction
# # -----------------------------
# def compute_lipschitz_along_direction(model, tokenizer, full_embeddings, last_token_idx, 
#                                      direction, epsilon):
#     """
#     Empirically compute Lipschitz constant: ||f(x + ε·δx) - f(x)|| / ε
    
#     Args:
#         model: GPT-OSS model
#         tokenizer: Tokenizer
#         full_embeddings: Full sequence embeddings
#         last_token_idx: Index of last token
#         direction: Direction vector in input space (should be normalized)
#         epsilon: Perturbation magnitude
    
#     Returns:
#         lipschitz_constant: Empirical Lipschitz constant
#         numerator: ||f(x + ε·δx) - f(x)||
#         token_info: Dictionary with prediction information
#     """
#     dtype = full_embeddings.dtype
    
#     # Ensure direction matches dtype (convert from float32 SVD back to bfloat16)
#     direction = direction.to(dtype)
    
#     # Get f(x) - original output
#     with torch.no_grad():
#         outputs_orig = model(inputs_embeds=full_embeddings, output_hidden_states=True)
#         fx = outputs_orig.hidden_states[-1][0, last_token_idx, :]
#         logits_orig = outputs_orig.logits[0, last_token_idx, :]
        
#         orig_token_id = torch.argmax(logits_orig).item()
#         orig_prob = F.softmax(logits_orig.float(), dim=-1)[orig_token_id].item()
#         orig_token = tokenizer.decode([orig_token_id])
#         orig_logit = logits_orig[orig_token_id].item()
    
#     # Perturb input: x + ε·δx
#     perturbed_embeddings = full_embeddings.clone()
#     perturbed_embeddings[0, last_token_idx, :] += epsilon * direction
    
#     # Get f(x + ε·δx) - perturbed output
#     with torch.no_grad():
#         outputs_pert = model(inputs_embeds=perturbed_embeddings, output_hidden_states=True)
#         fx_pert = outputs_pert.hidden_states[-1][0, last_token_idx, :]
#         logits_pert = outputs_pert.logits[0, last_token_idx, :]
        
#         pert_token_id = torch.argmax(logits_pert).item()
#         pert_prob = F.softmax(logits_pert.float(), dim=-1)[pert_token_id].item()
#         pert_token = tokenizer.decode([pert_token_id])
#         pert_logit = logits_pert[pert_token_id].item()
    
#     # Compute Lipschitz constant (convert to float for numerical stability)
#     numerator = torch.norm(fx_pert.float() - fx.float()).item()
#     lipschitz = numerator / epsilon
    
#     # Compute logit difference
#     logit_diff = torch.norm(logits_pert.float() - logits_orig.float()).item()
    
#     token_info = {
#         'orig_token': orig_token,
#         'orig_token_id': orig_token_id,
#         'orig_prob': orig_prob,
#         'orig_logit': orig_logit,
#         'pert_token': pert_token,
#         'pert_token_id': pert_token_id,
#         'pert_prob': pert_prob,
#         'pert_logit': pert_logit,
#         'token_changed': (orig_token_id != pert_token_id),
#         'logit_diff': logit_diff
#     }
    
#     return lipschitz, numerator, token_info

# # -----------------------------
# # Main analysis function
# # -----------------------------
# def analyze_lipschitz_constants(model, tokenizer, input_text, 
#                                 top_k=5, 
#                                 epsilon_powers=None,
#                                 analyze_dims=None):
#     """
#     Complete Lipschitz constant analysis.
    
#     Args:
#         model: GPT-OSS model
#         tokenizer: Tokenizer
#         input_text: Input text prompt
#         top_k: Number of top singular directions to analyze (1-5)
#         epsilon_powers: List of powers for epsilon values (e.g., [-1, -2, ..., -10])
#         analyze_dims: List of specific dimensions to analyze
    
#     Returns:
#         results_df: DataFrame with all results
#         jacobian: Jacobian matrix
#         U, S, Vt: SVD components
#     """
#     if epsilon_powers is None:
#         epsilon_powers = list(range(-1, -11, -1))  # 10^-1 to 10^-10
    
#     if analyze_dims is None:
#         # GPT-OSS has 2880 hidden dim
#         analyze_dims = [10, 50, 100, 200, 500, 1000, 2000, 2880]
    
#     dtype = next(model.parameters()).dtype
    
#     # Tokenize and get embeddings
#     print(f"\nTokenizing input: '{input_text}'")
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
#     # Ensure inputs are on CPU
#     inputs = {k: v.cpu() for k, v in inputs.items()}
    
#     # Get embeddings - keep in bfloat16
#     with torch.no_grad():
#         full_embeddings = model.model.embed_tokens(inputs["input_ids"]).to(dtype)
    
#     last_token_idx = inputs["input_ids"].shape[1] - 1
    
#     print(f"Sequence length: {inputs['input_ids'].shape[1]}")
#     print(f"Last token index: {last_token_idx}")
#     print(f"Embedding dimension: {full_embeddings.shape[-1]}")
#     print(f"Embedding dtype: {full_embeddings.dtype}")
#     print(f"Embedding device: {full_embeddings.device}")
    
#     # Compute Jacobian
#     print("\n" + "="*80)
#     print("STEP 1: COMPUTING JACOBIAN MATRIX")
#     print("="*80)
#     jacobian, last_token_embedding = compute_jacobian(model, full_embeddings, last_token_idx)
    
#     # Perform SVD
#     print("\n" + "="*80)
#     print("STEP 2: PERFORMING SVD TO FIND LIPSCHITZ CONSTANT")
#     print("="*80)
#     U, S, Vt = perform_svd(jacobian)
    
#     # Store original prediction
#     with torch.no_grad():
#         outputs_orig = model(inputs_embeds=full_embeddings)
#         logits_orig = outputs_orig.logits[0, last_token_idx, :]
#         orig_token_id = torch.argmax(logits_orig).item()
#         orig_token = tokenizer.decode([orig_token_id])
#         orig_prob = F.softmax(logits_orig.float(), dim=-1)[orig_token_id].item()
    
#     print(f"\nOriginal prediction: '{orig_token}' (token_id={orig_token_id}, prob={orig_prob:.4f})")
    
#     # Analyze top-k singular directions
#     print("\n" + "="*80)
#     print("STEP 3: EMPIRICAL VERIFICATION ALONG TOP-K SINGULAR DIRECTIONS")
#     print("="*80)
    
#     results = []
    
#     for k in range(1, min(top_k + 1, len(S) + 1)):
#         # Get k-th right singular vector (input direction)
#         direction = Vt[k-1, :]
#         sigma_k = S[k-1].item()
        
#         print(f"\n--- Singular Direction {k} (σ_{k} = {sigma_k:.6f}) ---")
        
#         for power in epsilon_powers:
#             epsilon = 10.0 ** power
            
#             lipschitz, output_diff, token_info = compute_lipschitz_along_direction(
#                 model, tokenizer, full_embeddings, last_token_idx, direction, epsilon
#             )
            
#             results.append({
#                 'analysis_type': 'top_k_singular',
#                 'dimension': k,
#                 'singular_value': sigma_k,
#                 'epsilon_power': power,
#                 'epsilon': epsilon,
#                 'lipschitz_empirical': lipschitz,
#                 'lipschitz_theoretical': sigma_k,
#                 'ratio_empirical_to_theoretical': lipschitz / sigma_k,
#                 'output_diff_norm': output_diff,
#                 **token_info
#             })
            
#             # Print results
#             status = " ✓ TOKEN CHANGED" if token_info['token_changed'] else ""
#             print(f"  ε=10^{power:3d} ({epsilon:.2e}): "
#                   f"L_emp={lipschitz:.6f}, L_theory={sigma_k:.6f}, "
#                   f"ratio={lipschitz/sigma_k:.4f} | "
#                   f"'{token_info['orig_token']}'→'{token_info['pert_token']}'{status}")
    
#     # Analyze specific dimensions
#     print("\n" + "="*80)
#     print("STEP 4: EMPIRICAL VERIFICATION FOR SPECIFIC DIMENSIONS")
#     print("="*80)
    
#     for dim_idx in analyze_dims:
#         if dim_idx > len(S):
#             print(f"\nSkipping dimension {dim_idx} (exceeds available singular values: {len(S)})")
#             continue
        
#         direction = Vt[dim_idx-1, :]
#         sigma = S[dim_idx-1].item()
        
#         print(f"\n--- Dimension {dim_idx} (σ = {sigma:.6f}) ---")
        
#         for power in epsilon_powers:
#             epsilon = 10.0 ** power
            
#             lipschitz, output_diff, token_info = compute_lipschitz_along_direction(
#                 model, tokenizer, full_embeddings, last_token_idx, direction, epsilon
#             )
            
#             results.append({
#                 'analysis_type': 'specific_dimension',
#                 'dimension': dim_idx,
#                 'singular_value': sigma,
#                 'epsilon_power': power,
#                 'epsilon': epsilon,
#                 'lipschitz_empirical': lipschitz,
#                 'lipschitz_theoretical': sigma,
#                 'ratio_empirical_to_theoretical': lipschitz / sigma,
#                 'output_diff_norm': output_diff,
#                 **token_info
#             })
            
#             status = " ✓" if token_info['token_changed'] else ""
#             print(f"  ε=10^{power:3d}: L_emp={lipschitz:.6f}, ratio={lipschitz/sigma:.4f}{status}")
    
#     return pd.DataFrame(results), jacobian, U, S, Vt

# # -----------------------------
# # Run analysis
# # -----------------------------
# if __name__ == "__main__":
#     print(f"Using precision: {PRECISION}")
#     print("Running on CPU only (GPU disabled)")
#     print("Loading GPT-OSS model (this will take time and use significant RAM)...")
    
#     model, tokenizer = load_gpt_model()
    
#     # Test inputs
#     test_inputs = [
#         "The capital of France is",
#         # "Machine learning is a subset of",
#         # "To solve this problem, we need to"
#     ]
    
#     all_results = {}
    
#     for i, input_text in enumerate(test_inputs):
#         print("\n" + "="*80)
#         print(f"TEST CASE {i+1}: '{input_text}'")
#         print("="*80)
        
#         results_df, jacobian, U, S, Vt = analyze_lipschitz_constants(
#             model, tokenizer, input_text,
#             top_k=5,
#             epsilon_powers=list(range(-1, -11, -1)),
#             analyze_dims=[10, 50, 100, 200, 500, 1000, 2000, 2880]
#         )
        
#         all_results[f"test_case_{i+1}"] = {
#             'results': results_df,
#             'jacobian': jacobian,
#             'U': U,
#             'S': S,
#             'Vt': Vt
#         }
        
#         # Save results
#         filename = f"gpt_oss_lipschitz_test_{i+1}_complete.csv"
#         results_df.to_csv(filename, index=False)
#         print(f"\n✓ Saved results to {filename}")
    
#     # Summary analysis
#     print("\n" + "="*80)
#     print("FINAL SUMMARY")
#     print("="*80)
    
#     for test_name, data in all_results.items():
#         df = data['results']
#         S_vals = data['S']
        
#         print(f"\n{test_name}:")
#         print(f"  LIPSCHITZ CONSTANT: {S_vals[0].item():.6f}")
#         print(f"  Top 5 singular values: {S_vals[:5].cpu().numpy()}")
        
#         # Analysis for top-k
#         topk_df = df[df['analysis_type'] == 'top_k_singular']
#         if len(topk_df) > 0:
#             print(f"\n  Top-K Analysis:")
#             for k in range(1, 6):
#                 subset = topk_df[topk_df['dimension'] == k]
#                 if len(subset) > 0:
#                     print(f"    Direction {k} (σ={S_vals[k-1].item():.4f}):")
#                     print(f"      Empirical Lipschitz range: [{subset['lipschitz_empirical'].min():.6f}, "
#                           f"{subset['lipschitz_empirical'].max():.6f}]")
#                     print(f"      Mean ratio (empirical/theoretical): {subset['ratio_empirical_to_theoretical'].mean():.4f}")
#                     print(f"      Token changes: {subset['token_changed'].sum()} / {len(subset)}")
        
#         # Analysis for specific dimensions
#         dims_df = df[df['analysis_type'] == 'specific_dimension']
#         if len(dims_df) > 0:
#             print(f"\n  Specific Dimensions Analysis:")
#             unique_dims = sorted(dims_df['dimension'].unique())
#             for dim in unique_dims:
#                 subset = dims_df[dims_df['dimension'] == dim]
#                 if len(subset) > 0:
#                     print(f"    Dimension {dim} (σ={subset['singular_value'].iloc[0]:.6f}):")
#                     print(f"      Lipschitz range: [{subset['lipschitz_empirical'].min():.6f}, "
#                           f"{subset['lipschitz_empirical'].max():.6f}]")
#                     print(f"      Token changes: {subset['token_changed'].sum()} / {len(subset)}")
    
#     print(f"\n✓ Analysis complete using precision: {PRECISION}")
#     print("Done!")