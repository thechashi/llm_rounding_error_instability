"""
Experiment 2 V2: Revised Average Lipschitz Constant Across Multiple Prompts

This is the REVISED VERSION of experiment2_avg_lipschitz.py, potentially
containing improvements, bug fixes, or methodological refinements.

Purpose:
--------
Same as experiment2_avg_lipschitz.py: Measures the average Lipschitz constant
of the Llama model across multiple diverse test prompts to provide a robust
characterization of model instability.


Relationship:
-------------
This file is a REVISION of experiment2_avg_lipschitz.py. The "v2" suffix
suggests:
1. Bug fixes from the original version
2. Methodological improvements
3. Additional analysis or better statistical handling
4. Code refactoring for clarity or efficiency

To understand what changed, compare this file with experiment2_avg_lipschitz.py:
- Check if test prompts differ
- Look for computational or statistical differences
- Verify if output format or analysis changed

Both versions:
- Extend lipschitz_const_llama.py to multiple prompts
- Compute Jacobian and SVD for each prompt
- Extract largest singular value (Lipschitz constant)
- Provide statistical summary across prompts

Methodology:
------------
Same core methodology as experiment2:
1. Load Llama model in float32
2. Define diverse test prompts
3. For each prompt:
   a. Compute Jacobian: ∂(last hidden state)/∂(last token embedding)
   b. Perform SVD on Jacobian
   c. Extract largest singular value σ_max
4. Compute statistics: mean, std, min, max, median
5. Generate distribution plots

Use Case:
---------
Use this version instead of experiment2_avg_lipschitz.py if:
- It contains known bug fixes
- It has improved statistical analysis
- It provides better visualization or output
- It's documented as the preferred version

Otherwise, compare results between both versions to understand sensitivity to
implementation details.

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, pandas, tqdm
- Llama-3.1-8B-Instruct model (float32)

Key Functions:
--------------
- load_llama_model(): Load model in float32
- model_forward_last_hidden(): Forward pass for Jacobian computation
- compute_jacobian(): Compute Jacobian matrix
- perform_svd(): SVD analysis to extract singular values
- (main loop): Iterate over prompts and compute average statistics

Output:
-------
- Timestamped results directory (results/exp2_v2_YYYY-MM-DD_HH-MM-SS/)
- CSV file with Lipschitz constants for each prompt
- Statistical summary
- Distribution plots

Note:
-----
Recommended to compare outputs from both experiment2 versions to validate
consistency or understand improvements. Consider consolidating if truly identical.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os
from datetime import datetime

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
# Compute Lipschitz constant along a direction (WITHOUT noise)
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
# Compute Lipschitz constant along a direction WITH RANDOM NOISE
# -----------------------------
def compute_lipschitz_along_direction_with_noise(model, full_embeddings, last_token_idx, direction, epsilon, 
                                                  n_samples=10):
    """
    Compute Lipschitz constant with random noise:
    1. Compute avg of f(x0 + epsilon*si + rand_v*si) over n samples
    2. Compute avg of f(x0 + rand_v*si) over n samples
    3. Return ||avg_rep1 - avg_rep2|| / epsilon
    
    where rand_v is a SCALAR: rand_v = noise_scale * uniform(-1, 1)
    
    Args:
        model: LLaMA model
        full_embeddings: Full sequence embeddings
        last_token_idx: Index of last token
        direction: Direction vector in input space (should be normalized) - this is the singular value direction
        epsilon: Perturbation magnitude
        n_samples: Number of random samples to generate
    
    Returns:
        lipschitz_constant: ||avg_rep1 - avg_rep2|| / epsilon
        numerator: ||avg_rep1 - avg_rep2||
        orig_token_id: Original predicted token ID (from avg_rep2)
        pert_token_id: Perturbed predicted token ID (from avg_rep1)
        orig_prob: Original prediction probability
        pert_prob: Perturbed prediction probability
        logit_diff: L2 norm of logit difference
        logits_orig: Original logits (from avg_rep2)
        logits_pert: Perturbed logits (from avg_rep1)
    """
    device = next(model.parameters()).device
    
    # Calculate noise scale: epsilon * (10^-2)
    noise_scale = epsilon * (10 ** -2)
    
    # Lists to store hidden states and logits for averaging
    rep1_hidden_states = []  # f(x0 + epsilon*si + rand_v*si)
    rep2_hidden_states = []  # f(x0 + rand_v*si)
    rep1_logits = []
    rep2_logits = []
    
    # Generate n random samples
    for sample_idx in range(n_samples):
        # Generate random SCALAR noise: rand_v = noise_scale * uniform(-1, 1)
        rand_v = noise_scale * (2 * torch.rand(1, device=device).item() - 1)  # scalar value
        
        # ===== REP1: f(x0 + epsilon*si + rand_v*si) =====
        perturbed_embeddings_rep1 = full_embeddings.clone()
        # Add perturbation: epsilon*direction + rand_v*direction = (epsilon + rand_v)*direction
        perturbed_embeddings_rep1[0, last_token_idx, :] += (epsilon + rand_v) * direction
        
        with torch.no_grad():
            outputs_rep1 = model(inputs_embeds=perturbed_embeddings_rep1, output_hidden_states=True)
            hidden_state_rep1 = outputs_rep1.hidden_states[-1][0, last_token_idx, :].cpu().detach()
            logits_rep1 = outputs_rep1.logits[0, last_token_idx, :].cpu().detach()
        
        rep1_hidden_states.append(hidden_state_rep1)
        rep1_logits.append(logits_rep1)
        
        # ===== REP2: f(x0 + rand_v*si) =====
        perturbed_embeddings_rep2 = full_embeddings.clone()
        # Add perturbation: rand_v*direction
        perturbed_embeddings_rep2[0, last_token_idx, :] += rand_v * direction
        
        with torch.no_grad():
            outputs_rep2 = model(inputs_embeds=perturbed_embeddings_rep2, output_hidden_states=True)
            hidden_state_rep2 = outputs_rep2.hidden_states[-1][0, last_token_idx, :].cpu().detach()
            logits_rep2 = outputs_rep2.logits[0, last_token_idx, :].cpu().detach()
        
        rep2_hidden_states.append(hidden_state_rep2)
        rep2_logits.append(logits_rep2)
    
    # Stack to create tensors of shape [n_samples, hidden_dim]
    rep1_hidden_states_tensor = torch.stack(rep1_hidden_states)  # Shape: [n_samples, 4096]
    rep2_hidden_states_tensor = torch.stack(rep2_hidden_states)  # Shape: [n_samples, 4096]
    
    # Average over all samples
    avg_rep1_hidden = rep1_hidden_states_tensor.mean(dim=0)  # Average hidden state
    avg_rep2_hidden = rep2_hidden_states_tensor.mean(dim=0)  # Average hidden state
    avg_rep1_logits = torch.stack(rep1_logits).mean(dim=0)  # Average logits
    avg_rep2_logits = torch.stack(rep2_logits).mean(dim=0)  # Average logits
    
    # Compute ||avg_rep1 - avg_rep2||
    numerator = torch.norm(avg_rep1_hidden - avg_rep2_hidden).item()
    
    # Compute Lipschitz constant
    lipschitz = numerator / epsilon
    
    # Get token predictions from averaged logits
    orig_token_id = torch.argmax(avg_rep2_logits).item()
    pert_token_id = torch.argmax(avg_rep1_logits).item()
    
    orig_prob = F.softmax(avg_rep2_logits, dim=-1)[orig_token_id].item()
    pert_prob = F.softmax(avg_rep1_logits, dim=-1)[pert_token_id].item()
    
    # Compute logit difference
    logit_diff = torch.norm(avg_rep1_logits - avg_rep2_logits).item()
    
    return (lipschitz, numerator, 
            orig_token_id, pert_token_id, 
            orig_prob, pert_prob, 
            logit_diff, 
            avg_rep2_logits, avg_rep1_logits,
            rep1_hidden_states_tensor, rep2_hidden_states_tensor)
# -----------------------------
# Main analysis function
# -----------------------------
def analyze_lipschitz_constants(model, tokenizer, input_text, top_k=5, epsilon_powers=None, analyze_dims=None,
                                use_noise=True, n_noise_samples_list=None, output_dir=None):
    """
    Analyze Lipschitz constants along top-k singular directions.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        input_text: Input text
        top_k: Number of top singular directions to analyze (1-5)
        epsilon_powers: List of powers for epsilon values (e.g., [-1, -2, ..., -18] for 10^-1 to 10^-18)
        analyze_dims: List of dimensions to analyze (e.g., [50, 100, 200, 500, 1000, 2000, 4000, 4096])
        use_noise: Whether to add random noise analysis
        n_noise_samples_list: List of sample sizes to test (e.g., [1, 2, 5, 2000])
        output_dir: Directory to save output files (default: "hidden_states_output")
    """
    if epsilon_powers is None:
        epsilon_powers = list(range(-1, -19, -1))  # 10^-1 to 10^-18
    
    if analyze_dims is None:
        analyze_dims = [50, 100, 200, 500, 1000, 2000, 4000, 4096]
    
    if n_noise_samples_list is None:
        n_noise_samples_list = [1, 2, 5, 2000]
    
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
            
            # Original computation (without noise)
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
                'n_samples': None,
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
            
            # WITH NOISE ANALYSIS
            if use_noise:
                for n_samples in n_noise_samples_list:
                    (lipschitz_noise, numerator_noise,
                    orig_tok_id_noise, pert_tok_id_noise, 
                    orig_p_noise, pert_p_noise,
                    logit_diff_noise, 
                    logits_orig_noise, logits_pert_noise,
                    rep1_hidden_tensor, rep2_hidden_tensor)  = compute_lipschitz_along_direction_with_noise(
                        model, full_embeddings, last_token_idx, direction, epsilon,
                        n_samples=n_samples
                    )
                    
                    # Decode tokens
                    orig_token_str_noise = tokenizer.decode([orig_tok_id_noise])
                    pert_token_str_noise = tokenizer.decode([pert_tok_id_noise])
                    token_changed_noise = (orig_tok_id_noise != pert_tok_id_noise)
                    
                    # Get the logit values for the predicted tokens
                    orig_logit_val_noise = logits_orig_noise[orig_tok_id_noise].item()
                    pert_logit_val_noise = logits_pert_noise[pert_tok_id_noise].item()
                    
                    results_topk.append({
                        'analysis_type': 'top_k_singular_with_noise',
                        'dimension': k,
                        'singular_value': S[k-1].item(),
                        'epsilon_power': power,
                        'epsilon': epsilon,
                        'n_samples': n_samples,
                        'lipschitz_constant': lipschitz_noise,
                        'output_diff_norm': numerator_noise,
                        'logit_diff': logit_diff_noise,
                        'orig_token_id': orig_tok_id_noise,
                        'orig_token': orig_token_str_noise,
                        'orig_prob': orig_p_noise,
                        'orig_logit': orig_logit_val_noise,
                        'pert_token_id': pert_tok_id_noise,
                        'pert_token': pert_token_str_noise,
                        'pert_prob': pert_p_noise,
                        'pert_logit': pert_logit_val_noise,
                        'token_changed': token_changed_noise
                    })
                    
                    if token_changed_noise:
                        print(f"    + noise(n={n_samples}): Lipschitz={lipschitz_noise:.6f}, ||Δf||={numerator_noise:.6e}, logit: {orig_logit_val_noise:.4f}→{pert_logit_val_noise:.4f}, ||Δlogits||={logit_diff_noise:.6e} | '{orig_token_str_noise}' (p={orig_p_noise:.4f}) → '{pert_token_str_noise}' (p={pert_p_noise:.4f}) ✓")
                    else:
                        print(f"    + noise(n={n_samples}): Lipschitz={lipschitz_noise:.6f}, ||Δf||={numerator_noise:.6e}, logit: {orig_logit_val_noise:.4f}→{pert_logit_val_noise:.4f}, ||Δlogits||={logit_diff_noise:.6e} | '{orig_token_str_noise}' (p={orig_p_noise:.4f}→{pert_p_noise:.4f})")
    
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
            
            # Original computation (without noise)
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
                'n_samples': None,
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
            
            # WITH NOISE ANALYSIS
            if use_noise:
                for n_samples in n_noise_samples_list:
                    (lipschitz_noise, numerator_noise,
                    orig_tok_id_noise, pert_tok_id_noise, 
                    orig_p_noise, pert_p_noise,
                    logit_diff_noise, 
                    logits_orig_noise, logits_pert_noise,
                    rep1_hidden_tensor, rep2_hidden_tensor) = compute_lipschitz_along_direction_with_noise(
                        model, full_embeddings, last_token_idx, direction, epsilon,
                        n_samples=n_samples
                    )

                    save_dir = output_dir if output_dir is not None else "hidden_states_output"
                    os.makedirs(save_dir, exist_ok=True)

                    # Generate timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Create filename with timestamp and n_samples
                    filename_prefix = f"dim{k if 'k' in locals() else dim_idx}_eps{power}_nsamples{n_samples}_time{timestamp}"

                    rep1_path = os.path.join(save_dir, f"{filename_prefix}_rep1_hidden.npy")
                    rep2_path = os.path.join(save_dir, f"{filename_prefix}_rep2_hidden.npy")

                    np.save(rep1_path, rep1_hidden_tensor.numpy())
                    np.save(rep2_path, rep2_hidden_tensor.numpy())

                    print(f"  Saved rep1_hidden_states to {rep1_path} with shape {rep1_hidden_tensor.shape}")
                    print(f"  Saved rep2_hidden_states to {rep2_path} with shape {rep2_hidden_tensor.shape}")
                    
                    # Decode tokens
                    orig_token_str_noise = tokenizer.decode([orig_tok_id_noise])
                    pert_token_str_noise = tokenizer.decode([pert_tok_id_noise])
                    token_changed_noise = (orig_tok_id_noise != pert_tok_id_noise)
                    
                    # Get the logit values for the predicted tokens
                    orig_logit_val_noise = logits_orig_noise[orig_tok_id_noise].item()
                    pert_logit_val_noise = logits_pert_noise[pert_tok_id_noise].item()
                    
                    results_dims.append({
                        'analysis_type': 'specific_dimension_with_noise',
                        'dimension': dim_idx,
                        'singular_value': S[dim_idx-1].item(),
                        'epsilon_power': power,
                        'epsilon': epsilon,
                        'n_samples': n_samples,
                        'lipschitz_constant': lipschitz_noise,
                        'output_diff_norm': numerator_noise,
                        'logit_diff': logit_diff_noise,
                        'orig_token_id': orig_tok_id_noise,
                        'orig_token': orig_token_str_noise,
                        'orig_prob': orig_p_noise,
                        'orig_logit': orig_logit_val_noise,
                        'pert_token_id': pert_tok_id_noise,
                        'pert_token': pert_token_str_noise,
                        'pert_prob': pert_p_noise,
                        'pert_logit': pert_logit_val_noise,
                        'token_changed': token_changed_noise
                    })
                    
                    if token_changed_noise:
                        print(f"    + noise(n={n_samples}): Lipschitz={lipschitz_noise:.6f}, ||Δf||={numerator_noise:.6e}, logit: {orig_logit_val_noise:.4f}→{pert_logit_val_noise:.4f}, ||Δlogits||={logit_diff_noise:.6e} | '{orig_token_str_noise}' (p={orig_p_noise:.4f}) → '{pert_token_str_noise}' (p={pert_p_noise:.4f}) ✓")
                    else:
                        print(f"    + noise(n={n_samples}): Lipschitz={lipschitz_noise:.6f}, ||Δf||={numerator_noise:.6e}, logit: {orig_logit_val_noise:.4f}→{pert_logit_val_noise:.4f}, ||Δlogits||={logit_diff_noise:.6e} | '{orig_token_str_noise}' (p={orig_p_noise:.4f}→{pert_p_noise:.4f})")
    
    # Combine all results
    all_results = results_topk + results_dims
    
    return pd.DataFrame(all_results), jacobian, U, S, Vt

# -----------------------------
# Run analysis
# -----------------------------
if __name__ == "__main__":
    print("Loading LLaMA 3.1 model...")
    model, tokenizer = load_llama_model()

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp2_v2_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Saving results to: {exp_dir}")

    # Test with different inputs
    test_inputs = [
        "The capital of France is",
        # "Machine learning is a subset of",
        # "To solve this problem, we need to"
    ]
    
    all_results = {}
    epsilon_powers = list(range(-7, -11, -1)) 
    analyze_dims = range(2)  # [50, 100, 200, 500, 1000, 2000, 4000, 4096]
    n_noise_samples_list = [2000]  # List of sample sizes to test
    
    for i, input_text in enumerate(test_inputs):
        print("\n" + "="*80)
        print(f"TEST CASE {i+1}: '{input_text}'")
        print("="*80)
        
        results_df, jacobian, U, S, Vt = analyze_lipschitz_constants(
            model, tokenizer, input_text,
            top_k=5,
            epsilon_powers=epsilon_powers,
            analyze_dims=analyze_dims,
            use_noise=True,
            n_noise_samples_list=n_noise_samples_list,
            output_dir=exp_dir
        )
        
        all_results[f"test_case_{i+1}"] = {
            'results': results_df,
            'jacobian': jacobian,
            'U': U,
            'S': S,
            'Vt': Vt
        }
        
        # Save results
        filename = f"lipschitz_analysis_test_{i+1}_with_noise.csv"
        results_df.to_csv(os.path.join(exp_dir, filename), index=False)
        print(f"\nSaved results to {os.path.join(exp_dir, filename)}")
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    for test_name, data in all_results.items():
        df = data['results']
        print(f"\n{test_name}:")
        print(f"  Top 5 singular values: {data['S'][:5].cpu().numpy()}")
        
        # Summary for top-k analysis WITHOUT noise
        topk_df = df[df['analysis_type'] == 'top_k_singular']
        if len(topk_df) > 0:
            print(f"\n  Top-K Singular Directions Analysis (WITHOUT noise):")
            for k in range(1, 6):
                subset = topk_df[topk_df['dimension'] == k]
                if len(subset) > 0:
                    print(f"    Direction {k} (σ={data['S'][k-1].item():.4f}):")
                    print(f"      Lipschitz range: [{subset['lipschitz_constant'].min():.6f}, {subset['lipschitz_constant'].max():.6f}]")
                    print(f"      Mean Lipschitz: {subset['lipschitz_constant'].mean():.6f}")
        
        # Summary for top-k analysis WITH noise
        topk_noise_df = df[df['analysis_type'] == 'top_k_singular_with_noise']
        if len(topk_noise_df) > 0:
            print(f"\n  Top-K Singular Directions Analysis (WITH noise):")
            for k in range(1, 6):
                subset = topk_noise_df[topk_noise_df['dimension'] == k]
                if len(subset) > 0:
                    print(f"    Direction {k} (σ={data['S'][k-1].item():.4f}):")
                    for n_samples in sorted(subset['n_samples'].unique()):
                        noise_subset = subset[subset['n_samples'] == n_samples]
                        print(f"      n_samples={n_samples}:")
                        print(f"        Lipschitz range: [{noise_subset['lipschitz_constant'].min():.6f}, {noise_subset['lipschitz_constant'].max():.6f}]")
                        print(f"        Mean Lipschitz: {noise_subset['lipschitz_constant'].mean():.6f}")
        
        # Summary for specific dimensions analysis WITHOUT noise
        dims_df = df[df['analysis_type'] == 'specific_dimension']
        if len(dims_df) > 0:
            print(f"\n  Specific Dimensions Analysis (WITHOUT noise):")
            for dim in analyze_dims:
                subset = dims_df[dims_df['dimension'] == dim]
                if len(subset) > 0:
                    print(f"    Dimension {dim} (σ={subset['singular_value'].iloc[0]:.6f}):")
                    print(f"      Lipschitz range: [{subset['lipschitz_constant'].min():.6f}, {subset['lipschitz_constant'].max():.6f}]")
                    print(f"      Mean Lipschitz: {subset['lipschitz_constant'].mean():.6f}")
        
        # Summary for specific dimensions analysis WITH noise
        dims_noise_df = df[df['analysis_type'] == 'specific_dimension_with_noise']
        if len(dims_noise_df) > 0:
            print(f"\n  Specific Dimensions Analysis (WITH noise):")
            for dim in analyze_dims:
                subset = dims_noise_df[dims_noise_df['dimension'] == dim]
                if len(subset) > 0:
                    print(f"    Dimension {dim} (σ={subset['singular_value'].iloc[0]:.6f}):")
                    for n_samples in sorted(subset['n_samples'].unique()):
                        noise_subset = subset[subset['n_samples'] == n_samples]
                        print(f"      n_samples={n_samples}:")
                        print(f"        Lipschitz range: [{noise_subset['lipschitz_constant'].min():.6f}, {noise_subset['lipschitz_constant'].max():.6f}]")
                        print(f"        Mean Lipschitz: {noise_subset['lipschitz_constant'].mean():.6f}")
    
    # Additional analysis: Compare noise vs no-noise
    print("\n" + "="*80)
    print("NOISE IMPACT ANALYSIS")
    print("="*80)
    
    for test_name, data in all_results.items():
        df = data['results']
        print(f"\n{test_name}:")
        
        # Compare for each epsilon power
        for power in epsilon_powers:
            epsilon = 10.0 ** power
            
            # Get results without noise
            no_noise = df[(df['analysis_type'].isin(['top_k_singular', 'specific_dimension'])) & 
                         (df['epsilon_power'] == power)]
            
            # Get results with noise
            with_noise = df[(df['analysis_type'].isin(['top_k_singular_with_noise', 'specific_dimension_with_noise'])) & 
                           (df['epsilon_power'] == power)]
            
            if len(no_noise) > 0 and len(with_noise) > 0:
                mean_lipschitz_no_noise = no_noise['lipschitz_constant'].mean()
                
                print(f"  ε=10^{power:3d}:")
                print(f"    Without noise: Mean Lipschitz = {mean_lipschitz_no_noise:.6f}")
                
                # Break down by n_samples
                for n_samples in sorted(with_noise['n_samples'].unique()):
                    noise_subset = with_noise[with_noise['n_samples'] == n_samples]
                    mean_lipschitz_with_noise = noise_subset['lipschitz_constant'].mean()
                    print(f"    With noise (n={n_samples}): Mean Lipschitz = {mean_lipschitz_with_noise:.6f}")
                    print(f"      Difference: {abs(mean_lipschitz_with_noise - mean_lipschitz_no_noise):.6f} ({100*abs(mean_lipschitz_with_noise - mean_lipschitz_no_noise)/mean_lipschitz_no_noise:.2f}%)")
    
    # Token change analysis
    print("\n" + "="*80)
    print("TOKEN CHANGE ANALYSIS")
    print("="*80)
    
    for test_name, data in all_results.items():
        df = data['results']
        print(f"\n{test_name}:")
        
        # Analyze token changes without noise
        no_noise_df = df[df['analysis_type'].isin(['top_k_singular', 'specific_dimension'])]
        if len(no_noise_df) > 0:
            total = len(no_noise_df)
            changed = no_noise_df['token_changed'].sum()
            print(f"  WITHOUT noise: {changed}/{total} ({100*changed/total:.2f}%) token changes")
        
        # Analyze token changes with noise
        with_noise_df = df[df['analysis_type'].isin(['top_k_singular_with_noise', 'specific_dimension_with_noise'])]
        if len(with_noise_df) > 0:
            total = len(with_noise_df)
            changed = with_noise_df['token_changed'].sum()
            print(f"  WITH noise:    {changed}/{total} ({100*changed/total:.2f}%) token changes")
            
            # Breakdown by n_samples
            for n_samples in sorted(with_noise_df['n_samples'].unique()):
                subset = with_noise_df[with_noise_df['n_samples'] == n_samples]
                total_subset = len(subset)
                changed_subset = subset['token_changed'].sum()
                print(f"    n_samples={n_samples}: {changed_subset}/{total_subset} ({100*changed_subset/total_subset:.2f}%) token changes")
    
    print("\nDone!")

'''
nohup python3 src/experiment2_v2_avg_lipschitz.py  > exp2_avg_lipschitz_v2_2000_.txt 2>&1 &
'''