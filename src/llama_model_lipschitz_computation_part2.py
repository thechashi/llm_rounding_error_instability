"""
Lipschitz Constant Computation for Llama Models (Part 2)

This script appears to be NEARLY IDENTICAL to lipschitz_const_llama.py,
potentially serving as an extension, backup, or variant implementation.

Purpose:
--------
Computes the Lipschitz constant of the Llama model's transformation from input
embeddings to final hidden states using Jacobian analysis and SVD, using the
same methodology as the original lipschitz_const_llama.py.


Relationship:
-------------
This file is a NEAR-DUPLICATE of lipschitz_const_llama.py. The "part2" naming
suggests it may be:
1. An extension with additional analysis (check the full file for differences)
2. A backup copy for iterative development
3. A variant for testing alternative approaches

Both files provide the same core functionality:
- Jacobian computation via torch.autograd.functional.jacobian
- SVD-based singular value extraction
- Lipschitz constant = largest singular value

The methodology from both files is foundational for:
- experiment2 series: Average Lipschitz analysis
- experiment5: Layer-wise Lipschitz analysis
- experiment6-7: Jacobian-based sensitivity analysis

Theoretical Background:
-----------------------
The Lipschitz constant L quantifies the maximum rate of change:
  ||f(x + Δx) - f(x)|| ≤ L * ||Δx||

For neural networks, L = largest singular value of the Jacobian matrix.

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
Same as lipschitz_const_llama.py:
- Compute theoretical upper bound on output sensitivity
- Understand perturbation propagation through the model
- Validate empirical instability measurements

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, pandas
- Llama-3.1-8B-Instruct model
- Uses float32 for numerical precision

Key Functions:
--------------
- load_llama_model(): Load model in float32 precision
- model_forward_last_hidden(): Forward pass for Jacobian computation
- compute_jacobian(): Compute Jacobian matrix using torch.autograd
- perform_svd(): SVD analysis to extract singular values
- test_lipschitz_constant(): Full pipeline for Lipschitz constant computation

Note:
-----
Check if this file differs from lipschitz_const_llama.py in implementation
details or if it represents iterative development. Consider consolidating if
they are truly identical.
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
# Get hidden states for unperturbed and perturbed inputs
# -----------------------------
def get_hidden_states_and_logits(model, full_embeddings, last_token_idx):
    """
    Get the hidden state and logits for a given input.
    
    Returns:
        hidden_state: Last token's final hidden state [hidden_dim]
        logits: Logits for prediction [vocab_size]
    """
    with torch.no_grad():
        outputs = model(inputs_embeds=full_embeddings, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
        logits = outputs.logits[0, last_token_idx, :]
    
    return hidden_state, logits

# -----------------------------
# Analyze rounding errors with singular direction 1
# -----------------------------
def analyze_rounding_errors(model, tokenizer, input_text, epsilon=1e-7, output_dir="./rounding_error_analysis"):
    """
    Analyze potential rounding errors by saving hidden states for unperturbed and perturbed inputs.
    Uses singular direction 1 with specified epsilon.
    
    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        input_text: Input text
        epsilon: Perturbation magnitude (default 10^-7)
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"Epsilon: {epsilon}")
    print(f"Data type: {full_embeddings.dtype}")
    
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
    
    # Get singular direction 1 (first right singular vector)
    direction_1 = Vt[0, :]  # Shape: [hidden_dim]
    singular_value_1 = S[0].item()
    
    print(f"\n" + "="*80)
    print(f"ANALYZING ROUNDING ERRORS")
    print(f"Using Singular Direction 1 (σ={singular_value_1:.6f}) with ε={epsilon}")
    print("="*80)
    
    # Get f(x) - unperturbed hidden state and logits
    print("\n1. Computing f(x) - unperturbed output...")
    fx_hidden, fx_logits = get_hidden_states_and_logits(model, full_embeddings, last_token_idx)
    
    # Get predicted token for unperturbed
    fx_token_id = torch.argmax(fx_logits).item()
    fx_token = tokenizer.decode([fx_token_id])
    fx_prob = F.softmax(fx_logits, dim=-1)[fx_token_id].item()
    fx_logit_val = fx_logits[fx_token_id].item()
    
    print(f"   Predicted token: '{fx_token}' (id={fx_token_id})")
    print(f"   Probability: {fx_prob:.6f}")
    print(f"   Logit value: {fx_logit_val:.6f}")
    print(f"   Hidden state shape: {fx_hidden.shape}")
    print(f"   Hidden state norm: {torch.norm(fx_hidden).item():.6f}")
    
    # Perturb input: x + ε·v₁
    print(f"\n2. Computing f(x + ε·v₁) - perturbed output...")
    perturbed_embeddings = full_embeddings.clone()
    perturbed_embeddings[0, last_token_idx, :] += epsilon * direction_1
    
    # Get f(x + ε·v₁) - perturbed hidden state and logits
    fx_pert_hidden, fx_pert_logits = get_hidden_states_and_logits(model, perturbed_embeddings, last_token_idx)
    
    # Get predicted token for perturbed
    fx_pert_token_id = torch.argmax(fx_pert_logits).item()
    fx_pert_token = tokenizer.decode([fx_pert_token_id])
    fx_pert_prob = F.softmax(fx_pert_logits, dim=-1)[fx_pert_token_id].item()
    fx_pert_logit_val = fx_pert_logits[fx_pert_token_id].item()
    
    print(f"   Predicted token: '{fx_pert_token}' (id={fx_pert_token_id})")
    print(f"   Probability: {fx_pert_prob:.6f}")
    print(f"   Logit value: {fx_pert_logit_val:.6f}")
    print(f"   Hidden state shape: {fx_pert_hidden.shape}")
    print(f"   Hidden state norm: {torch.norm(fx_pert_hidden).item():.6f}")
    
    # Compute differences
    print("\n3. Computing differences...")
    hidden_diff = fx_pert_hidden - fx_hidden
    logit_diff = fx_pert_logits - fx_logits
    hidden_diff_norm = torch.norm(hidden_diff).item()
    logit_diff_norm = torch.norm(logit_diff).item()
    lipschitz = hidden_diff_norm / epsilon
    
    print(f"   ||f(x+ε·v₁) - f(x)||: {hidden_diff_norm:.6e}")
    print(f"   Lipschitz constant: {lipschitz:.6f}")
    print(f"   ||logits(x+ε·v₁) - logits(x)||: {logit_diff_norm:.6e}")
    print(f"   Token changed: {fx_token_id != fx_pert_token_id}")
    
    # Save to text file
    print("\n4. Saving results...")
    text_output_path = os.path.join(output_dir, "rounding_error_analysis.txt")
    with open(text_output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROUNDING ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input text: '{input_text}'\n")
        f.write(f"Sequence length: {inputs['input_ids'].shape[1]}\n")
        f.write(f"Last token index: {last_token_idx}\n")
        f.write(f"Data type: {full_embeddings.dtype}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Singular value 1: {singular_value_1:.6f}\n\n")
        
        f.write("UNPERTURBED OUTPUT f(x):\n")
        f.write(f"  Predicted token: '{fx_token}' (id={fx_token_id})\n")
        f.write(f"  Probability: {fx_prob:.6f}\n")
        f.write(f"  Logit value: {fx_logit_val:.6f}\n")
        f.write(f"  Hidden state norm: {torch.norm(fx_hidden).item():.6f}\n\n")
        
        f.write("PERTURBED OUTPUT f(x + ε·v₁):\n")
        f.write(f"  Predicted token: '{fx_pert_token}' (id={fx_pert_token_id})\n")
        f.write(f"  Probability: {fx_pert_prob:.6f}\n")
        f.write(f"  Logit value: {fx_pert_logit_val:.6f}\n")
        f.write(f"  Hidden state norm: {torch.norm(fx_pert_hidden).item():.6f}\n\n")
        
        f.write("DIFFERENCES:\n")
        f.write(f"  ||f(x+ε·v₁) - f(x)||: {hidden_diff_norm:.6e}\n")
        f.write(f"  Lipschitz constant: {lipschitz:.6f}\n")
        f.write(f"  ||logits(x+ε·v₁) - logits(x)||: {logit_diff_norm:.6e}\n")
        f.write(f"  Token changed: {fx_token_id != fx_pert_token_id}\n")
    
    print(f"   Saved text output to: {text_output_path}")
    
    # Save vectors to NumPy files
    # Convert to CPU and numpy
    fx_hidden_np = fx_hidden.cpu().numpy()
    fx_pert_hidden_np = fx_pert_hidden.cpu().numpy()
    hidden_diff_np = hidden_diff.cpu().numpy()
    fx_logits_np = fx_logits.cpu().numpy()
    fx_pert_logits_np = fx_pert_logits.cpu().numpy()
    logit_diff_np = logit_diff.cpu().numpy()
    direction_1_np = direction_1.cpu().numpy()
    
    # Save individual vectors
    np.save(os.path.join(output_dir, "fx_hidden_state.npy"), fx_hidden_np)
    np.save(os.path.join(output_dir, "fx_pert_hidden_state.npy"), fx_pert_hidden_np)
    np.save(os.path.join(output_dir, "hidden_state_diff.npy"), hidden_diff_np)
    np.save(os.path.join(output_dir, "fx_logits.npy"), fx_logits_np)
    np.save(os.path.join(output_dir, "fx_pert_logits.npy"), fx_pert_logits_np)
    np.save(os.path.join(output_dir, "logit_diff.npy"), logit_diff_np)
    np.save(os.path.join(output_dir, "singular_direction_1.npy"), direction_1_np)
    
    # Save all data in a single .npz file for convenience
    np.savez(
        os.path.join(output_dir, "rounding_error_data.npz"),
        fx_hidden_state=fx_hidden_np,
        fx_pert_hidden_state=fx_pert_hidden_np,
        hidden_state_diff=hidden_diff_np,
        fx_logits=fx_logits_np,
        fx_pert_logits=fx_pert_logits_np,
        logit_diff=logit_diff_np,
        singular_direction_1=direction_1_np,
        epsilon=epsilon,
        singular_value_1=singular_value_1,
        hidden_diff_norm=hidden_diff_norm,
        logit_diff_norm=logit_diff_norm,
        lipschitz_constant=lipschitz
    )
    
    print(f"   Saved NumPy arrays:")
    print(f"     - fx_hidden_state.npy")
    print(f"     - fx_pert_hidden_state.npy")
    print(f"     - hidden_state_diff.npy")
    print(f"     - fx_logits.npy")
    print(f"     - fx_pert_logits.npy")
    print(f"     - logit_diff.npy")
    print(f"     - singular_direction_1.npy")
    print(f"     - rounding_error_data.npz (all data in one file)")
    
    # Create summary
    summary = {
        'input_text': input_text,
        'epsilon': epsilon,
        'singular_value_1': singular_value_1,
        'data_type': str(full_embeddings.dtype),
        'unperturbed': {
            'token': fx_token,
            'token_id': fx_token_id,
            'probability': fx_prob,
            'logit_value': fx_logit_val,
            'hidden_norm': torch.norm(fx_hidden).item()
        },
        'perturbed': {
            'token': fx_pert_token,
            'token_id': fx_pert_token_id,
            'probability': fx_pert_prob,
            'logit_value': fx_pert_logit_val,
            'hidden_norm': torch.norm(fx_pert_hidden).item()
        },
        'differences': {
            'hidden_diff_norm': hidden_diff_norm,
            'lipschitz_constant': lipschitz,
            'logit_diff_norm': logit_diff_norm,
            'token_changed': bool(fx_token_id != fx_pert_token_id)
        }
    }
    
    return summary

# -----------------------------
# Run analysis
# -----------------------------
if __name__ == "__main__":
    print("Loading LLaMA 3.1 model...")
    model, tokenizer = load_llama_model()
    
    # Test input
    input_text = "The capital of France is"
    
    print("\n" + "="*80)
    print("ROUNDING ERROR ANALYSIS")
    print("Singular Direction 1, Epsilon = 10^-7")
    print("="*80)
    
    summary = analyze_rounding_errors(
        model, 
        tokenizer, 
        input_text,
        epsilon=1e-7,
        output_dir="./rounding_error_analysis"
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll files saved to: ./rounding_error_analysis/")
    print("\nTo load in MATLAB:")
    print("  data = load('rounding_error_data.npz');")
    print("  fx = data.fx_hidden_state;")
    print("  fx_pert = data.fx_pert_hidden_state;")
    print("  diff = data.hidden_state_diff;")
    
    print("\nDone!")