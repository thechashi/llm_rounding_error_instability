"""
Experiment 3: Float32 Precision Analysis

This experiment specifically focuses on analyzing model behavior and Lipschitz
constants using float32 precision, potentially comparing against other precision
modes (bfloat16, float16) used in earlier experiments.

Purpose:
--------
Tests whether using float32 precision (vs bfloat16 or float16) affects:
1. Computed Lipschitz constants
2. Jacobian numerical stability
3. SVD singular value accuracy
4. Overall instability measurements


Relationship:
-------------
This experiment ISOLATES THE EFFECT OF NUMERICAL PRECISION:
- Most previous experiments (lipschitz, experiment1-2) use float32 by default
- This experiment may explicitly test or validate float32 behavior
- Results establish baseline for precision comparison

The relationship to experiment4 is interesting:
- experiment3: Tests different PRECISION modes (float32 vs others)
- experiment4: Tests different GPU HARDWARE (same precision, different compute)
- Together they separate software precision from hardware computation effects

Methodology:
------------
Similar to experiment2, but with explicit focus on float32:
1. Load Llama model in FLOAT32 precision
2. Compute Jacobian in float32
3. Perform SVD in float32
4. Extract singular values
5. Compare results (potentially with other precision modes if included)
6. Document any numerical differences or stability improvements

Hypotheses to Test:
-------------------
1. Does float32 reduce numerical errors in Jacobian computation?
2. Do Lipschitz constants differ significantly between precisions?
3. Is SVD more stable in float32?
4. Does higher precision reduce observed instability?

Use Case:
---------
Use this experiment to:
- Establish float32 as baseline precision for comparisons
- Validate that float32 provides sufficient numerical precision
- Understand if instability is due to model architecture (intrinsic) or
  numerical precision (artifact)
- Determine if using lower precision (bfloat16) introduces additional instability

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, pandas
- Llama-3.1-8B-Instruct model
- EXPLICITLY uses torch.float32 dtype

Key Functions:
--------------
- load_llama_model(): Load model in float32
- model_forward_last_hidden(): Forward pass for Jacobian (float32)
- compute_jacobian(): Compute Jacobian in float32 precision
- perform_svd(): SVD in float32

Output:
-------
- Timestamped results directory (results/exp3_YYYY-MM-DD_HH-MM-SS/)
- Lipschitz constants computed in float32
- Comparison with other precision modes (if included)
- Numerical stability analysis

Note:
-----
This experiment helps separate intrinsic model instability from numerical
precision artifacts, which is crucial for understanding the root causes of
the rounding error instability phenomenon.
"""

import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

def load_llama_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

def model_forward_last_hidden(model, flattened_last_token_embedding, original_shape, full_embeddings, last_token_idx):
    last_token_embedding = flattened_last_token_embedding.view(original_shape)
    modified_embeddings = full_embeddings.clone()
    modified_embeddings[0, last_token_idx, :] = last_token_embedding
    outputs = model(inputs_embeds=modified_embeddings, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
    return last_hidden_state

def compute_jacobian(model, full_embeddings, last_token_idx):
    device = next(model.parameters()).device
    last_token_embedding = full_embeddings[0, last_token_idx, :].clone()
    original_shape = last_token_embedding.shape
    flattened_embedding = last_token_embedding.flatten()
    flattened_embedding = flattened_embedding.detach().requires_grad_(True)
    
    from functools import partial
    forward_fn = partial(
        model_forward_last_hidden,
        model,
        original_shape=original_shape,
        full_embeddings=full_embeddings.detach(),
        last_token_idx=last_token_idx
    )
    
    jacobian = torch.autograd.functional.jacobian(
        forward_fn,
        flattened_embedding,
        vectorize=True
    )
    
    return jacobian, last_token_embedding

def perform_svd(jacobian):
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    return U, S, Vt

# Create experiment directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join("../results", f"exp3_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)
print(f"Results will be saved to: {exp_dir}")

print("Loading LLaMA 3.1 model...")
model, tokenizer = load_llama_model()

input_text = "The capital of France is"
device = next(model.parameters()).device

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    full_embeddings = model.model.embed_tokens(inputs["input_ids"])

last_token_idx = inputs["input_ids"].shape[1] - 1

print("Computing Jacobian...")
jacobian, last_token_embedding = compute_jacobian(model, full_embeddings, last_token_idx)

print("Performing SVD...")
U, S, Vt = perform_svd(jacobian)

print(f"Original embedding shape: {last_token_embedding.shape}")
print(f"Top 5 singular values: {S[:5].cpu().numpy()}")

# Get first singular direction
direction = Vt[0, :]  # Shape: [4096]
singular_value = S[0].item()

print(f"\nUsing Direction 1 with singular value: {singular_value:.6f}")

# Original embedding values
original_embedding = last_token_embedding.cpu().numpy()

# Epsilon powers to test
epsilon_powers = [-5, -6, -7, -8, -9, -10, -11, -12]

# Create DataFrame
data = {'dim_index': np.arange(4096)}
data['original_value'] = original_embedding

direction_np = direction.cpu().numpy()

for power in epsilon_powers:
    epsilon = 10.0 ** power
    
    # Perturbation: epsilon * singular_value * direction
    perturbation = epsilon * singular_value * direction_np
    
    # New value: original + perturbation
    new_value = original_embedding + perturbation
    
    # Difference
    difference = abs(new_value - original_embedding)
    
    # Add columns
    data[f'eps_1e{power}_perturbation'] = perturbation
    data[f'eps_1e{power}_new_value'] = new_value
    data[f'eps_1e{power}_difference'] = difference

df = pd.DataFrame(data)

# Save with maximum precision
output_file = os.path.join(exp_dir, 'embedding_perturbation_analysis.csv')
df.to_csv(output_file, index=False, float_format='%.15e')

print(f"\nSaved to {output_file}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: dim_index, original_value, then for each epsilon: perturbation, new_value, difference")
print(f"\nFirst few rows of original values:")
print(df[['dim_index', 'original_value']].head())

# Summary statistics - check at multiple precision levels
print("\n" + "="*80)
print("SUMMARY: Testing at different precision levels")
print("="*80)

precision_levels = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

for power in epsilon_powers:
    diff_col = f'eps_1e{power}_difference'
    max_diff = df[diff_col].abs().max()
    mean_diff = df[diff_col].abs().mean()
    
    print(f"\nε = 10^{power}:")
    print(f"  Max absolute difference: {max_diff:.15e}")
    print(f"  Mean absolute difference: {mean_diff:.15e}")
    
    # Check at each precision level
    print(f"  Dimensions changed at different precisions:")
    for precision in precision_levels:
        original_rounded = np.round(df['original_value'].values, precision)
        new_rounded = np.round(df[f'eps_1e{power}_new_value'].values, precision)
        num_changed = np.sum(original_rounded != new_rounded)
        
        if num_changed > 0:
            print(f"    {precision} decimals: {num_changed}/{4096} ({100*num_changed/4096:.2f}%)")
        else:
            print(f"    {precision} decimals: 0 changes (precision limit reached)")
            break

print("\nDone!")