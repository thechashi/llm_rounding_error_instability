"""
Experiment 11: Singular Vector Rotation Analysis for Layer 0

This experiment investigates the effect of rotating the perturbation direction (a singular
vector) on the resulting changes in the first layer's hidden state.

Purpose:
--------
Based on the observation from `exp1` that a small perturbation along the primary
singular vector causes a change in only a single index of the Layer 0
representation, this experiment explores whether that phenomenon is location-
specific. By cyclically shifting the singular vector, we can analyze how the
location and number of changed indices in Layer 0 are affected.

Methodology:
------------
1.  Load the Llama model and compute the Jacobian SVD for the full model, identical
    to `exp1`.
2.  Select the primary singular vector (corresponding to the largest singular value).
3.  Define two close perturbation magnitudes, `e1` and `e2`, based on the values
    from `exp1_output.log` that produced a single change.
4.  Iterate through a range of rotation shifts `n` (e.g., from 0 to 4095).
5.  In each iteration:
    a. Rotate the singular vector by `n` elements using `torch.roll()`.
    b. Apply perturbations `e1` and `e2` to the input embedding using this
       *rotated* vector as the direction.
    c. Pass the two perturbed embeddings through the model.
    d. Extract the hidden state for Layer 0 for both outputs. Layer 0 here refers
       to the first set of hidden states returned by the model, which corresponds
       to the output of the initial embedding layer.
    e. Compare the two Layer 0 hidden states to find the number and indices of
       elements that have changed.
    f. Print the rotation shift `n`, the number of changes, and the specific
       indices that changed.

Use Case:
---------
Use this experiment to understand if the single-index-change sensitivity is tied
to specific dimensions of the embedding or if it's an artifact of the perturbation
direction's overall structure. It helps diagnose the localized impact of input
perturbations.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
import json

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Loads the model and tokenizer in float32."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32
    )
    return model, tokenizer

def compute_jacobian_svd(model, embeddings, last_token_idx):
    """Computes the Jacobian SVD for the full model."""
    def forward_fn(flat_emb):
        emb = flat_emb.view(1, -1)
        mod_emb = embeddings.clone()
        mod_emb[0, last_token_idx, :] = emb
        outputs = model(inputs_embeds=mod_emb, output_hidden_states=True)
        return outputs.hidden_states[-1][0, last_token_idx, :]
    
    last_emb = embeddings[0, last_token_idx, :].clone().detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(forward_fn, last_emb, vectorize=True)
    _, _, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    return Vt

def get_hidden_states(model, embeddings, last_token_idx):
    """
    Get hidden states from Layer 0 (embeddings) and Layer 1 (first transformer block).
    """
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        # hidden_states[0] is the embedding output
        # hidden_states[1] is the output of the first transformer layer
        layer0_state = outputs.hidden_states[0][0, last_token_idx, :].cpu().numpy()
        layer1_state = outputs.hidden_states[1][0, last_token_idx, :].cpu().numpy()
        return layer0_state, layer1_state

def analyze_singular_vector_rotation(
    e1,
    step_size,
    jumps,
    singular_idx=0,
    text="The capital of France is",
    threshold=0,
    max_rotations=None,
    rotation_step_size=1,
    output_path="exp11_results.json",
    use_float64_perturbation=True
):
    """
    Main function to run the singular vector rotation analysis and save results to JSON.

    Args:
        e1 (float): Base perturbation magnitude.
        step_size (float): The smallest step to add to the perturbation.
        jumps (list): List of multipliers for the step_size to test.
        singular_idx (int): Index of the singular vector to use.
        text (str): Input text for the model.
        threshold (float): Threshold to consider a value as "changed".
        max_rotations (int, optional): Limit rotations for a quicker test.
        rotation_step_size (int): The step size for the rotation loop.
        output_path (str): Path to save the JSON results.
        use_float64_perturbation (bool): If True, perform perturbation in float64.
    """
    print("="*80)
    print("Experiment 11: Singular Vector Rotation Analysis")
    print(f"Base perturbation e1: {e1:.15e}")
    print(f"Step size: {step_size:.15e}")
    print(f"Jumps to test: {jumps}")
    print(f"Singular vector index: {singular_idx}")
    print(f"Max rotations to test: {max_rotations or 4096}")
    print(f"Rotation step size: {rotation_step_size}")
    print(f"Use float64 for perturbation: {use_float64_perturbation}")
    print(f"Output will be saved to: {output_path}")
    print("="*80)

    # 1. Load model and tokenizer
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    
    # 2. Prepare input
    print("[2/5] Tokenizing and preparing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()
    
    # 3. Compute SVD and get the direction vector
    print("[3/5] Computing Jacobian SVD...")
    Vt = compute_jacobian_svd(model, embeddings, last_idx)
    direction = Vt[singular_idx, :].clone()
    
    # Determine the number of rotations
    embedding_dim = direction.shape[0]
    num_rotations = max_rotations if max_rotations is not None else embedding_dim

    # List to store all results
    all_results = []

    # 4. Loop through specified jumps
    for jump in jumps:
        e2 = e1 + step_size * jump
        
        print("\n" + "="*80)
        print(f"Analyzing for jump = {jump}")
        print(f"Comparing perturbations: e1={e1:.15e}, e2={e2:.15e}")
        print("="*80)

        print(f"Starting analysis for {num_rotations} rotations...")
        
        jump_results = {
            "jump": jump,
            "e1": e1,
            "e2": e2,
            "rotations": []
        }

        # 5. Loop through rotations for the current jump
        for n in range(0, num_rotations, rotation_step_size):
            print(f"  Processing rotation {n}/{num_rotations}...", end='\r')
            
            # Rotate the direction vector
            rotated_direction = torch.roll(direction, shifts=n, dims=0)
            
            # Create perturbed embeddings
            if use_float64_perturbation:
                perturbed_emb1 = (original_input_emb.double() + e1 * rotated_direction.double()).float()
                perturbed_emb2 = (original_input_emb.double() + e2 * rotated_direction.double()).float()
            else:
                perturbed_emb1 = original_input_emb + e1 * rotated_direction
                perturbed_emb2 = original_input_emb + e2 * rotated_direction
            
            # Get hidden states for both perturbations
            embeddings1 = embeddings.clone()
            embeddings1[0, last_idx, :] = perturbed_emb1
            rep1_layer0, rep1_layer1 = get_hidden_states(model, embeddings1, last_idx)
            
            embeddings2 = embeddings.clone()
            embeddings2[0, last_idx, :] = perturbed_emb2
            rep2_layer0, rep2_layer1 = get_hidden_states(model, embeddings2, last_idx)
            
            # Compare Layer 0 (Embedding Output)
            diff0 = np.abs(rep1_layer0 - rep2_layer0)
            changed_indices0 = np.where(diff0 > threshold)[0]
            num_changed0 = len(changed_indices0)
            
            # Compare Layer 1 (Transformer Block 1 Output)
            diff1 = np.abs(rep1_layer1 - rep2_layer1)
            changed_indices1 = np.where(diff1 > threshold)[0]
            num_changed1 = len(changed_indices1)

            # Store results for this rotation
            rotation_data = {
                "rotation_shift": n,
                "layer0": {
                    "num_changes": num_changed0,
                    "changed_indices": changed_indices0.tolist()
                },
                "layer1": {
                    "num_changes": num_changed1,
                    "changed_indices": changed_indices1.tolist()
                }
            }
            jump_results["rotations"].append(rotation_data)

        all_results.append(jump_results)
        print(f"\nFinished analysis for jump {jump}.")

    # 6. Save results to a JSON file
    print(f"\n[4/5] Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n[5/5] Analysis complete.")
    print("="*80)


if __name__ == "__main__":
    # These values are taken from `exp1_layerwise_svd_perturbation_analysis.py`
    e1 = 1e-6 + 1815 * 2e-13
    step_size = 3 * 2e-14
    jumps_to_test = [1, 10, 20, 50, 100]

    # Run the analysis.
    # To run a full analysis (all 4096 rotations), set max_rotations=None.
    # For a quicker test, you can set it to a smaller number, e.g., 100.
    analyze_singular_vector_rotation(
        e1=e1,
        step_size=step_size,
        jumps=jumps_to_test,
        singular_idx=0,
        max_rotations=4096, # Set to a smaller number for a quick test
        rotation_step_size=64, # Jump k steps in rotation
        use_float64_perturbation=True
    )
