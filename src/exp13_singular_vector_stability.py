"""
Experiment 13: Singular Vector Stability Analysis

This experiment analyzes the stability of the Llama model by finding
the maximum perturbation magnitude along each singular vector direction that
doesn't change the output representation.

Purpose:
--------
For each singular vector, determine how far the input embedding can be perturbed
along that direction before the model's output representation changes.

Methodology:
------------
1. Load Llama model in float32.
2. Compute Jacobian SVD for full model (last token embedding → last hidden state).
3. For each singular vector (e_i) from Vt:
   a. Perturbation direction is e_i.
   b. Binary search for maximum s where output doesn't change.
   c. Perturbation: x_perturbed = x0 + s * e_i.
4. Generate a plot showing max_s for each singular vector.

Output:
-------
- Timestamped results directory (results/exp13_YYYY-MM-DD_HH-MM-SS/)
- Plot showing max_s for each singular vector.
- Data file (.npz and .csv) with singular vector indices and corresponding max_s, low, and high values.
- Detailed statistics about the max_s values across all singular vectors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
from tqdm import tqdm


def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Load model and tokenizer in float32"""
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
    """Compute Jacobian and SVD for the model"""
    def forward_fn(flat_emb):
        emb = flat_emb.view(1, -1)
        mod_emb = embeddings.clone()
        mod_emb[0, last_token_idx, :] = emb
        outputs = model(inputs_embeds=mod_emb, output_hidden_states=True)
        return outputs.hidden_states[-1][0, last_token_idx, :]

    last_emb = embeddings[0, last_token_idx, :].clone().detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(forward_fn, last_emb, vectorize=True)
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    return U, S, Vt


def get_hidden_state(model, embeddings, last_token_idx):
    """Get the final hidden state for the last token"""
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][0, last_token_idx, :].cpu().numpy()
    return hidden_state


def check_output_changed(original_hidden, perturbed_hidden, threshold=0):
    """
    Check if the output has changed between original and perturbed

    Args:
        original_hidden: Original hidden state
        perturbed_hidden: Perturbed hidden state
        threshold: Threshold for considering a change (default: 0 for exact comparison)

    Returns:
        True if changed, False if unchanged
    """
    diff = np.abs(original_hidden - perturbed_hidden)
    max_diff = np.max(diff)
    return max_diff > threshold


import numpy as np

def binary_search_max_s(
    model,
    original_embeddings,
    last_token_idx,
    direction,
    original_hidden,
    s_min=0.0,
    s_max=1e-6,
    tolerance=1e-15,
    max_iterations=100,
    threshold=0,
):
    """
    Finds the exact float32 max_s using:
    (1) coarse binary search
    (2) ULP-precise refinement via nextafter
    """

    device = next(model.parameters()).device
    original_emb = original_embeddings[0, last_token_idx, :].clone()

    # ---------- 1. Binary search (coarse) ----------
    low = s_min
    high = s_max
    best_s = s_min

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        mid = float(np.float32(mid))  # force float32 lattice

        perturbed_emb = original_emb + mid * direction
        embeddings_perturbed = original_embeddings.clone()
        embeddings_perturbed[0, last_token_idx, :] = perturbed_emb

        perturbed_hidden = get_hidden_state(
            model, embeddings_perturbed, last_token_idx
        )

        changed = check_output_changed(
            original_hidden, perturbed_hidden, threshold
        )

        if changed:
            high = mid
        else:
            best_s = mid
            low = mid

        if high - low < tolerance:
            break

    # ---------- 2. ULP refinement (exact) ----------
    s = np.float32(best_s)

    while True:
        s_next = np.nextafter(s, np.float32(np.inf))

        perturbed_emb = original_emb + torch.tensor(
            float(s_next), device=device
        ) * direction

        embeddings_perturbed = original_embeddings.clone()
        embeddings_perturbed[0, last_token_idx, :] = perturbed_emb

        perturbed_hidden = get_hidden_state(
            model, embeddings_perturbed, last_token_idx
        )

        changed = check_output_changed(
            original_hidden, perturbed_hidden, threshold
        )

        if changed:
            break

        s = s_next
    # print(f'Binary search best_s: {best_s}, ULP max_s: {float(s)}', ) # Suppress this print
    max_s = float(s)

    return max_s, float(low), float(high)



def singular_vector_stability_analysis(text="The capital of France is",
                                       s_max=1e-6,
                                       threshold=0,
                                       exp_dir="./results/exp13"):
    """
    Main function to perform singular vector stability analysis.

    Args:
        text: Input text.
        s_max: Maximum s value to search (default: 1e-6).
        threshold: Threshold for considering output as changed.
        exp_dir: Directory to save results.
    """
    print("="*80)
    print("EXPERIMENT 13: Singular Vector Stability Analysis")
    print("="*80)
    print(f"Input text: '{text}'")
    print(f"Max s value: {s_max:.2e}")
    print(f"Threshold: {threshold}")
    print("="*80)

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Tokenize input
    print("[2/5] Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])

    last_idx = inputs["input_ids"].shape[1] - 1

    # Compute SVD
    print("[3/5] Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    num_singular_vectors = Vt.shape[0]

    print(f"\nTop 5 singular values:")
    for i in range(min(5, len(S))):
        print(f"  σ_{i} = {S[i].item():.6f}")
    print(f"\nTotal number of singular vectors: {num_singular_vectors}")

    # Get original hidden state
    print("\n[4/5] Computing original hidden state...")
    original_hidden = get_hidden_state(model, embeddings, last_idx)

    singular_vector_indices = np.arange(num_singular_vectors)
    max_s_values = np.zeros(num_singular_vectors)
    low_values = np.zeros(num_singular_vectors)
    high_values = np.zeros(num_singular_vectors)

    # Binary search for each singular vector
    print(f"\n[5/5] Binary searching for max s along each singular vector...")
    print(f"Total singular vectors to process: {num_singular_vectors}")

    for i in tqdm(range(num_singular_vectors), desc="Processing singular vectors"):
        direction = Vt[i, :] # Get the i-th singular vector
        direction_tensor = direction.to(device).float() # Ensure it's on device and float32

        # Binary search for max s
        max_s, low_s, high_s = binary_search_max_s(
            model, embeddings, last_idx, direction_tensor,
            original_hidden, s_min=0, s_max=s_max, threshold=threshold
        )

        max_s_values[i] = max_s
        low_values[i] = low_s
        high_values[i] = high_s

    # Save results
    print("\n[6/6] Saving results...")
    os.makedirs(exp_dir, exist_ok=True)

    # Save data to NPZ
    npz_path = os.path.join(exp_dir, "singular_vector_stability_data.npz")
    np.savez(
        npz_path,
        singular_vector_indices=singular_vector_indices,
        max_s_values=max_s_values,
        low_values=low_values,
        high_values=high_values,
        singular_values_all=S.cpu().numpy(),
        input_text=text,
        s_max=s_max,
        threshold=threshold
    )
    print(f"Saved data to NPZ: {npz_path}")

    # Save data to CSV
    csv_path = os.path.join(exp_dir, "singular_vector_stability_data.csv")
    csv_data = np.vstack((singular_vector_indices, max_s_values, low_values, high_values)).T
    np.savetxt(csv_path, csv_data, delimiter=",", header="singular_vector_index,max_s,low,high", comments="")
    print(f"Saved data to CSV: {csv_path}")

    # Print statistics
    print("\n" + "="*80)
    print("STABILITY STATISTICS ACROSS SINGULAR VECTORS")
    print("="*80)
    print(f"Mean max s:        {np.mean(max_s_values):.6e}")
    print(f"Median max s:      {np.median(max_s_values):.6e}")
    print(f"Min max s:         {np.min(max_s_values):.6e}")
    print(f"Max max s:         {np.max(max_s_values):.6e}")
    print(f"Std dev:           {np.std(max_s_values):.6e}")
    print(f"Range:             {np.ptp(max_s_values):.6e}")

    min_idx = np.argmin(max_s_values)
    max_idx = np.argmax(max_s_values)
    print(f"\nMin max_s at singular vector index: {min_idx}")
    print(f"Max max_s at singular vector index: {max_idx}")

    # Create plot
    print("\n[7/7] Creating plot of max_s values...")
    fig, ax = plt.subplots(figsize=(16, 8)) # Wider figure for 4096 points
    ax.plot(singular_vector_indices, max_s_values, linewidth=1, color='purple', alpha=0.7)
    ax.axhline(y=np.mean(max_s_values), color='red', linestyle='--',
               label=f'Mean max s: {np.mean(max_s_values):.2e}')
    ax.axhline(y=np.median(max_s_values), color='green', linestyle=':',
               label=f'Median max s: {np.median(max_s_values):.2e}')
    ax.set_xlabel('Singular Vector Index', fontsize=12)
    ax.set_ylabel('Max Perturbation Magnitude (s)', fontsize=12)
    ax.set_title(f'Max Perturbation Magnitude along Each Singular Vector\nInput: "{text}"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plot_path = os.path.join(exp_dir, "singular_vector_stability_plot.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")

    plot_path_png = os.path.join(exp_dir, "singular_vector_stability_plot.png")
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path_png}")

    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)

    return singular_vector_indices, max_s_values


if __name__ == "__main__":
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp13_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Run the analysis
    singular_vector_indices, max_s_values = singular_vector_stability_analysis(
        text="The capital of France is",
        s_max=1e-6,
        threshold=0,
        exp_dir=exp_dir,
    )
