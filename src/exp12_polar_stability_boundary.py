"""
Experiment 12: Polar Stability Boundary Analysis

This experiment analyzes the stability boundary of the Llama model by finding
the maximum perturbation magnitude that doesn't change the output for different
angular directions in the 2D space spanned by the first two singular vectors.

Purpose:
--------
Creates a polar plot showing the "stability boundary" - for each angle theta,
how far you can perturb the input embedding before the model output changes.

Methodology:
------------
1. Load Llama model in float32
2. Compute Jacobian SVD for full model (last token embedding → last hidden state)
3. Extract first two singular vectors (e1, e2)
4. For each theta from 0 to 2π (1000 steps):
   a. Create perturbation direction: d = cos(theta)*e1 + sin(theta)*e2
   b. Binary search for maximum s where output doesn't change
   c. Optionally, refine with a linear search.
   d. Perturbation: x_perturbed = x0 + s*d
5. Generate polar plot showing stability boundary

Output:
-------
- Timestamped results directory (results/exp12_YYYY-MM-DD_HH-MM-SS/)
- Polar plot showing stability boundary
- Data file (.npz and .csv) with theta and corresponding max_s, low, and high values.
- Detailed statistics about the boundary shape
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
    use_float64_perturbation=True
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

        if use_float64_perturbation:
            perturbed_emb = (original_emb.double() + mid * direction.double()).float()
        else:
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

        if use_float64_perturbation:
            perturbed_emb = (original_emb.double() + torch.tensor(float(s_next), device=device, dtype=torch.double) * direction.double()).float()
        else:
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

    max_s = float(s)

    return max_s, float(low), float(high)



def polar_stability_analysis(text="The capital of France is",
                             num_angles=1000,
                             s_max=1e-6,
                             threshold=0,
                             exp_dir="./results/exp12",
                             linear_refinement_step=0,
                             use_float64_perturbation=True):
    """
    Main function to perform polar stability boundary analysis

    Args:
        text: Input text
        num_angles: Number of angles to sample (default: 1000)
        s_max: Maximum s value to search (default: 1e-6)
        threshold: Threshold for considering output as changed
        exp_dir: Directory to save results
        linear_refinement_step: If > 0, step size for linear search refinement.
        use_float64_perturbation (bool): If True, perform perturbation in float64.
    """
    print("="*80)
    print("EXPERIMENT 12: Polar Stability Boundary Analysis")
    print("="*80)
    print(f"Input text: '{text}'")
    print(f"Number of angles: {num_angles}")
    print(f"Max s value: {s_max:.2e}")
    print(f"Threshold: {threshold}")
    print(f"Use float64 for perturbation: {use_float64_perturbation}")
    if linear_refinement_step > 0:
        print(f"Linear refinement step: {linear_refinement_step:.2e}")
    print("="*80)

    # Load model
    print("\n[1/6] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Tokenize input
    print("[2/6] Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])

    last_idx = inputs["input_ids"].shape[1] - 1

    # Compute SVD
    print("[3/6] Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)

    print(f"\nTop 5 singular values:")
    for i in range(min(5, len(S))):
        print(f"  σ_{i} = {S[i].item():.6f}")

    # Get first two singular vectors
    e1 = Vt[0, :]
    e2 = Vt[1, :]

    print(f"\nFirst singular vector norm: {torch.norm(e1).item():.6f}")
    print(f"Second singular vector norm: {torch.norm(e2).item():.6f}")
    print(f"Dot product (should be ~0): {torch.dot(e1, e2).item():.6e}")

    # Get original hidden state
    print("\n[4/6] Computing original hidden state...")
    original_hidden = get_hidden_state(model, embeddings, last_idx)

    # Generate theta values
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    max_s_values = np.zeros(num_angles)
    low_values = np.zeros(num_angles)
    high_values = np.zeros(num_angles)

    # Binary search for each theta
    print(f"\n[5/6] Binary searching for max s at each angle...")
    print(f"Total angles to process: {num_angles}")

    for i, theta in enumerate(tqdm(thetas, desc="Processing angles")):
        # Create perturbation direction
        direction = np.cos(theta) * e1.cpu().numpy() + np.sin(theta) * e2.cpu().numpy()
        direction_tensor = torch.from_numpy(direction).to(device).float()

        # Binary search for max s
        max_s, low_s, high_s = binary_search_max_s(
            model, embeddings, last_idx, direction_tensor,
            original_hidden, s_min=0, s_max=s_max, threshold=threshold,
            use_float64_perturbation=use_float64_perturbation
        )

        max_s_values[i] = max_s
        low_values[i] = low_s
        high_values[i] = high_s

    # Save results
    print("\n[6/6] Saving results...")
    os.makedirs(exp_dir, exist_ok=True)

    # Save data to NPZ
    npz_path = os.path.join(exp_dir, "polar_boundary_data.npz")
    np.savez(
        npz_path,
        thetas=thetas,
        max_s_values=max_s_values,
        low_values=low_values,
        high_values=high_values,
        singular_values=S[:2].cpu().numpy(),
        e1=e1.cpu().numpy(),
        e2=e2.cpu().numpy(),
        input_text=text,
        num_angles=num_angles,
        s_max=s_max,
        threshold=threshold
    )
    print(f"Saved data to NPZ: {npz_path}")

    # Save data to CSV
    csv_path = os.path.join(exp_dir, "polar_boundary_data.csv")
    csv_data = np.vstack((thetas, max_s_values, low_values, high_values)).T
    np.savetxt(csv_path, csv_data, delimiter=",", header="theta,max_s,low,high", comments="")
    print(f"Saved data to CSV: {csv_path}")

    # Print statistics
    print("\n" + "="*80)
    print("BOUNDARY STATISTICS")
    print("="*80)
    print(f"Mean max s:        {np.mean(max_s_values):.6e}")
    print(f"Median max s:      {np.median(max_s_values):.6e}")
    print(f"Min max s:         {np.min(max_s_values):.6e}")
    print(f"Max max s:         {np.max(max_s_values):.6e}")
    print(f"Std dev:           {np.std(max_s_values):.6e}")
    print(f"Range:             {np.ptp(max_s_values):.6e}")

    # Find angles with min and max values
    min_idx = np.argmin(max_s_values)
    max_idx = np.argmax(max_s_values)
    print(f"\nMin at theta={thetas[min_idx]:.4f} rad ({np.degrees(thetas[min_idx]):.2f}°)")
    print(f"Max at theta={thetas[max_idx]:.4f} rad ({np.degrees(thetas[max_idx]):.2f}°)")

    # Create polar plot
    print("\n[7/7] Creating polar plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Plot the boundary
    ax.plot(thetas, max_s_values, linewidth=2, color='blue', label='Stability Boundary')
    ax.fill(thetas, max_s_values, alpha=0.3, color='blue')

    # Mark min and max points
    ax.plot(thetas[min_idx], max_s_values[min_idx], 'ro', markersize=10,
            label=f'Min: {max_s_values[min_idx]:.2e}')
    ax.plot(thetas[max_idx], max_s_values[max_idx], 'go', markersize=10,
            label=f'Max: {max_s_values[max_idx]:.2e}')

    # Formatting
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(f'Stability Boundary in Singular Vector Space\n"{text}"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    # Add radial label
    ax.set_ylabel('Max perturbation magnitude (s)', labelpad=30)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(exp_dir, "polar_stability_boundary.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")

    # Also save as PNG
    plot_path_png = os.path.join(exp_dir, "polar_stability_boundary.png")
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path_png}")

    plt.close()

    # Create additional Cartesian plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.degrees(thetas), max_s_values, linewidth=2, color='blue')
    ax.axhline(y=np.mean(max_s_values), color='red', linestyle='--',
               label=f'Mean: {np.mean(max_s_values):.2e}')
    ax.fill_between(np.degrees(thetas), max_s_values, alpha=0.3, color='blue')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Max perturbation magnitude (s)', fontsize=12)
    ax.set_title(f'Stability Boundary vs Angle\n"{text}"', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    cartesian_path = os.path.join(exp_dir, "stability_boundary_cartesian.pdf")
    plt.savefig(cartesian_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cartesian plot to: {cartesian_path}")

    cartesian_path_png = os.path.join(exp_dir, "stability_boundary_cartesian.png")
    plt.savefig(cartesian_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved Cartesian plot to: {cartesian_path_png}")

    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)

    return thetas, max_s_values


if __name__ == "__main__":
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp12_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Run the analysis
    thetas, max_s_values = polar_stability_analysis(
        text="The capital of France is",
        num_angles=1000,
        s_max=1e-6,
        threshold=0,
        exp_dir=exp_dir,
        use_float64_perturbation=True
    )
