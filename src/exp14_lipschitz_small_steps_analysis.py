"""
Experiment 14: Lipschitz Constants via Small Steps Analysis

This experiment investigates the local smoothness and numerical stability of the
Llama model by taking extremely small consecutive steps along a principal
singular direction and measuring how the output changes.

Purpose:
--------
Tests whether the model function behaves smoothly at very small scales by:
1. Taking tiny consecutive steps (e.g., 2e-14) along the first singular vector
2. Measuring consecutive output differences ||f(x+ε_{t+1}·Δx) - f(x+ε_t·Δx)||
3. Identifying discontinuities or unexpected jumps due to rounding errors
4. Empirically estimating local Lipschitz behavior

Key Questions:
--------------
- Is the function locally Lipschitz continuous at floating-point precision?
- Do consecutive differences scale linearly with step size?
- Where do numerical precision effects dominate over smooth behavior?
- Are there discrete jumps in the output despite smooth input changes?

Methodology:
------------
1. Load Llama model in float32
2. Compute Jacobian SVD for full model (last token embedding → last hidden state)
3. Extract first right singular vector (direction of maximum sensitivity)
4. Generate sequence of epsilon values: ε_0, ε_0 + δ, ε_0 + 2δ, ..., ε_0 + Nδ
   where δ is very small (e.g., 2e-14)
5. For each epsilon:
   a. Perturb input: x_perturbed = x_original + ε · v_1
   b. Compute output: y = f(x_perturbed)
   c. Measure consecutive difference: ||y_t - y_{t-1}||
6. Analyze patterns:
   - Identify zero-difference steps (no change despite input change)
   - Detect sudden jumps (rounding-induced discontinuities)
   - Compute statistics on difference magnitudes

Analysis:
---------
Computes and visualizes:
- Consecutive difference norms over all steps
- Distribution of difference magnitudes (histogram)
- Identification of "jump points" where differences are non-zero
- Comparison of theoretical vs observed Lipschitz constants
- Effects of floating-point precision on apparent smoothness

Expected Behavior:
------------------
IF function is smooth and Lipschitz:
  - Consecutive differences ≈ constant (proportional to step size)
  - No sudden jumps or zero plateaus

OBSERVED behavior (due to float32 precision):
  - Most steps have ZERO difference (no change in output)
  - Occasional SPIKES when rounding crosses a threshold
  - Non-uniform spacing of jumps
  - Function appears discontinuous at this scale

Use Case:
---------
Use this experiment to:
- Understand numerical stability at machine precision scales
- Quantify how floating-point arithmetic affects LLM behavior
- Identify critical thresholds where rounding matters
- Design robust perturbation magnitudes for other experiments

Relationship to Other Experiments:
----------------------------------
- Builds on experiment 1's SVD framework
- Complements experiment 8's perturbation analysis
- Provides micro-scale view vs exp1's macro comparisons
- Establishes lower bounds for meaningful perturbation sizes

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, matplotlib
- Llama-3.1-8B-Instruct model (float32)

Key Functions:
--------------
- load_model(): Load model in float32
- compute_jacobian_svd(): Compute Jacobian and SVD for full model
- get_hidden_state(): Extract final hidden state for perturbed input
- run_small_steps_experiment(): Main experiment loop
- plot_consecutive_differences(): Visualize difference patterns
- identify_jump_points(): Find steps with non-zero differences

Output:
-------
- Timestamped results directory (results/exp14_YYYY-MM-DD_HH-MM-SS/)
- Input embeddings for all epsilon values (.npy)
- Output embeddings for all epsilon values (.npy)
- Consecutive difference norms (.npy)
- Complete data archive (.npz)
- Visualization plots (PDF)
- Jump point analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from datetime import datetime
import argparse


def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Load Llama model in float32 precision"""
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
    """
    Compute Jacobian matrix and its SVD for the transformation:
    last token embedding -> last token hidden state

    Args:
        model: The language model
        embeddings: Input embeddings tensor
        last_token_idx: Index of the last token

    Returns:
        S, Vt: Singular values and right singular vectors
               (U not returned as it's not used in this experiment)
    """
    def forward_fn(flat_emb):
        emb = flat_emb.view(1, -1)
        mod_emb = embeddings.clone()
        mod_emb[0, last_token_idx, :] = emb
        outputs = model(inputs_embeds=mod_emb, output_hidden_states=True)
        return outputs.hidden_states[-1][0, last_token_idx, :]

    last_emb = embeddings[0, last_token_idx, :].clone().detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(forward_fn, last_emb, vectorize=True)
    _, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    return S, Vt


def get_hidden_state(model, embeddings, last_token_idx):
    """
    Get the final hidden state for the last token

    Args:
        model: The language model
        embeddings: Input embeddings tensor
        last_token_idx: Index of the last token

    Returns:
        hidden_state: Final layer hidden state for last token
    """
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][0, last_token_idx, :]
    return hidden_state


def apply_perturbation(original_embedding, epsilon, direction, use_float64=True):
    """
    Apply perturbation to embedding with optional float64 precision

    This function adds noise to the original embedding. When use_float64=True,
    it performs the addition in float64 precision to minimize rounding errors,
    then converts back to float32 for model compatibility.

    Args:
        original_embedding: Original embedding tensor (float32)
        epsilon: Perturbation magnitude (scalar)
        direction: Perturbation direction vector (float32)
        use_float64: If True, perform addition in float64 then convert back (default: True)

    Returns:
        perturbed_embedding: Perturbed embedding tensor (float32)
    """
    if use_float64:
        # Convert to float64 for precise arithmetic
        orig_f64 = original_embedding.double()
        dir_f64 = direction.double()
        eps_f64 = float(epsilon)  # Ensure epsilon is also high precision

        # Perform perturbation in float64
        perturbed_f64 = orig_f64 + eps_f64 * dir_f64

        # Convert back to float32 for model
        perturbed_embedding = perturbed_f64.float()
    else:
        # Direct float32 arithmetic (may have more rounding errors)
        perturbed_embedding = original_embedding + epsilon * direction

    return perturbed_embedding


def run_small_steps_experiment(
    text="The capital of France is",
    epsilon_start=1e-6,
    epsilon_step=2e-14,
    total_steps=500,
    use_float64=True,
    save_prefix="exp14"
):
    """
    Main experiment: take small consecutive steps and measure output changes

    Args:
        text: Input text to analyze
        epsilon_start: Starting epsilon value
        epsilon_step: Step size (very small, e.g., 2e-14)
        total_steps: Number of steps to take
        use_float64: If True, perform perturbation in float64 then convert to float32 (default: True)
        save_prefix: Prefix for saved files (should include exp_dir path)

    Returns:
        Dictionary containing all experiment results
    """
    print("="*80)
    print("LIPSCHITZ CONSTANTS - SMALL STEPS EXPERIMENT")
    print("="*80)
    print(f"  Input text: '{text}'")
    print(f"  Epsilon start: {epsilon_start:.2e}")
    print(f"  Epsilon step: {epsilon_step:.2e}")
    print(f"  Total steps: {total_steps}")
    print(f"  Epsilon end: {epsilon_start + total_steps * epsilon_step:.2e}")
    print(f"  Use float64 for perturbation: {use_float64}")
    print("="*80)

    # Load model
    print("\n[1/6] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Tokenize and get embeddings
    print("[2/6] Tokenizing input and getting embeddings...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])

    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()

    print(f"  Original embedding shape: {original_input_emb.shape}")
    print(f"  Original embedding norm: {torch.norm(original_input_emb).item():.6f}")

    # Compute SVD
    print("\n[3/6] Computing Jacobian SVD...")
    S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    direction_1 = Vt[0, :]  # First right singular vector

    print(f"\nTop 5 singular values:")
    for i in range(min(5, len(S))):
        print(f"  σ_{i} = {S[i].item():.6f}")
    print(f"\nUsing direction of 1st singular vector (σ_0 = {S[0].item():.6f})")

    # Generate epsilon values
    print("\n[4/6] Generating epsilon sequence...")
    epsilons = []
    eps = epsilon_start
    for _ in range(total_steps + 1):  # +1 to include endpoint
        epsilons.append(eps)
        eps += epsilon_step
    epsilons = np.array(epsilons)

    print(f"  Number of epsilon values: {len(epsilons)}")
    print(f"  Epsilon range: {epsilons[0]:.2e} to {epsilons[-1]:.2e}")

    # Storage for results
    all_input_embeddings = []
    all_output_embeddings = []
    difference_norms = []

    # Compute outputs for all epsilons
    print("\n[5/6] Computing outputs for all epsilons...")
    prev_output = None

    for i, eps in enumerate(tqdm(epsilons, desc="Processing epsilons")):
        # Perturb input embedding using float64 precision if enabled
        current_input = apply_perturbation(original_input_emb, eps, direction_1, use_float64)

        # Get output
        perturbed_emb = embeddings.clone()
        perturbed_emb[0, last_idx, :] = current_input
        current_output = get_hidden_state(model, perturbed_emb, last_idx)

        # Store
        all_input_embeddings.append(current_input.cpu().numpy())
        all_output_embeddings.append(current_output.cpu().numpy())

        # Compute consecutive difference (if not first iteration)
        if prev_output is not None:
            diff_norm = torch.norm(current_output - prev_output).item()
            difference_norms.append(diff_norm)

        # Update previous
        prev_output = current_output

    # Convert to numpy arrays
    all_input_embeddings = np.array(all_input_embeddings)
    all_output_embeddings = np.array(all_output_embeddings)
    difference_norms = np.array(difference_norms)

    print(f"\n[6/6] Analysis complete.")
    print(f"  Input embeddings shape: {all_input_embeddings.shape}")
    print(f"  Output embeddings shape: {all_output_embeddings.shape}")
    print(f"  Difference norms shape: {difference_norms.shape}")

    # Compute statistics
    print("\n" + "="*80)
    print("CONSECUTIVE DIFFERENCE STATISTICS")
    print("="*80)
    print(f"  Mean: {difference_norms.mean():.6e}")
    print(f"  Std:  {difference_norms.std():.6e}")
    print(f"  Min:  {difference_norms.min():.6e}")
    print(f"  Max:  {difference_norms.max():.6e}")

    # Identify jump points (non-zero differences)
    jump_indices = np.where(difference_norms > 0)[0]
    print(f"\n  Jump points (non-zero differences): {len(jump_indices)} out of {len(difference_norms)}")
    print(f"  Percentage of jumps: {len(jump_indices)/len(difference_norms)*100:.2f}%")

    if len(jump_indices) > 0:
        print(f"\n  First 10 jump points:")
        print(f"  {'Step':<8} {'Epsilon':<20} {'Difference Norm':<20}")
        print("  " + "-"*50)
        for idx in jump_indices[:10]:
            print(f"  {idx:<8} {epsilons[idx]:.12e} {difference_norms[idx]:.12e}")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Individual arrays
    np.save(f'{save_prefix}_input_embeddings.npy', all_input_embeddings)
    np.save(f'{save_prefix}_output_embeddings.npy', all_output_embeddings)
    np.save(f'{save_prefix}_difference_norms.npy', difference_norms)
    print(f"  Saved: {save_prefix}_input_embeddings.npy")
    print(f"  Saved: {save_prefix}_output_embeddings.npy")
    print(f"  Saved: {save_prefix}_difference_norms.npy")

    # Complete data archive
    np.savez(
        f'{save_prefix}_complete.npz',
        epsilons=epsilons,
        input_embeddings=all_input_embeddings,
        output_embeddings=all_output_embeddings,
        difference_norms=difference_norms,
        singular_values=S.cpu().numpy(),
        singular_vector_1=direction_1.cpu().numpy(),
        original_input_embedding=original_input_emb.cpu().numpy(),
        epsilon_start=epsilon_start,
        epsilon_step=epsilon_step,
        total_steps=total_steps,
        use_float64=use_float64,
        text=text
    )
    print(f"  Saved: {save_prefix}_complete.npz")

    # Return results dictionary
    return {
        'epsilons': epsilons,
        'input_embeddings': all_input_embeddings,
        'output_embeddings': all_output_embeddings,
        'difference_norms': difference_norms,
        'singular_values': S.cpu().numpy(),
        'jump_indices': jump_indices,
        'save_prefix': save_prefix
    }


def plot_consecutive_differences(results):
    """
    Create comprehensive visualization of consecutive differences

    Args:
        results: Dictionary returned by run_small_steps_experiment
    """
    epsilons = results['epsilons']
    difference_norms = results['difference_norms']
    jump_indices = results['jump_indices']
    save_prefix = results['save_prefix']

    # Compute epsilon range info
    epsilon_start = epsilons[0]
    epsilon_end = epsilons[-1]
    epsilon_step = epsilons[1] - epsilons[0] if len(epsilons) > 1 else 0
    total_steps = len(epsilons) - 1

    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)

    # Main plot: consecutive differences vs steps
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(len(difference_norms)), difference_norms, linewidth=0.8, alpha=0.7, color='blue')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('||f(x+ε(t+1)·Δx) - f(x+ε(t)·Δx)||', fontsize=12)
    plt.title(
        f'Consecutive Difference Norms vs Steps\n'
        f'ε from {epsilon_start:.2e} to {epsilon_end:.2e}, '
        f'step size {epsilon_step:.2e}, total steps {total_steps}',
        fontsize=14
    )
    plt.grid(True, alpha=0.3)

    # Highlight jump points
    if len(jump_indices) > 0:
        plt.scatter(jump_indices, difference_norms[jump_indices],
                   color='red', s=30, zorder=5, alpha=0.7,
                   label=f'Jumps ({len(jump_indices)})')
        plt.legend()

    # Zoomed-in view of first few jumps
    plt.subplot(2, 1, 2)
    if len(jump_indices) > 0:
        # Show region around first few jumps
        window = 50
        first_jump = jump_indices[0]
        start_idx = max(0, first_jump - window)
        end_idx = min(len(difference_norms), first_jump + window)

        plt.plot(range(start_idx, end_idx),
                difference_norms[start_idx:end_idx],
                linewidth=0.8, alpha=0.7, color='blue')

        # Highlight jumps in this window
        window_jumps = jump_indices[(jump_indices >= start_idx) & (jump_indices < end_idx)]
        if len(window_jumps) > 0:
            plt.scatter(window_jumps, difference_norms[window_jumps],
                       color='red', s=30, zorder=5, alpha=0.7)

        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Difference Norm', fontsize=12)
        plt.title(f'Zoomed View: Steps {start_idx} to {end_idx} (around first jump)', fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No jumps detected',
                ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_consecutive_differences.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_prefix}_consecutive_differences.pdf")
    plt.close()

    # Distribution plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if len(jump_indices) > 0:
        plt.hist(difference_norms[jump_indices], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Difference Norm', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Non-Zero Differences', fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No non-zero differences',
                ha='center', va='center', fontsize=14)

    plt.subplot(1, 2, 2)
    # Show spacing between jumps
    if len(jump_indices) > 1:
        jump_spacings = np.diff(jump_indices)
        plt.hist(jump_spacings, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Steps Between Jumps', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Jump Spacing', fontsize=12)
        plt.grid(True, alpha=0.3)
        print(f"  Jump spacing - Mean: {jump_spacings.mean():.2f}, Std: {jump_spacings.std():.2f}")
    else:
        plt.text(0.5, 0.5, 'Insufficient jumps for spacing analysis',
                ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_distributions.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_prefix}_distributions.pdf")
    plt.close()


def analyze_jump_patterns(results):
    """
    Detailed analysis of jump patterns

    Args:
        results: Dictionary returned by run_small_steps_experiment
    """
    epsilons = results['epsilons']
    difference_norms = results['difference_norms']
    jump_indices = results['jump_indices']
    save_prefix = results['save_prefix']

    print("\n" + "="*80)
    print("JUMP PATTERN ANALYSIS")
    print("="*80)

    if len(jump_indices) == 0:
        print("  No jumps detected - function appears constant at this scale.")
        return

    # Print all jump points
    print(f"\n  All jump points ({len(jump_indices)} total):")
    print(f"  {'Step':<8} {'Epsilon':<22} {'Difference Norm':<22}")
    print("  " + "-"*55)
    for idx in jump_indices:
        print(f"  {idx:<8} {epsilons[idx]:.15e} {difference_norms[idx]:.15e}")

    # Statistics on jumps
    jump_magnitudes = difference_norms[jump_indices]
    print(f"\n  Jump magnitude statistics:")
    print(f"    Mean:   {jump_magnitudes.mean():.6e}")
    print(f"    Std:    {jump_magnitudes.std():.6e}")
    print(f"    Min:    {jump_magnitudes.min():.6e}")
    print(f"    Max:    {jump_magnitudes.max():.6e}")
    print(f"    Median: {np.median(jump_magnitudes):.6e}")

    # Spacing analysis
    if len(jump_indices) > 1:
        jump_spacings = np.diff(jump_indices)
        print(f"\n  Jump spacing statistics:")
        print(f"    Mean spacing:   {jump_spacings.mean():.2f} steps")
        print(f"    Std spacing:    {jump_spacings.std():.2f} steps")
        print(f"    Min spacing:    {jump_spacings.min()} steps")
        print(f"    Max spacing:    {jump_spacings.max()} steps")
        print(f"    Median spacing: {np.median(jump_spacings):.2f} steps")

    # Save detailed jump analysis
    jump_data = {
        'jump_indices': jump_indices,
        'jump_epsilons': epsilons[jump_indices],
        'jump_magnitudes': jump_magnitudes
    }
    np.savez(f'{save_prefix}_jump_analysis.npz', **jump_data)
    print(f"\n  Saved: {save_prefix}_jump_analysis.npz")


def main():
    """Main execution function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description='Experiment 14: Lipschitz Constants via Small Steps Analysis'
    )
    parser.add_argument('--text', type=str,
                       default="The capital of France is",
                       help='Input text to analyze')
    parser.add_argument('--epsilon-start', type=float,
                       default=1e-6 + 1815*2e-13,
                       help='Starting epsilon value (default: 1e-6)')
    parser.add_argument('--epsilon-step', type=float,
                       default=3*2e-14,
                       help='Step size for epsilon (default: 2e-14)')
    parser.add_argument('--total-steps', type=int,
                       default=500,
                       help='Number of steps to take (default: 500)')
    parser.add_argument('--use-float64-perturbation', type=lambda x: x.lower() == 'true',
                       default=False,
                       help='Use float64 precision for perturbation calculation (default: True)')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory (default: auto-generated with timestamp)')

    args = parser.parse_args()

    # Create experiment directory with timestamp
    if args.output_dir:
        exp_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join("../results", f"exp14_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    print(f"\nResults will be saved to: {exp_dir}\n")

    # Run experiment
    save_prefix = os.path.join(exp_dir, "lipschitz_small_steps")
    results = run_small_steps_experiment(
        text=args.text,
        epsilon_start=args.epsilon_start,
        epsilon_step=args.epsilon_step,
        total_steps=args.total_steps,
        use_float64=args.use_float64_perturbation,
        save_prefix=save_prefix
    )

    # Create visualizations
    plot_consecutive_differences(results)

    # Analyze jump patterns
    analyze_jump_patterns(results)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"All results saved to: {exp_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
