import torch
import numpy as np
import os
import argparse

def analyze_perturbation_effect(dim, device, weight_scale):
    """
    Analyzes the effect of a small perturbation on the output of a sequence of
    matrix multiplications for a range of epsilon values.

    Args:
        dim (int): The dimension of the square matrices.
        device (str): The device to run the computations on.
        weight_scale (float): Scaling factor for the randomly generated weight matrices.
    """
    print("=" * 80)
    print("Experiment 10, Part 3: Perturbation Impact Analysis with 12 Sequential Matrices")
    print(f"Matrix dimension: {dim}")
    print(f"Weight scale: {weight_scale}")
    print(f"Running on device: {device}")
    print("=" * 80)

    # 1. Create 12 random weight matrices
    torch.manual_seed(0) # for reproducibility of matrices
    weight_matrices = [
        torch.randn(dim, dim, device=device, dtype=torch.float32) * weight_scale
        for _ in range(12)
    ]
    print(f"Successfully created 12 random weight matrices with shape: ({dim}, {dim}) and scale {weight_scale}\n")

    # For context, calculate the spectral norm of the combined weight matrix once
    # The matrices are applied in order W_0, W_1, ..., W_11 to an input x.
    # y = W_11 @ ... @ W_1 @ W_0 @ x.
    # The combined matrix is M = W_11 @ ... @ W_0.
    combined_matrix = torch.eye(dim, device=device, dtype=torch.float32)
    for matrix in weight_matrices:
        combined_matrix = torch.matmul(matrix, combined_matrix)

    spectral_norm = torch.linalg.matrix_norm(combined_matrix, ord=2)
    print("--- Theoretical Maximum Amplification ---")
    print(f"Spectral Norm of Combined Weight Matrix: {spectral_norm.item():.6f}")
    print("(This is the maximum possible amplification for any input vector)\n")

    # 2. Define the range of epsilons to test
    epsilons = np.logspace(-6, -19, num=14)

    # 3. Create a reproducible initial input vector and perturbation direction
    torch.manual_seed(42)
    input_vector = torch.randn(dim, device=device, dtype=torch.float32)
    perturbation_direction = torch.randn(dim, device=device, dtype=torch.float32)
    perturbation_direction /= torch.linalg.norm(perturbation_direction) # This is the unit vector
    
    # --- Print statistics of the initial input vector ---
    print("--- Initial Input Vector Statistics ---")
    print(f"  Dimensions: {dim}")
    print(f"  Norm (L2): {torch.linalg.norm(input_vector).item():.6f}")
    print(f"  Mean: {torch.mean(input_vector).item():.6e}")
    print(f"  Std Dev: {torch.std(input_vector).item():.6e}")
    print(f"  Min: {torch.min(input_vector).item():.6e}")
    print(f"  Max: {torch.max(input_vector).item():.6e}\n")

    # Calculate original output by applying matrices sequentially
    original_output = input_vector
    for matrix in weight_matrices:
        original_output = torch.matmul(matrix, original_output)

    # 4. Loop through each epsilon
    for epsilon in epsilons:
        print("-" * 80)
        print(f"Analyzing for Epsilon: {epsilon:.2e}")
        print("-" * 80)

        # Apply the perturbation
        input_change = epsilon * perturbation_direction
        perturbed_input = input_vector + input_change

        # Perform the matrix multiplications sequentially
        perturbed_output = perturbed_input
        for matrix in weight_matrices:
            perturbed_output = torch.matmul(matrix, perturbed_output)

        # Calculate the change in output
        output_change = perturbed_output - original_output

        # Count non-zero elements in the changes
        num_nonzero_input_changes = torch.count_nonzero(input_change).item()
        num_nonzero_output_changes = torch.count_nonzero(output_change).item()

        # Compute the magnitudes (L2 norm)
        norm_input_change = torch.linalg.norm(input_change)
        norm_output_change = torch.linalg.norm(output_change)

        # Calculate the amplification factor
        if norm_input_change.item() == 0:
            amplification_factor = float('inf')
        else:
            amplification_factor = norm_output_change / norm_input_change

        # --- Print Results for this Epsilon ---
        print(f"Non-Zero Input Changes:  {num_nonzero_input_changes}/{dim}")
        print(f"Non-Zero Output Changes: {num_nonzero_output_changes}/{dim}\n")

        print(f"Norm of Input Change:  {norm_input_change.item():.6e}")
        print(f"Norm of Output Change: {norm_output_change.item():.6e}")
        print(f"Magnification Factor:  {amplification_factor.item():.6f}\n")

    print("=" * 80)
    print("Analysis Complete.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the effect of a small perturbation on a sequence of matrix multiplications for a range of epsilons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=4096,
        help="Dimension of the square weight matrices."
    )
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=0.01,
        help="Scaling factor for the randomly generated weight matrices."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the computations on ('cuda' or 'cpu')."
    )

    args = parser.parse_args()
    
    analyze_perturbation_effect(
        dim=args.dim,
        device=args.device,
        weight_scale=args.weight_scale
    )
