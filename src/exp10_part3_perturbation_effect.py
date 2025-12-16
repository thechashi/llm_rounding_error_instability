
import torch
import numpy as np
import os
import argparse

def analyze_perturbation_effect(weight_path, device):
    """
    Analyzes the effect of a small perturbation on the output of a matrix multiplication
    for a range of epsilon values.

    Args:
        weight_path (str): Path to the .npy file containing the weight matrix.
        device (str): The device to run the computations on.
    """
    print("=" * 80)
    print("Experiment 10, Part 3: Perturbation Impact Analysis for a Range of Epsilons")
    print(f"Loading weight matrix from: {weight_path}")
    print(f"Running on device: {device}")
    print("=" * 80)

    # 1. Load the weight matrix
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        print("Please run 'exp9_part2_save_first_layer_weights.py' to generate the weight files.")
        return

    try:
        weight_matrix = np.load(weight_path)
        weight_matrix = torch.from_numpy(weight_matrix).to(device=device, dtype=torch.float32)
        dim = weight_matrix.shape[1]
        print(f"Successfully loaded weight matrix with shape: {weight_matrix.shape}\n")
    except Exception as e:
        print(f"Error loading weight matrix: {e}")
        return

    # For context, calculate the spectral norm of the weight matrix once
    spectral_norm = torch.linalg.matrix_norm(weight_matrix, ord=2)
    print("--- Theoretical Maximum Amplification ---")
    print(f"Spectral Norm of Weight Matrix: {spectral_norm.item():.6f}")
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

    original_output = torch.matmul(weight_matrix, input_vector)

    # 4. Loop through each epsilon
    for epsilon in epsilons:
        print("-" * 80)
        print(f"Analyzing for Epsilon: {epsilon:.2e}")
        print("-" * 80)

        # Apply the perturbation
        input_change = epsilon * perturbation_direction
        perturbed_input = input_vector + input_change

        # Perform the matrix multiplication
        perturbed_output = torch.matmul(weight_matrix, perturbed_input)

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
        description="Analyze the effect of a small perturbation on a matrix multiplication for a range of epsilons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="first_layer_weights/layer0_self_attn_q_proj_weights.npy",
        help="Path to the .npy file for the weight matrix."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the computations on ('cuda' or 'cpu')."
    )

    args = parser.parse_args()
    
    analyze_perturbation_effect(
        weight_path=args.weight_path,
        device=args.device
    )
'''
1. Create an Initial Vector: First, a random input_vector of dimension 4096 is generated.
      This serves as our baseline "clean" input.
   2. Define a Direction: A second random vector, perturbation_direction, is created. This
      vector determines the direction in which we will apply the nudge.
   3. Normalize the Direction: This direction vector is then normalized (divided by its own
      length or L2 norm) to create a unit vector. This is a critical step that ensures the
      vector has a length of exactly 1. By doing this, we separate the direction of the
      perturbation from its magnitude.
   4. Scale by Epsilon: The normalized direction vector is then multiplied by a very small
      scalar value called epsilon (e.g., 1e-6). This creates the final perturbation vector,
      input_change, which points in the chosen random direction and has a length exactly equal
      to epsilon.
   5. Apply the Perturbation: Finally, this small input_change vector is added to the original
      input_vector to create the perturbed_input.

  In short, we are nudging the original input vector by a tiny, precise amount (epsilon) in a
  randomly chosen but consistent direction. This allows us to see how the system reacts to very
  small, controlled changes.
'''