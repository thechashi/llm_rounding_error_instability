import torch
import numpy as np
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Loads the model and tokenizer in float32."""
    print("Loading Llama model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32
    )
    print("Model loaded successfully.\n")
    return model, tokenizer

def analyze_embedding_perturbation(weight_path, text):
    """
    Analyzes the effect of a small perturbation on a model embedding before a matrix multiplication.

    Args:
        weight_path (str): Path to the .npy file containing the weight matrix.
        text (str): The input text to generate the embedding from.
    """
    print("=" * 80)
    print("Experiment 10, Part 4: Embedding Perturbation Impact Analysis")
    print(f"Input Text: '{text}'")
    print(f"Loading weight matrix from: {weight_path}")
    print("=" * 80)

    # 1. Load model and tokenizer
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    # 2. Load the weight matrix
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return

    try:
        weight_matrix = np.load(weight_path)
        weight_matrix = torch.from_numpy(weight_matrix).to(device=device, dtype=torch.float32)
        print(f"Successfully loaded weight matrix with shape: {weight_matrix.shape}\n")
    except Exception as e:
        print(f"Error loading weight matrix: {e}")
        return

    # 3. Get the initial embedding vector
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    input_vector = embeddings[0, last_token_idx, :].clone()
    dim = input_vector.shape[0]

    # For context, calculate the spectral norm of the weight matrix once
    spectral_norm = torch.linalg.matrix_norm(weight_matrix, ord=2)
    print("--- Theoretical Maximum Amplification ---")
    print(f"Spectral Norm of Weight Matrix: {spectral_norm.item():.6f}")
    print("(This is the maximum possible amplification for any input vector)\n")
    
    # --- Print statistics of the initial input vector ---
    print("--- Initial Embedding Vector Statistics ---")
    print(f"  Dimensions: {dim}")
    print(f"  Norm (L2): {torch.linalg.norm(input_vector).item():.6f}")
    print(f"  Mean: {torch.mean(input_vector).item():.6e}")
    print(f"  Std Dev: {torch.std(input_vector).item():.6e}")
    print(f"  Min: {torch.min(input_vector).item():.6e}")
    print(f"  Max: {torch.max(input_vector).item():.6e}\n")

    # 4. Define the range of epsilons and perturbation direction
    epsilons = np.logspace(-6, -19, num=14)
    torch.manual_seed(42) # for reproducible perturbation direction
    perturbation_direction = torch.randn(dim, device=device, dtype=torch.float32)
    perturbation_direction /= torch.linalg.norm(perturbation_direction)
    
    original_output = torch.matmul(weight_matrix, input_vector)

    # 5. Loop through each epsilon
    for epsilon in epsilons:
        print("-" * 80)
        print(f"Analyzing for Epsilon: {epsilon:.2e}")
        print("-" * 80)

        input_change = epsilon * perturbation_direction
        perturbed_input = input_vector + input_change
        perturbed_output = torch.matmul(weight_matrix, perturbed_input)
        output_change = perturbed_output - original_output

        num_nonzero_input_changes = torch.count_nonzero(input_change).item()
        num_nonzero_output_changes = torch.count_nonzero(output_change).item()

        norm_input_change = torch.linalg.norm(input_change)
        norm_output_change = torch.linalg.norm(output_change)

        if norm_input_change.item() == 0:
            amplification_factor = float('inf')
        else:
            amplification_factor = norm_output_change / norm_input_change

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
        description="Analyze the effect of a small perturbation on a Llama model embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="first_layer_weights/layer0_self_attn_q_proj_weights.npy",
        help="Path to the .npy file for the weight matrix."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The capital of France is",
        help="Input text to generate the embedding from."
    )
    args = parser.parse_args()
    
    analyze_embedding_perturbation(
        weight_path=args.weight_path,
        text=args.text
    )

'''
Loads Llama Model Components: It loads the Llama model and its tokenizer to obtain a
      realistic input.
   2. Loads a Single Weight Matrix: It loads a single weight matrix (by default,
      layer0_self_attn_q_proj_weights.npy) from your first_layer_weights directory.
   3. Generates an Initial Input: It obtains an actual embedding vector from the Llama model,
      derived from a given input text (e.g., "The capital of France is"). It also prints
      statistics of this initial embedding.
   4. Applies a Controlled Perturbation:
       * A small input_change is created by scaling a randomly generated unit vector by an
         epsilon value.
       * This input_change is added to the original embedding to create a perturbed_input.
   5. Performs a Single Matrix Multiplication:
       * It applies the loaded single weight matrix to both the original and perturbed
         embeddings via torch.matmul().
       * It then calculates the output_change by finding the difference between these two
         results.
   6. Analyzes the Impact (for a single matrix):
       * This process is repeated for a range of `epsilon` values (from 1e-6 down to 1e-19).
       * For each epsilon, it calculates:
           * The number of non-zero elements in the input_change and output_change.
           * The L2 norm (magnitude) of both the input_change and output_change.
           * The Magnification Factor: The ratio of the output change's magnitude to the input
             change's magnitude, showing how much this single matrix multiplication amplifies
             the perturbation.
       * It also prints the spectral norm of this single weight matrix as a theoretical maximum
         amplification factor.
  In essence, this script currently examines the sensitivity of a single linear transformation
  (matrix multiplication) to small input perturbations when using a real model embedding.
'''
