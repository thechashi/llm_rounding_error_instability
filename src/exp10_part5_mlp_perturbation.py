
import torch
import numpy as np
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

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

def mlp_block_forward(input_tensor, gate_proj, up_proj, down_proj):
    """Performs a forward pass through a Llama MLP block."""
    gate = torch.matmul(gate_proj, input_tensor)
    up = torch.matmul(up_proj, input_tensor)
    
    # SiLU activation
    activated_gate = F.silu(gate)
    
    # Element-wise multiplication
    intermediate_vec = activated_gate * up
    
    # Final down projection
    output_tensor = torch.matmul(down_proj, intermediate_vec)
    return output_tensor

def analyze_mlp_perturbation(text):
    """
    Analyzes the effect of a small perturbation on an embedding vector
    as it passes through a full Llama MLP block (SwiGLU).
    """
    print("=" * 80)
    print("Experiment 10, Part 5: MLP Block (SwiGLU) Perturbation Analysis")
    print(f"Input Text: '{text}'")
    print("=" * 80)

    # 1. Load model and tokenizer to get an embedding
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    # 2. Load the MLP weight matrices from the 'first_layer_weights' directory
    weights_dir = "first_layer_weights"
    try:
        print("Loading MLP weight matrices...")
        gate_proj_w = np.load(os.path.join(weights_dir, "layer0_mlp_gate_proj_weights.npy"))
        up_proj_w = np.load(os.path.join(weights_dir, "layer0_mlp_up_proj_weights.npy"))
        down_proj_w = np.load(os.path.join(weights_dir, "layer0_mlp_down_proj_weights.npy"))
        
        gate_proj_w = torch.from_numpy(gate_proj_w).to(device=device, dtype=torch.float32)
        up_proj_w = torch.from_numpy(up_proj_w).to(device=device, dtype=torch.float32)
        down_proj_w = torch.from_numpy(down_proj_w).to(device=device, dtype=torch.float32)
        print("MLP weights loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: MLP weight files not found in '{weights_dir}'")
        print("Please run 'exp9_part2_save_first_layer_weights.py' first.")
        return

    # 3. Get the initial embedding vector
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    input_vector = embeddings[0, last_token_idx, :].clone()
    dim = input_vector.shape[0]
    
    # --- Print statistics of the initial input vector ---
    print("--- Initial Embedding Vector Statistics ---")
    print(f"  Dimensions: {dim}")
    print(f"  Norm (L2): {torch.linalg.norm(input_vector).item():.6f}")
    print(f"  Mean: {torch.mean(input_vector).item():.6e}")
    print(f"  Std Dev: {torch.std(input_vector).item():.6e}\n")

    # 4. Define the range of epsilons and perturbation direction
    epsilons = np.logspace(-6, -19, num=14)
    torch.manual_seed(42) # for reproducible perturbation direction
    perturbation_direction = torch.randn(dim, device=device, dtype=torch.float32)
    perturbation_direction /= torch.linalg.norm(perturbation_direction)
    
    # Calculate the original output from the MLP block
    original_output = mlp_block_forward(input_vector, gate_proj_w, up_proj_w, down_proj_w)

    # 5. Loop through each epsilon
    for epsilon in epsilons:
        print("-" * 80)
        print(f"Analyzing for Epsilon: {epsilon:.2e}")
        print("-" * 80)

        input_change = epsilon * perturbation_direction
        perturbed_input = input_vector + input_change
        
        # Calculate the perturbed output from the MLP block
        perturbed_output = mlp_block_forward(perturbed_input, gate_proj_w, up_proj_w, down_proj_w)
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
        print(f"Magnification Factor (MLP Block): {amplification_factor.item():.6f}\n")

    print("=" * 80)
    print("Analysis Complete.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the effect of a small perturbation on an embedding passing through a Llama MLP block.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The capital of France is",
        help="Input text to generate the embedding from."
    )
    args = parser.parse_args()
    
    analyze_mlp_perturbation(
        text=args.text
    )

'''
exp10_part5_mlp_perturbation.py is designed to analyze how small numerical perturbations
  propagate through a fundamental building block of the Llama model: the Multi-Layer Perceptron
  (MLP) block, which uses a SwiGLU activation.

  Here's what the script does:

   1. Loads Llama Model Components:
       * It first loads the Llama model and its tokenizer to obtain a realistic input.
       * It then explicitly loads three specific weight matrices from the first layer's MLP
         block (gate_proj, up_proj, and down_proj) from your first_layer_weights directory.

   2. Generates an Initial Input:
       * Instead of a random vector, it takes an actual embedding vector from the Llama model,
         derived from a given input text (e.g., "The capital of France is").
       * It also prints statistics (norm, mean, std dev, min, max) of this initial embedding
         for context.

   3. Applies a Controlled Perturbation:
       * It defines a "perturbation direction" using a randomly generated unit vector.
       * A tiny, precisely controlled perturbation (input_change) is created by scaling this
         unit vector by a small epsilon value.
       * This input_change is added to the original embedding to create a perturbed_input.

   4. Simulates the Llama MLP Block:
       * It defines a function mlp_block_forward that manually executes the Llama MLP's forward
         pass:
           * The input (original or perturbed embedding) is first projected through gate_proj
             and up_proj weights.
           * The output of gate_proj is passed through a SiLU (Swish) activation function.
           * The activated gate_proj output is then element-wise multiplied with the up_proj
             output.
           * Finally, the result is projected through the down_proj weights to produce the MLP
             block's output.
       * This full MLP forward pass is performed for both the original and perturbed
         embeddings.

   5. Analyzes the Impact:
       * The entire process (perturbation and MLP forward pass) is repeated for a range of
         `epsilon` values, decreasing from 1e-6 down to 1e-19.
       * For each epsilon, it calculates the output_change by subtracting the original MLP
         output from the perturbed MLP output.
       * It quantifies:
           * The number of non-zero elements in the input_change (perturbation) and
             output_change (MLP output difference), showing how many dimensions are affected.
           * The L2 norm (magnitude) of both the input_change and output_change.
           * The Magnification Factor: This is the ratio of the output change's magnitude to
             the input change's magnitude, indicating how much the entire MLP block amplifies
             or attenuates the initial perturbation.

  In essence, exp10_part5 isolates a key component of the Llama model—the MLP block—and
  systematically investigates its sensitivity to incredibly small input variations, shedding
  light on numerical stability within the model's core computations.
'''