
import torch
import os
import argparse

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="PyTorch determinism test script.")
    parser.add_argument("--output_file", type=str, default="tensor_output.pt", help="Path to save the output tensor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the test on (e.g., 'cuda:0').")
    args = parser.parse_args()

    # --- Enforce Deterministic Behavior ---
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # The following is a new option in PyTorch 1.8+ for further determinism
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        print("torch.use_deterministic_algorithms(True) not available in this PyTorch version.")

    print(f"Running on device: {args.device}")
    print(f"Deterministic backend: {torch.backends.cudnn.deterministic}")

    # --- Create Tensors ---
    try:
        device = torch.device(args.device)
        # Using bfloat16 as it's common in modern LLMs and sensitive to precision
        initial_tensor = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        weight_tensors = [
            torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) for _ in range(10)
        ]
    except Exception as e:
        print(f"Error creating tensors on {args.device}: {e}")
        return

    # --- Perform Operations ---
    # A chain of matrix multiplications to simulate a deep network
    result_tensor = initial_tensor
    for i, weight in enumerate(weight_tensors):
        print(f"Performing matmul #{i+1}...")
        result_tensor = torch.matmul(result_tensor, weight)

    # --- Save Output ---
    print(f"Saving final tensor to {args.output_file}")
    torch.save(result_tensor, args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()
