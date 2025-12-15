import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compare two PyTorch tensor files.")
    parser.add_argument("file1", type=str, help="Path to the first tensor file.")
    parser.add_argument("file2", type=str, help="Path to the second tensor file.")
    args = parser.parse_args()

    print(f"Loading tensor from {args.file1}")
    tensor1 = torch.load(args.file1)
    print(f"Loading tensor from {args.file2}")
    tensor2 = torch.load(args.file2)

    if tensor1.shape != tensor2.shape:
        print("--- Tensors are DIFFERENT ---")
        print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return

    if tensor1.dtype != tensor2.dtype:
        print("--- Tensors are DIFFERENT ---")
        print(f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}")
        return

    # Element-wise comparison
    are_equal = torch.equal(tensor1, tensor2)

    if are_equal:
        print("\n✅ --- Tensors are IDENTICAL ---")
        print("The files are bit-for-bit the same.")
    else:
        print("\n❌ --- Tensors are DIFFERENT ---")
        # Calculate the difference
        diff = torch.abs(tensor1.float() - tensor2.float())
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        num_different_elements = torch.sum(diff > 0).item()
        total_elements = tensor1.numel()
        
        print(f"Number of different elements: {num_different_elements} / {total_elements}")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

if __name__ == "__main__":
    main()
