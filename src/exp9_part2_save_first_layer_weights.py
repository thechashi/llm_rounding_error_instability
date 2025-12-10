import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import os

def compute_weight_statistics(weight_matrix):
    """
    Compute comprehensive statistics for a weight matrix.
    
    Args:
        weight_matrix: torch tensor of shape (out_features, in_features) or (features,)
    
    Returns:
        Dictionary with statistics
    """
    w = weight_matrix.detach().cpu().numpy()
    w_flat = w.flatten()
    w_abs = np.abs(w_flat)
    
    if w.ndim == 1:
        out_features = w.shape[0]
        in_features = 1
        frobenius_norm = np.linalg.norm(w)
        spectral_norm = 'N/A'
        nuclear_norm = 'N/A'
    else:
        out_features = w.shape[0]
        in_features = w.shape[1]
        frobenius_norm = np.linalg.norm(w, 'fro')
        spectral_norm = np.linalg.norm(w, 2)
        nuclear_norm = np.linalg.norm(w, 'nuc')


    stats_dict = {
        'shape': w.shape,
        'out_features': out_features,
        'in_features': in_features,
        'total_params': w.size,
        
        # Basic statistics
        'mean': np.mean(w_flat),
        'std': np.std(w_flat),
        'var': np.var(w_flat),
        'median': np.median(w_flat),
        'min': np.min(w_flat),
        'max': np.max(w_flat),
        'range': np.ptp(w_flat),
        
        # Absolute value statistics
        'mean_abs': np.mean(w_abs),
        'std_abs': np.std(w_abs),
        'median_abs': np.median(w_abs),
        'min_abs': np.min(w_abs),
        'max_abs': np.max(w_abs),
        
        # Distribution shape
        'skewness': stats.skew(w_flat),
        'kurtosis': stats.kurtosis(w_flat),
        
        # Sign distribution
        'n_positive': np.sum(w_flat > 0),
        'n_negative': np.sum(w_flat < 0),
        'n_zero': np.sum(w_flat == 0),
        'pct_positive': (np.sum(w_flat > 0) / len(w_flat)) * 100,
        'pct_negative': (np.sum(w_flat < 0) / len(w_flat)) * 100,
        'pct_zero': (np.sum(w_flat == 0) / len(w_flat)) * 100,
        
        # Percentiles
        'percentile_1': np.percentile(w_abs, 1),
        'percentile_5': np.percentile(w_abs, 5),
        'percentile_10': np.percentile(w_abs, 10),
        'percentile_25': np.percentile(w_abs, 25),
        'percentile_50': np.percentile(w_abs, 50),
        'percentile_75': np.percentile(w_abs, 75),
        'percentile_90': np.percentile(w_abs, 90),
        'percentile_95': np.percentile(w_abs, 95),
        'percentile_99': np.percentile(w_abs, 99),
        
        # Norms
        'frobenius_norm': frobenius_norm,
        'spectral_norm': spectral_norm,
        'nuclear_norm': nuclear_norm,
        
        # Magnitude ranges
        'n_geq_1e-1': np.sum(w_abs >= 1e-1),
        'n_1e-2_to_1e-1': np.sum((w_abs >= 1e-2) & (w_abs < 1e-1)),
        'n_1e-3_to_1e-2': np.sum((w_abs >= 1e-3) & (w_abs < 1e-2)),
        'n_1e-4_to_1e-3': np.sum((w_abs >= 1e-4) & (w_abs < 1e-3)),
        'n_lt_1e-4': np.sum(w_abs < 1e-4),
    }
    
    return stats_dict, w

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Load Llama model in float32"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32
    )
    return model, tokenizer

def save_first_layer_weights(model, save_dir):
    """
    Saves the weights of all submodules in the first transformer layer and prints basic statistics.
    """
    print("="*80)
    print("SAVING WEIGHTS AND PRINTING STATISTICS FOR THE FIRST TRANSFORMER LAYER")
    print("="*80)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the first layer
    first_layer = model.model.layers[0]
    
    # Iterate through all submodules in the first layer
    for name, submodule in first_layer.named_modules():
        # Check if the submodule has a 'weight' attribute and it's a tensor
        if hasattr(submodule, 'weight') and isinstance(submodule.weight, torch.Tensor):
            print(f"\nProcessing submodule: {name}")
            
            # Compute statistics
            stats_dict, weights_np = compute_weight_statistics(submodule.weight)
            
            print(f"  Shape: {stats_dict['shape']}")
            print(f"  Out Features: {stats_dict['out_features']}, In Features: {stats_dict['in_features']}")
            print(f"  Size: {stats_dict['total_params']}")
            print(f"  Min: {stats_dict['min']:.6e}")
            print(f"  Max: {stats_dict['max']:.6e}")
            print(f"  Mean: {stats_dict['mean']:.6e}")
            print(f"  Std: {stats_dict['std']:.6e}")
            if weights_np.ndim == 1:
                print(f"  L2 Norm: {stats_dict['frobenius_norm']:.6f}")
            else:
                print(f"  Frobenius Norm: {stats_dict['frobenius_norm']:.6f}")

            
            # Sanitize the name for use as a filename
            safe_name = name.replace('.', '_')
            
            # Define the output path
            output_path = os.path.join(save_dir, f"layer0_{safe_name}_weights.npy")
            
            # Save the weights
            np.save(output_path, weights_np)
            print(f"  Saved weights to: {output_path}")

def main():
    # Define the directory to save the weights
    save_dir = "first_layer_weights"
    
    # Load model
    print("[1/3] Loading model...")
    model, _ = load_model()
    
    # Save first layer weights
    print("[2/3] Saving first layer weights...")
    save_first_layer_weights(model, save_dir)
    
    print("\n[3/3] Listing saved files...")
    saved_files = os.listdir(save_dir)
    for file in saved_files:
        print(f"  - {file}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"First layer weights saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
