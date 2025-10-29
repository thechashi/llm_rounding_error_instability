import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
from scipy import stats

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
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

def get_all_layer_hidden_states(model, embeddings, last_token_idx):
    """Get hidden states from all layers for the last token"""
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        all_layers = []
        for hidden_state in outputs.hidden_states:
            all_layers.append(hidden_state[0, last_token_idx, :].cpu().numpy())
    return all_layers

def print_singular_vector_stats(singular_vector, singular_value, singular_idx):
    """
    Print comprehensive statistics for a singular vector
    
    Args:
        singular_vector: numpy array of the singular vector
        singular_value: the corresponding singular value
        singular_idx: index of this singular vector
    """
    print(f"\n{'='*80}")
    print(f"SINGULAR VECTOR {singular_idx} STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nBasic Properties:")
    print(f"  Singular value:          {singular_value:.6e}")
    print(f"  Dimension:               {len(singular_vector)}")
    print(f"  L2 norm:                 {np.linalg.norm(singular_vector):.6f}")
    
    print(f"\nCentral Tendency:")
    print(f"  Mean:                    {np.mean(singular_vector):.6e}")
    print(f"  Mean (absolute):         {np.mean(np.abs(singular_vector)):.6e}")
    print(f"  Median:                  {np.median(singular_vector):.6e}")
    print(f"  Median (absolute):       {np.median(np.abs(singular_vector)):.6e}")
    
    print(f"\nDispersion:")
    print(f"  Std deviation:           {np.std(singular_vector):.6e}")
    print(f"  Variance:                {np.var(singular_vector):.6e}")
    print(f"  Range:                   {np.ptp(singular_vector):.6e}")
    
    print(f"\nExtreme Values:")
    print(f"  Max:                     {np.max(singular_vector):.6e}")
    print(f"  Min:                     {np.min(singular_vector):.6e}")
    print(f"  Max (absolute):          {np.max(np.abs(singular_vector)):.6e}")
    print(f"  Min (absolute):          {np.min(np.abs(singular_vector)):.6e}")
    
    print(f"\nPercentiles (absolute values):")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(np.abs(singular_vector), p)
        print(f"  {p:2d}th percentile:         {val:.6e}")
    
    print(f"\nDistribution Shape:")
    print(f"  Skewness:                {stats.skew(singular_vector):.6f}")
    print(f"  Kurtosis:                {stats.kurtosis(singular_vector):.6f}")
    
    print(f"\nSign Distribution:")
    n_positive = np.sum(singular_vector > 0)
    n_negative = np.sum(singular_vector < 0)
    n_zero = np.sum(singular_vector == 0)
    print(f"  Positive:                {n_positive} ({n_positive/len(singular_vector)*100:.2f}%)")
    print(f"  Negative:                {n_negative} ({n_negative/len(singular_vector)*100:.2f}%)")
    print(f"  Zero:                    {n_zero} ({n_zero/len(singular_vector)*100:.2f}%)")
    
    print(f"\nMagnitude Distribution:")
    abs_sv = np.abs(singular_vector)
    ranges = [
        (1e-1, float('inf'), "≥1e-1"),
        (1e-2, 1e-1, "1e-2 to 1e-1"),
        (1e-3, 1e-2, "1e-3 to 1e-2"),
        (1e-4, 1e-3, "1e-4 to 1e-3"),
        (1e-5, 1e-4, "1e-5 to 1e-4"),
        (0, 1e-5, "<1e-5")
    ]
    for low, high, label in ranges:
        count = np.sum((abs_sv >= low) & (abs_sv < high))
        pct = count / len(singular_vector) * 100
        print(f"  {label:20s}: {count:4d} ({pct:5.2f}%)")

def compare_layer_embeddings(rep1_layers, rep2_layers, threshold=1e-6):
    """
    Compare embeddings layer by layer
    
    Args:
        rep1_layers: List of layer embeddings for epsilon1
        rep2_layers: List of layer embeddings for epsilon2
        threshold: Threshold to consider a value as "changed"
    
    Returns:
        results: Dictionary with comparison statistics per layer
    """
    num_layers = len(rep1_layers)
    results = {
        'num_changed_indices': [],
        'percent_changed': [],
        'mean_absolute_diff': [],
        'max_absolute_diff': [],
        'l2_distance': [],
        'cosine_similarity': []
    }
    
    for layer_idx in range(num_layers):
        rep1 = rep1_layers[layer_idx]
        rep2 = rep2_layers[layer_idx]
        
        # Calculate differences
        diff = np.abs(rep1 - rep2)
        
        # Count changed indices (above threshold)
        changed_indices = np.sum(diff > threshold)
        percent_changed = (changed_indices / len(rep1)) * 100
        
        # Statistical measures
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        l2_dist = np.linalg.norm(rep1 - rep2)
        
        # Cosine similarity
        cosine_sim = np.dot(rep1, rep2) / (np.linalg.norm(rep1) * np.linalg.norm(rep2))
        
        results['num_changed_indices'].append(changed_indices)
        results['percent_changed'].append(percent_changed)
        results['mean_absolute_diff'].append(mean_diff)
        results['max_absolute_diff'].append(max_diff)
        results['l2_distance'].append(l2_dist)
        results['cosine_similarity'].append(cosine_sim)
    
    return results

def plot_comparison_results(results, e1, e2, singular_idx, save_prefix="comparison"):
    """Create visualization plots for the comparison results"""
    num_layers = len(results['num_changed_indices'])
    layers = np.arange(num_layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Layer-wise Comparison: ε1={e1:.2e}, ε2={e2:.2e}, Singular Vector {singular_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Number of changed indices
    axes[0, 0].plot(layers, results['num_changed_indices'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Number of Changed Indices')
    axes[0, 0].set_title('Changed Indices per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Percentage changed
    axes[0, 1].plot(layers, results['percent_changed'], marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Percentage Changed (%)')
    axes[0, 1].set_title('Percentage of Changed Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean absolute difference
    axes[0, 2].plot(layers, results['mean_absolute_diff'], marker='o', linewidth=2, color='green')
    axes[0, 2].set_xlabel('Layer Index')
    axes[0, 2].set_ylabel('Mean Absolute Difference')
    axes[0, 2].set_title('Mean Absolute Difference per Layer')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Plot 4: Max absolute difference
    axes[1, 0].plot(layers, results['max_absolute_diff'], marker='o', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Max Absolute Difference')
    axes[1, 0].set_title('Maximum Absolute Difference per Layer')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 5: L2 distance
    axes[1, 1].plot(layers, results['l2_distance'], marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('L2 Distance')
    axes[1, 1].set_title('L2 Distance between Representations')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Cosine similarity
    axes[1, 2].plot(layers, results['cosine_similarity'], marker='o', linewidth=2, color='brown')
    axes[1, 2].set_xlabel('Layer Index')
    axes[1, 2].set_ylabel('Cosine Similarity')
    axes[1, 2].set_title('Cosine Similarity between Representations')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf', 
                dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf")
    plt.show()

def create_heatmap(rep1_layers, rep2_layers, e1, e2, singular_idx, save_prefix="heatmap"):
    """Create a heatmap showing the difference magnitude across layers and dimensions"""
    # Calculate differences for all layers
    diff_matrix = []
    for layer_idx in range(len(rep1_layers)):
        diff = np.abs(rep1_layers[layer_idx] - rep2_layers[layer_idx])
        diff_matrix.append(diff)
    
    diff_matrix = np.array(diff_matrix)  # Shape: (num_layers, embedding_dim)
    
    # Create heatmap (sample dimensions for visibility)
    sample_dims = min(200, diff_matrix.shape[1])  # Sample dimensions if too large
    step = diff_matrix.shape[1] // sample_dims
    sampled_diff = diff_matrix[:, ::step]
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(sampled_diff, cmap='YlOrRd', 
                cbar_kws={'label': 'Absolute Difference'})
    plt.xlabel('Embedding Dimension (sampled)')
    plt.ylabel('Layer Index')
    plt.title(f'Layer-wise Embedding Differences: ε1={e1:.2e}, ε2={e2:.2e}, SV={singular_idx}')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf', 
                dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf")
    plt.show()

def compare_perturbations(e1, e2, step_size, jumps, singular_idx, text="The capital of France is", 
                         threshold=1e-6, save_prefix="comparison"):
    """
    Main function to compare two perturbations
    
    Args:
        e1: First epsilon value
        e2: Second epsilon value
        singular_idx: Index of singular vector to use (0 for largest singular value)
        text: Input text
        threshold: Threshold for considering a value as "changed"
        save_prefix: Prefix for saved files
    """
    print("="*80)

    print(f"  Singular vector index = {singular_idx}")
    print(f"  Input text: '{text}'")
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
    original_input_emb = embeddings[0, last_idx, :].clone()
    
    # Compute SVD
    print("[3/6] Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    
    print(f"\nTop 5 singular values:")
    for i in range(min(5, len(S))):
        print(f"  σ_{i} = {S[i].item():.6f}")
    
    # Get the perturbation direction
    direction = Vt[singular_idx, :]
    print(f"\nUsing singular vector {singular_idx} with σ = {S[singular_idx].item():.6f}")
    
    # Print singular vector statistics
    direction_np = direction.cpu().numpy()
    print_singular_vector_stats(direction_np, S[singular_idx].item(), singular_idx)
    
    for jump in jumps: 
        e2 = e1 + step_size*jump 
        print("\n" + "="*80)
        print(f'Jump: {jump}')
        print(f"Comparing perturbations:")
        print(f"  ε1 = {e1:.15e}")
        print(f"  ε2 = {e2:.15e}")
        print("="*80)

        # Create perturbations
        print(f"\n[4/6] Creating perturbations...")
        perturbed_emb1 = original_input_emb + e1 * direction
        perturbed_emb2 = original_input_emb + e2 * direction
        
        # Get layer-wise representations for both perturbations
        print("[5/6] Computing layer-wise representations...")
        
        embeddings1 = embeddings.clone()
        embeddings1[0, last_idx, :] = perturbed_emb1
        rep1_layers = get_all_layer_hidden_states(model, embeddings1, last_idx)
        
        embeddings2 = embeddings.clone()
        embeddings2[0, last_idx, :] = perturbed_emb2
        rep2_layers = get_all_layer_hidden_states(model, embeddings2, last_idx)
        
        print(f"  Number of layers: {len(rep1_layers)}")
        print(f"  Embedding dimension: {len(rep1_layers[0])}")
        
        # Compare
        print(f"\n[6/6] Comparing representations (threshold={threshold})...")
        results = compare_layer_embeddings(rep1_layers, rep2_layers, threshold)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"{'Layer':<8} {'Changed':<12} {'% Changed':<12} {'Mean Diff':<15} {'Max Diff':<15} {'L2 Dist':<15} {'Cos Sim':<10}")
        print("-"*80)
        
        for layer_idx in range(len(rep1_layers)):
            print(f"{layer_idx:<8} "
                f"{results['num_changed_indices'][layer_idx]:<12} "
                f"{results['percent_changed'][layer_idx]:<12.2f} "
                f"{results['mean_absolute_diff'][layer_idx]:<15.6e} "
                f"{results['max_absolute_diff'][layer_idx]:<15.6e} "
                f"{results['l2_distance'][layer_idx]:<15.6e} "
                f"{results['cosine_similarity'][layer_idx]:<10.6f}")
        
        # Save results
        print(f"\n[7/7] Saving results...")
        np.savez(f'{save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.npz',
                e1=e1,
                e2=e2,
                singular_idx=singular_idx,
                singular_value=S[singular_idx].cpu().numpy(),
                rep1_layers=np.array(rep1_layers),
                rep2_layers=np.array(rep2_layers),
                **results)
        print(f"Saved: {save_prefix}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.npz")
        
        # Create plots
        plot_comparison_results(results, e1, e2, singular_idx, save_prefix)
        create_heatmap(rep1_layers, rep2_layers, e1, e2, singular_idx, save_prefix)
        
    return 0, 0, 0 #results, rep1_layers, rep2_layers


# Example usage
if __name__ == "__main__":
    jumps = [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
    e1 = 1e-6 + 1815*2e-13 # 1e-6 + 1080*2.5e-14 # 
    step_size = 3*2e-14
    e2 = e1 + step_size # 1e-6 + 1090*2.5e-14 # 
    singular_idx = 0  # Use the first (largest) singular vector
    
    results, rep1, rep2 = compare_perturbations(
        e1=e1, 
        e2=e2, 
        step_size=step_size,
        jumps = jumps,
        singular_idx=singular_idx,
        text="The capital of France is",
        threshold=0,
        save_prefix="perturbation_comparison"

    )