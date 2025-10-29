import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
from scipy import stats
from collections import OrderedDict

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

def compute_jacobian_svd_whole_model(model, embeddings, last_token_idx):
    """
    Compute Jacobian SVD for the WHOLE MODEL (final layer output)
    w.r.t. the last token embedding - same as original script
    """
    def forward_fn(flat_emb):
        emb = flat_emb.view(1, -1)
        mod_emb = embeddings.clone()
        mod_emb[0, last_token_idx, :] = emb
        outputs = model(inputs_embeds=mod_emb, output_hidden_states=True)
        return outputs.hidden_states[-1][0, last_token_idx, :]
    
    last_emb = embeddings[0, last_token_idx, :].clone().detach().requires_grad_(True)
    print(f"  Computing Jacobian for whole model (final layer output)...")
    jacobian = torch.autograd.functional.jacobian(forward_fn, last_emb, vectorize=True)
    print(f"  Computing SVD...")
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    return U, S, Vt

def register_hooks_for_layer(model, layer_idx):
    """
    Register hooks to capture outputs from specific submodules in a layer.
    Returns a dictionary to store activations and the hook handles.
    """
    activations = OrderedDict()
    hooks = []
    
    def save_output(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook_fn
    
    # Register hooks for the specified layer's submodules
    for name, module in model.named_modules():
        if f"layers.{layer_idx}" not in name:
            continue
        hook = module.register_forward_hook(save_output(f"layer_{layer_idx}_{name}"))
        hooks.append(hook)
        # Track key submodules
        # if name.endswith("self_attn") and "rotary_emb" not in name:
        #     hook = module.register_forward_hook(save_output(f"layer_{layer_idx}_self_attn"))
        #     hooks.append(hook)
        # elif name.endswith("mlp"):
        #     hook = module.register_forward_hook(save_output(f"layer_{layer_idx}_mlp"))
        #     hooks.append(hook)
        # elif name.endswith("input_layernorm"):
        #     hook = module.register_forward_hook(save_output(f"layer_{layer_idx}_input_ln"))
        #     hooks.append(hook)
        # elif name.endswith("post_attention_layernorm"):
        #     hook = module.register_forward_hook(save_output(f"layer_{layer_idx}_post_attn_ln"))
        #     hooks.append(hook)
    
    return activations, hooks

def remove_hooks(hooks):
    """Remove all registered hooks"""
    for hook in hooks:
        hook.remove()

def get_submodule_outputs(model, embeddings, last_token_idx, layer_idx):
    """
    Get outputs from all submodules in a specific layer for the last token
    """
    activations, hooks = register_hooks_for_layer(model, layer_idx)
    
    with torch.no_grad():
        _ = model(inputs_embeds=embeddings, output_hidden_states=True)
    
    # Extract last token outputs
    submodule_outputs = {}
    for name, act in activations.items():
        submodule_outputs[name] = act[0, last_token_idx, :].cpu().numpy()
    
    # Clean up hooks
    remove_hooks(hooks)
    
    return submodule_outputs

def print_singular_vector_stats(singular_vector, singular_value, singular_idx):
    """Print comprehensive statistics for a singular vector"""
    print(f"\n{'='*80}")
    print(f"SINGULAR VECTOR {singular_idx} STATISTICS (WHOLE MODEL)")
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

def compare_submodule_outputs(outputs1, outputs2, threshold=1e-6):
    """
    Compare outputs from two perturbations for each submodule
    """
    results = {}
    
    for submodule_name in outputs1.keys():
        if submodule_name not in outputs2:
            continue
        
        rep1 = outputs1[submodule_name]
        rep2 = outputs2[submodule_name]
        
        # Calculate differences
        diff = np.abs(rep1 - rep2)
        
        # Count changed indices
        changed_indices = np.sum(diff > threshold)
        percent_changed = (changed_indices / len(rep1)) * 100
        
        # Statistical measures
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        l2_dist = np.linalg.norm(rep1 - rep2)
        
        # Cosine similarity
        cosine_sim = np.dot(rep1, rep2) / (np.linalg.norm(rep1) * np.linalg.norm(rep2))
        
        results[submodule_name] = {
            'num_changed_indices': changed_indices,
            'percent_changed': percent_changed,
            'mean_absolute_diff': mean_diff,
            'max_absolute_diff': max_diff,
            'l2_distance': l2_dist,
            'cosine_similarity': cosine_sim
        }
    
    return results

def plot_submodule_comparison(results, e1, e2, layer_idx, singular_idx, save_prefix="submodule_comparison"):
    """Create visualization for submodule comparison"""
    submodules = list(results.keys())
    metrics = ['num_changed_indices', 'percent_changed', 'mean_absolute_diff', 
               'max_absolute_diff', 'l2_distance', 'cosine_similarity']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Submodule Comparison - Layer {layer_idx}: ε1={e1:.2e}, ε2={e2:.2e}, SV={singular_idx} (Whole Model)', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results[sub][metric] for sub in submodules]
        axes[idx].bar(range(len(submodules)), values)
        axes[idx].set_xticks(range(len(submodules)))
        axes[idx].set_xticklabels([s.split('_')[-1] for s in submodules], rotation=45, ha='right')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].grid(True, alpha=0.3)
        
        if 'diff' in metric or 'distance' in metric:
            axes[idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf', 
                dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf")
    plt.show()

def create_heatmap_submodules(outputs1, outputs2, e1, e2, layer_idx, singular_idx, save_prefix="heatmap_submodules"):
    """Create a heatmap showing the difference magnitude across submodules and dimensions"""
    submodules = list(outputs1.keys())
    
    # Calculate differences for all submodules
    diff_matrix = []
    for submodule_name in submodules:
        diff = np.abs(outputs1[submodule_name] - outputs2[submodule_name])
        diff_matrix.append(diff)
    
    diff_matrix = np.array(diff_matrix)  # Shape: (num_submodules, embedding_dim)
    
    # Create heatmap (sample dimensions for visibility)
    sample_dims = min(200, diff_matrix.shape[1])
    step = max(1, diff_matrix.shape[1] // sample_dims)
    sampled_diff = diff_matrix[:, ::step]
    
    plt.figure(figsize=(16, 6))
    sns.heatmap(sampled_diff, cmap='YlOrRd', 
                yticklabels=[s.split('_')[-1] for s in submodules],
                cbar_kws={'label': 'Absolute Difference'})
    plt.xlabel('Embedding Dimension (sampled)')
    plt.ylabel('Submodule')
    plt.title(f'Submodule Embedding Differences - Layer {layer_idx}: ε1={e1:.2e}, ε2={e2:.2e}, SV={singular_idx}')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf', 
                dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.pdf")
    plt.show()

def compare_perturbations_submodules(e1, e2, step_size, jumps, layer_idx, singular_idx, 
                                     text="The capital of France is", 
                                     threshold=1e-6, save_prefix="submodule_experiment"):
    """
    Main function to compare perturbations at the submodule level
    Uses SVD from the WHOLE MODEL, then tracks effects in specific layer
    """
    print("="*80)
    print(f"SUBMODULE-LEVEL PERTURBATION ANALYSIS")
    print(f"  SVD computed from: WHOLE MODEL (final layer)")
    print(f"  Tracking layer: {layer_idx}")
    print(f"  Singular vector index: {singular_idx}")
    print(f"  Input text: '{text}'")
    print("="*80)
    
    # Load model
    print("\n[1/7] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    
    # Tokenize input
    print("[2/7] Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()
    
    # Compute SVD for WHOLE MODEL
    print(f"\n[3/7] Computing Jacobian SVD for WHOLE MODEL...")
    U, S, Vt = compute_jacobian_svd_whole_model(model, embeddings, last_idx)
    
    print(f"\nTop 5 singular values (whole model):")
    for i in range(min(5, len(S))):
        print(f"  σ_{i} = {S[i].item():.6f}")
    
    # Save top 5 singular vectors
    print("\n[4/7] Saving top 5 singular vectors (whole model)...")
    top_5_singular_vectors = Vt[:5, :].cpu().numpy().T  # Shape: (embedding_dim, 5)
    top_5_singular_values = S[:5].cpu().numpy()
    
    np.save(f'{save_prefix}_whole_model_top5_singular_vectors.npy', top_5_singular_vectors)
    np.save(f'{save_prefix}_whole_model_top5_singular_values.npy', top_5_singular_values)
    
    print(f"Saved: {save_prefix}_whole_model_top5_singular_vectors.npy")
    print(f"  Shape: {top_5_singular_vectors.shape} (rows=dimensions, cols=singular vectors)")
    for i in range(5):
        print(f"  Column {i}: {i+1}st/nd/rd/th singular vector (σ={top_5_singular_values[i]:.6f})")
    
    # Get perturbation direction from whole model SVD
    direction = Vt[singular_idx, :]
    print(f"\n[5/7] Using singular vector {singular_idx} (whole model) with σ = {S[singular_idx].item():.6f}")
    
    # Print statistics
    direction_np = direction.cpu().numpy()
    print_singular_vector_stats(direction_np, S[singular_idx].item(), singular_idx)
    
    # Detect available submodules in target layer
    print(f"\n[6/7] Detecting available submodules in layer {layer_idx}...")
    activations, hooks = register_hooks_for_layer(model, layer_idx)
    with torch.no_grad():
        _ = model(inputs_embeds=embeddings, output_hidden_states=True)
    available_submodules = list(activations.keys())
    remove_hooks(hooks)
    
    print(f"  Available submodules: {available_submodules}")
    
    if not available_submodules:
        raise ValueError(f"No submodules found for layer {layer_idx}")
    
    # Compare perturbations for different jumps
    for jump in jumps:
        e2 = e1 + step_size * jump
        print("\n" + "="*80)
        print(f'Jump: {jump}')
        print(f"Comparing perturbations:")
        print(f"  ε1 = {e1:.15e}")
        print(f"  ε2 = {e2:.15e}")
        print("="*80)
        
        # Create perturbations using whole model singular vector
        print(f"\n[7/7] Creating perturbations and analyzing submodules...")
        perturbed_emb1 = original_input_emb + e1 * direction
        perturbed_emb2 = original_input_emb + e2 * direction
        
        # Get submodule outputs for both perturbations
        embeddings1 = embeddings.clone()
        embeddings1[0, last_idx, :] = perturbed_emb1
        outputs1 = get_submodule_outputs(model, embeddings1, last_idx, layer_idx)
        
        embeddings2 = embeddings.clone()
        embeddings2[0, last_idx, :] = perturbed_emb2
        outputs2 = get_submodule_outputs(model, embeddings2, last_idx, layer_idx)
        
        print(f"  Tracked submodules: {list(outputs1.keys())}")
        
        # Compare
        print(f"\nComparing submodule outputs (threshold={threshold})...")
        results = compare_submodule_outputs(outputs1, outputs2, threshold)
        
        # Print summary
        print("\n" + "="*80)
        print(f"SUMMARY STATISTICS - Layer {layer_idx}")
        print("="*80)
        print(f"{'Submodule':<30} {'Changed':<10} {'% Changed':<12} {'Mean Diff':<15} "
              f"{'Max Diff':<15} {'L2 Dist':<15} {'Cos Sim':<10}")
        print("-"*115)
        
        for submodule_name, metrics in results.items():
            short_name = submodule_name.split('_')[-2] + '_' + submodule_name.split('_')[-1]
            print(f"{short_name:<30} "
                  f"{metrics['num_changed_indices']:<10} "
                  f"{metrics['percent_changed']:<12.2f} "
                  f"{metrics['mean_absolute_diff']:<15.6e} "
                  f"{metrics['max_absolute_diff']:<15.6e} "
                  f"{metrics['l2_distance']:<15.6e} "
                  f"{metrics['cosine_similarity']:<10.6f}")
        
        # Save results
        print(f"\nSaving results...")
        np.savez(f'{save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.npz',
                 e1=e1,
                 e2=e2,
                 layer_idx=layer_idx,
                 singular_idx=singular_idx,
                 singular_value=S[singular_idx].cpu().numpy(),
                 outputs1={k: v for k, v in outputs1.items()},
                 outputs2={k: v for k, v in outputs2.items()},
                 results=results)
        print(f"Saved: {save_prefix}_layer{layer_idx}_e1_{e1:.2e}_e2_{e2:.2e}_sv{singular_idx}.npz")
        
        # Create plots
        plot_submodule_comparison(results, e1, e2, layer_idx, singular_idx, save_prefix)
        # create_heatmap_submodules(outputs1, outputs2, e1, e2, layer_idx, singular_idx, save_prefix)
    
    return results, outputs1, outputs2


# Example usage
if __name__ == "__main__":
    jumps = [1] #, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
    e1 = 1e-6 + 1815*2e-13
    step_size = 3*2e-14
    layer_idx = 31  # Specify which layer to analyze
    singular_idx = 0  # Use the first (largest) singular vector from whole model
    
    results, out1, out2 = compare_perturbations_submodules(
        e1=e1,
        e2=e1 + step_size,
        step_size=step_size,
        jumps=jumps,
        layer_idx=layer_idx,
        singular_idx=singular_idx,
        text="The capital of France is",
        threshold=0,
        save_prefix="submodule_whole_model_svd"
    )
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from collections import OrderedDict

# def load_llama_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         output_hidden_states=True
#     )
#     model.eval()
#     return model, tokenizer

# def register_forward_hooks(model, layer_idx_to_track=None):
#     """
#     Register hooks to capture intermediate outputs for a specific transformer layer
#     and its submodules (attention, feedforward, etc.).
#     """
#     activations = OrderedDict()

#     # Helper function for saving hook outputs
#     def save_output(name):
#         def hook_fn(module, input, output):
#             # If output is a tuple, take the first tensor (hidden states)
#             if isinstance(output, tuple):
#                 output = output[0]
#             activations[name] = output.detach().float().cpu()

#         return hook_fn

#     for name, module in model.named_modules():
#         if layer_idx_to_track is not None and f"layers.{layer_idx_to_track}" not in name:
#             continue  # Only track a specific layer

#         # Example submodule names (Llama, Falcon, etc. follow similar structure)
#         if "self_attn" in name and "rotary_emb" not in name:
#             module.register_forward_hook(save_output(f"{name}_self_attn_out"))
#         elif "mlp" in name:
#             module.register_forward_hook(save_output(f"{name}_mlp_out"))
#         elif "input_layernorm" in name:
#             module.register_forward_hook(save_output(f"{name}_input_layernorm_out"))
#         elif "post_attention_layernorm" in name:
#             module.register_forward_hook(save_output(f"{name}_post_attention_layernorm_out"))

#     return activations


# @torch.no_grad()
# def analyze_layer_evolution(model, tokenizer, text="The capital of France is", layer_to_track=0):
#     device = next(model.parameters()).device
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     print(f"\n[INFO] Tracking layer {layer_to_track} through its submodules...")
#     activations = register_forward_hooks(model, layer_to_track)

#     # Run forward pass
#     _ = model(**inputs)

#     # Summarize outputs
#     print(f"\nCaptured {len(activations)} intermediate activations:")
#     for name, act in activations.items():
#         print(f"  {name:<60} -> shape={tuple(act.shape)}")

#     # Example: analyze last token embedding changes
#     last_token_embs = {k: v[0, -1, :].numpy() for k, v in activations.items()}

#     return activations, last_token_embs


# if __name__ == "__main__":
#     model, tokenizer = load_llama_model()
#     activations, last_embs = analyze_layer_evolution(model, tokenizer, 
#                                                      text="The capital of France is Paris.",
#                                                      layer_to_track=5)

#     # Example: compare attention vs. MLP output difference
#     attn = last_embs.get('model.model.layers.5.self_attn_out')
#     mlp  = last_embs.get('model.model.layers.5.mlp_out')
#     if attn is not None and mlp is not None:
#         diff = ((mlp - attn) ** 2).mean() ** 0.5
#         print(f"\nL2 difference between attention and MLP outputs (last token): {diff:.6f}")
