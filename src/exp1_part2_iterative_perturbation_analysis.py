import csv
import torch
import numpy as np
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Loads the model on CPU with float32 precision."""
    print("Loading model on CPU with float32 precision...")
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
    """Computes the SVD of the Jacobian of the model's last hidden state with respect to the last token's embedding."""
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

def run_iterative_perturbation_analysis(num_steps, e1, step_size, singular_idx, text, threshold, save_prefix, use_float64_perturbation=True):
    """
    Main function to compare two perturbations iteratively.
    """

    print("="*80)
    print(f"  Singular vector index = {singular_idx}")
    print(f"  Input text: '{text}'")
    print(f"  Number of steps: {num_steps}")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    
    # Tokenize input
    print("[2/5] Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()
    
    # Compute SVD
    print("[3/5] Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    
    direction = Vt[singular_idx, :]
    print(f"\nUsing singular vector {singular_idx} with σ = {S[singular_idx].item():.6f}")

    # Prepare for iteration
    rep1_layers = get_all_layer_hidden_states(model, embeddings, last_idx)
    num_layers = len(rep1_layers)

    csv_path = f'{save_prefix}_iterative_log.csv'
    print(f"\n[4/5] Starting iterative analysis... Logging to {csv_path}")

    with open(csv_path, 'w', newline='') as csvfile:
        header = ['step', 'epsilon', 'layer_idx', 'num_changed_indices', 'percent_changed', 'mean_absolute_diff', 'max_absolute_diff', 'l2_distance', 'cosine_similarity']
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for step in range(1, num_steps + 1):
            e2 = e1 + step_size * step
            
            if use_float64_perturbation:
                # Perform perturbation calculation in float64 for higher precision
                perturbed_emb2_64 = original_input_emb.double() + float(e2) * direction.double()
                # Cast back to float32 before passing to the model
                perturbed_emb2 = perturbed_emb2_64.float()
            else:
                perturbed_emb2 = original_input_emb + float(e2) * direction
            
            embeddings2 = embeddings.clone()
            embeddings2[0, last_idx, :] = perturbed_emb2
            rep2_layers = get_all_layer_hidden_states(model, embeddings2, last_idx)
            
            results = compare_layer_embeddings(rep1_layers, rep2_layers, threshold)
            
            for layer_idx in range(num_layers):
                row = {
                    'step': step,
                    'epsilon': e2,
                    'layer_idx': layer_idx,
                    'num_changed_indices': results['num_changed_indices'][layer_idx],
                    'percent_changed': results['percent_changed'][layer_idx],
                    'mean_absolute_diff': results['mean_absolute_diff'][layer_idx],
                    'max_absolute_diff': results['max_absolute_diff'][layer_idx],
                    'l2_distance': results['l2_distance'][layer_idx],
                    'cosine_similarity': results['cosine_similarity'][layer_idx]
                }
                writer.writerow(row)
            
            if step % 100 == 0:
                print(f"  Step {step}/{num_steps} complete. Epsilon: {e2:.15e}")

    print(f"\n[5/5] Iterative analysis complete. Results saved to {csv_path}")
    return csv_path

# Example usage
if __name__ == "__main__":
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp1_part2_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # --- Parameters for the iterative experiment ---
    NUM_STEPS = 1000
    E1 = 1e-6 + 1815*2e-13
    STEP_SIZE = 3*2e-14
    SINGULAR_IDX = 0  # Use the first (largest) singular vector
    INPUT_TEXT = "The capital of France is"
    THRESHOLD = 0 # Exact comparison
    # ---

    run_iterative_perturbation_analysis(
        num_steps=NUM_STEPS,
        e1=E1,
        step_size=STEP_SIZE,
        singular_idx=SINGULAR_IDX,
        text=INPUT_TEXT,
        threshold=THRESHOLD,
        save_prefix=os.path.join(exp_dir, "iterative_perturbation"),
        use_float64_perturbation=True
    )
