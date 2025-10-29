"""
Experiment 7 Part 1: Logit Difference Heatmap Generation

This script generates 2D heatmaps showing how the logit difference (L1 - L2) changes
when perturbing an optimized embedding along two singular vector directions simultaneously.

The process:
1. Finds an embedding x0 where the top-2 tokens have equal logits (L1(x0) ≈ L2(x0))
2. Computes the Jacobian and its SVD to identify sensitive directions
3. Creates a 2D grid of perturbations: x = x0 + e1*h1 + e2*h2
   - h1, h2 are right singular vectors from the Jacobian SVD
   - e1, e2 are perturbation magnitudes (epsilon values)
4. Evaluates L1(x) - L2(x) at each grid point
5. Generates both color heatmaps and binary decision boundary maps

The heatmaps reveal the complex, non-linear decision boundary between token predictions
and demonstrate how small perturbations can cause unpredictable logit flips.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os
from datetime import datetime

def load_model(model_path):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "22GB", 1: "22GB"}
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model, tokenizer

def get_unembedding_matrix(model):
    """Get unembedding matrix"""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    elif hasattr(model, 'embed_out'):
        return model.embed_out.weight
    else:
        raise ValueError("Cannot find unembedding matrix")

def optimize_for_equal_logits(model, tokenizer, input_text, num_iterations=500, lr=0.001, tolerance=1e-10):
    """Find x0 where L1(x0) ≈ L2(x0)"""
    device = next(model.parameters()).device
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"]).float()
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    W_U = get_unembedding_matrix(model)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        h_init = outputs.hidden_states[-1][0, last_token_idx, :]
        all_dots = h_init @ W_U.to(h_init.device).T
        top2_values, top2_indices = torch.topk(all_dots, k=2)
        
        token1 = tokenizer.decode([top2_indices[0].item()])
        token2 = tokenizer.decode([top2_indices[1].item()])
        print(f"Target tokens: '{token1}' vs '{token2}'")
    
    v1 = W_U[top2_indices[0], :].to(h_init.device)
    v2 = W_U[top2_indices[1], :].to(h_init.device)
    diff_vector = v1 - v2
    
    embeddings = embeddings.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([embeddings], lr=lr)
    
    best_loss = float('inf')
    best_embeddings = None
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, last_token_idx, :]
        
        orthogonality_loss = torch.dot(h, diff_vector).pow(2)
        loss = orthogonality_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            break
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embeddings = embeddings.clone().detach()
        
        loss.backward()
        
        with torch.no_grad():
            mask = torch.zeros_like(embeddings)
            mask[0, last_token_idx, :] = 1.0
            if embeddings.grad is not None:
                embeddings.grad *= mask
                grad_norm = torch.norm(embeddings.grad[0, last_token_idx, :])
                if grad_norm > 1.0:
                    embeddings.grad[0, last_token_idx, :] *= (1.0 / grad_norm)
        
        optimizer.step()
        
        if abs(torch.dot(h, diff_vector).item()) < tolerance:
            break
    
    if best_embeddings is not None:
        embeddings = best_embeddings
    
    x0 = embeddings[0, last_token_idx, :].detach()
    
    return x0, top2_indices, embeddings, last_token_idx

def compute_svd_directions(model, tokenizer, embeddings, last_token_idx):
    """Compute SVD of the Jacobian at x0"""
    device = next(model.parameters()).device
    x0 = embeddings[0, last_token_idx, :].clone().detach().requires_grad_(True)
    
    # Create modified embeddings with x0
    embeddings_modified = embeddings.clone().detach()
    embeddings_modified[0, last_token_idx, :] = x0
    
    outputs = model(inputs_embeds=embeddings_modified, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, last_token_idx, :]
    
    # Compute Jacobian
    embedding_dim = x0.shape[0]
    hidden_dim = h.shape[0]
    
    jacobian = torch.zeros(hidden_dim, embedding_dim, device=device)
    
    for i in range(hidden_dim):
        if x0.grad is not None:
            x0.grad.zero_()
        
        h[i].backward(retain_graph=True)
        jacobian[i, :] = x0.grad.clone()
    
    # Perform SVD
    U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
    
    # Vt has shape (embedding_dim, embedding_dim)
    # Right singular vectors are rows of Vt
    return Vt
# # Normal
# e_min=-1e-6, e_max=1e-6, step=1e-8

# # 2x zoom (half the range)
# e_min=-5e-7, e_max=5e-7, step=5e-9

# # 4x zoom (quarter the range)
# e_min=-2.5e-7, e_max=2.5e-7, step=2.5e-9

# # 10x zoom
# e_min=-1e-7, e_max=1e-7, step=1e-9

# # 100x zoom (focused around origin)
# e_min=-1e-8, e_max=1e-8, step=1e-10
def compute_logit_difference_grid(model, embeddings, last_token_idx, x0, h1, h2, token_indices, e_min=-2.5e-7, e_max=2.5e-7, step=2.5e-9):
    """Compute L1(x0 + e1*h1 + e2*h2) - L2(x0 + e1*h1 + e2*h2) over a grid"""
    device = next(model.parameters()).device
    W_U = get_unembedding_matrix(model)
    
    token1_idx = token_indices[0].item()
    token2_idx = token_indices[1].item()
    
    v1 = W_U[token1_idx, :].to(device)
    v2 = W_U[token2_idx, :].to(device)
    
    # Create grid
    num_points = int((e_max - e_min) / step) + 1
    e1_values = torch.linspace(e_min, e_max, num_points, device=device)
    e2_values = torch.linspace(e_min, e_max, num_points, device=device)
    
    logit_diff_grid = torch.zeros(num_points, num_points, device=device)
    
    print(f"Computing grid of size {num_points}x{num_points}...")
    
    for i, e1 in tqdm(enumerate(e1_values)):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_points}")
        
        for j, e2 in enumerate(e2_values):
            # Compute perturbed embedding
            x_perturbed = x0 + e1 * h1 + e2 * h2
            
            # Create modified embeddings
            embeddings_modified = embeddings.clone().detach()
            embeddings_modified[0, last_token_idx, :] = x_perturbed
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs_embeds=embeddings_modified, output_hidden_states=True)
                h = outputs.hidden_states[-1][0, last_token_idx, :]
                
                # Compute logits
                L1 = torch.dot(h, v1).item()
                L2 = torch.dot(h, v2).item()
                
                logit_diff_grid[i, j] = L1 - L2
    
    return logit_diff_grid.cpu().numpy(), e1_values.cpu().numpy(), e2_values.cpu().numpy()

def save_heatmap(grid, e1_values, e2_values, title, filename):
    """Save color heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(grid.T, extent=[e1_values[0], e1_values[-1], e2_values[0], e2_values[-1]], 
               origin='lower', cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='L1 - L2')
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def save_bw_heatmap(grid, e1_values, e2_values, title, filename):
    """Save black and white heatmap where L1 >= L2 is white (1), L1 < L2 is black (0)"""
    # Create binary grid: 1 where L1 >= L2 (positive or zero), 0 where L1 < L2 (negative)
    binary_grid = (grid >= 0).astype(int)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_grid.T, extent=[e1_values[0], e1_values[-1], e2_values[0], e2_values[-1]], 
               origin='lower', cmap='binary', aspect='auto', vmin=0, vmax=1)
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.title(title + ' (Binary: White = L1≥L2, Black = L1<L2)')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def save_matrix(grid, e1_values, e2_values, filename):
    """Save the grid matrix and axis values to a .npz file"""
    np.savez(filename, 
             grid=grid, 
             e1_values=e1_values, 
             e2_values=e2_values)
    print(f"Saved matrix: {filename}")

if __name__ == "__main__":
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp7_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved to: {exp_dir}\n")

    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"

    torch.cuda.empty_cache()
    
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    
    input_text = "The capital of France is"
    
    print("\nStep 1: Finding x0...")
    x0, token_indices, embeddings, last_token_idx = optimize_for_equal_logits(
        model, tokenizer, input_text,
        num_iterations=1000,
        lr=0.001,
        tolerance=1e-10
    )
    
    print("\nStep 2: Computing SVD...")
    Vt = compute_svd_directions(model, tokenizer, embeddings, last_token_idx)
    
    # Extract singular vectors
    h1_first = Vt[0, :]  # First right singular vector
    h2_second = Vt[1, :]  # Second right singular vector
    h2_tenth = Vt[9, :]  # Tenth right singular vector
    h2_last = Vt[-1, :]  # 4096th right singular vector
    
    print(f"\nSingular vector shapes: {Vt.shape}")
    
    # Case 1: h1 = first, h2 = second
    print("\nCase 1: h1=1st, h2=2nd singular vectors")
    grid1, e1_vals, e2_vals = compute_logit_difference_grid(
        model, embeddings, last_token_idx, x0, h1_first, h2_second, token_indices
    )
    save_heatmap(grid1, e1_vals, e2_vals,
                 "h1=1st, h2=2nd singular vectors",
                 os.path.join(exp_dir, "logit_diff_1st_2nd.png"))
    save_bw_heatmap(grid1, e1_vals, e2_vals,
                    "h1=1st, h2=2nd singular vectors",
                    os.path.join(exp_dir, "logit_diff_1st_2nd_bw.png"))
    save_matrix(grid1, e1_vals, e2_vals, os.path.join(exp_dir, "logit_diff_1st_2nd.npz"))
    
    # Case 2: h1 = first, h2 = tenth
    print("\nCase 2: h1=1st, h2=10th singular vectors")
    grid2, e1_vals, e2_vals = compute_logit_difference_grid(
        model, embeddings, last_token_idx, x0, h1_first, h2_tenth, token_indices
    )
    save_heatmap(grid2, e1_vals, e2_vals,
                 "h1=1st, h2=10th singular vectors",
                 os.path.join(exp_dir, "logit_diff_1st_10th.png"))
    save_bw_heatmap(grid2, e1_vals, e2_vals,
                    "h1=1st, h2=10th singular vectors",
                    os.path.join(exp_dir, "logit_diff_1st_10th_bw.png"))
    save_matrix(grid2, e1_vals, e2_vals, os.path.join(exp_dir, "logit_diff_1st_10th.npz"))
    
    # Case 3: h1 = first, h2 = 4096th
    print("\nCase 3: h1=1st, h2=4096th singular vectors")
    grid3, e1_vals, e2_vals = compute_logit_difference_grid(
        model, embeddings, last_token_idx, x0, h1_first, h2_last, token_indices
    )
    save_heatmap(grid3, e1_vals, e2_vals,
                 "h1=1st, h2=4096th singular vectors",
                 os.path.join(exp_dir, "logit_diff_1st_4096th.png"))
    save_bw_heatmap(grid3, e1_vals, e2_vals,
                    "h1=1st, h2=4096th singular vectors",
                    os.path.join(exp_dir, "logit_diff_1st_4096th_bw.png"))
    save_matrix(grid3, e1_vals, e2_vals, os.path.join(exp_dir, "logit_diff_1st_4096th.npz"))
    
    print("\nAll done!") 