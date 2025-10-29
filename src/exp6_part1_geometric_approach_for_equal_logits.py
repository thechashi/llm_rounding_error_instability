"""
This script explores a geometric approach to finding input embeddings that produce equal logits for the top-2 most probable next tokens in a large language model.

The core idea is based on the observation that the logit for a token `i` is computed as the dot product of the final hidden state `h` and the token's unembedding vector `v_i` from the unembedding matrix `W_U`.
To make the logits for two tokens, `v1` and `v2`, equal, we need:
h · v1 = h · v2
This can be rewritten as:
h · (v1 - v2) = 0

This means the final hidden state `h` must be orthogonal to the difference vector of the two token unembeddings.

This script implements an optimization process to achieve this:
1.  It takes an input text and identifies the top-2 predicted next tokens.
2.  It then freezes the model's parameters and optimizes the input embeddings via gradient descent.
3.  The loss function is `(h · (v1 - v2))^2`, which drives the dot product to zero.
4.  The optimization modifies only the embeddings, not the model weights.

The script saves the optimized embedding and the resulting hidden state for further analysis.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def load_model(model_path):
    """Load model and tokenizer with memory optimization"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,  # Changed to float32
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "22GB", 1: "22GB"}
    )
    
    # Freeze all model parameters - no gradients needed
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model, tokenizer

def get_unembedding_matrix(model):
    """Get unembedding matrix W_U"""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    elif hasattr(model, 'embed_out'):
        return model.embed_out.weight
    else:
        raise ValueError("Cannot find unembedding matrix")

def optimize_for_equal_top2_geometric(
    model, 
    tokenizer, 
    input_text,
    num_iterations=500,
    lr=0.001,
    tolerance=1e-10
):
    """
    Optimize last token embedding using pure geometry: h·(v1-v2) = 0
    """
    device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get initial embeddings in float32
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"]).float()
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    
    # Get unembedding matrix
    W_U = get_unembedding_matrix(model)
    
    # First pass: identify top-2 tokens
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        h_init = outputs.hidden_states[-1][0, last_token_idx, :]
        
        # Get top-2 tokens - move W_U to same device as h_init
        all_dots = h_init @ W_U.to(h_init.device).T
        top2_values, top2_indices = torch.topk(all_dots, k=2)
        
        token1 = tokenizer.decode([top2_indices[0].item()])
        token2 = tokenizer.decode([top2_indices[1].item()])
        print(f"Target tokens: '{token1}' vs '{token2}'")
    
    # Extract unembedding vectors for top-2 - move to same device as h_init
    v1 = W_U[top2_indices[0], :].to(h_init.device)
    v2 = W_U[top2_indices[1], :].to(h_init.device)
    
    # Compute difference vector (FIXED - computed once)
    diff_vector = v1 - v2
    
    # Make ONLY embeddings trainable - THIS IS THE KEY
    embeddings = embeddings.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([embeddings], lr=lr)
    
    print(f"Input: '{input_text}'")
    print(f"Last token index: {last_token_idx}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"h_init device: {h_init.device}")
    print(f"v1 device: {v1.device}")
    print(f"diff_vector device: {diff_vector.device}")
    print(f"Model parameters frozen: {not any(p.requires_grad for p in model.parameters())}")
    print(f"Embeddings require grad: {embeddings.requires_grad}")
    print(f"\nOptimizing with PURE GEOMETRIC LOSS: [h·(v1-v2)]²")
    print(f"Stopping criterion: |h·(v1-v2)| < {tolerance}")
    
    best_loss = float('inf')
    best_embeddings = None
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass to get h - gradients only flow through embeddings
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, last_token_idx, :]
        
        # PURE GEOMETRIC LOSS: h·(v1-v2) should be 0
        orthogonality_loss = torch.dot(h, diff_vector).pow(2)
        
        loss = orthogonality_loss
        
        # Check for NaN or inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf detected at iteration {iteration}")
            break
        
        # Track best embeddings
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embeddings = embeddings.clone().detach()
        
        # Backward - gradients only computed for embeddings
        loss.backward()
        
        # Update only last token embedding
        with torch.no_grad():
            mask = torch.zeros_like(embeddings)
            mask[0, last_token_idx, :] = 1.0
            if embeddings.grad is not None:
                embeddings.grad *= mask
                
                # Gradient clipping
                grad_norm = torch.norm(embeddings.grad[0, last_token_idx, :])
                if grad_norm > 1.0:
                    embeddings.grad[0, last_token_idx, :] *= (1.0 / grad_norm)
        
        optimizer.step()
        
        if iteration % 50 == 0:
            with torch.no_grad():
                # Compute actual dot products (for monitoring only)
                dot_v1 = torch.dot(h, v1).item()
                dot_v2 = torch.dot(h, v2).item()
                dot_diff = torch.dot(h, diff_vector).item()
                
                # Compute cosine similarity to diff_vector
                h_norm = h / (h.norm() + 1e-8)
                diff_norm = diff_vector / (diff_vector.norm() + 1e-8)
                cos_sim = torch.dot(h_norm, diff_norm).item()
                
                print(f"Iter {iteration}: Loss={loss.item():.10f}")
                print(f"  h·(v1-v2) = {dot_diff:.10f}")
                print(f"  h·v1 = {dot_v1:.10f}, h·v2 = {dot_v2:.10f}, diff = {dot_v1-dot_v2:.10f}")
                print(f"  Cosine similarity to (v1-v2): {cos_sim:.6f}")
        
        # Early stopping if converged
        if abs(torch.dot(h, diff_vector).item()) < tolerance:
            print(f"\nConverged at iteration {iteration}!")
            print(f"  |h·(v1-v2)| = {abs(torch.dot(h, diff_vector).item()):.10f} < {tolerance}")
            break
    
    # Use best embeddings (lowest loss)
    if best_embeddings is not None:
        embeddings = best_embeddings
        print(f"\nUsing best embeddings with loss = {best_loss:.10f}")
    
    # Final check
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        h_final = outputs.hidden_states[-1][0, last_token_idx, :]
        logits = outputs.logits[0, last_token_idx, :]
        
        # Geometric check in float32
        h_final_f32 = h_final.float()
        v1_f32 = v1.float()
        v2_f32 = v2.float()
        diff_vector_f32 = diff_vector.float()
        
        dot_v1 = torch.dot(h_final_f32, v1_f32).item()
        dot_v2 = torch.dot(h_final_f32, v2_f32).item()
        dot_diff = torch.dot(h_final_f32, diff_vector_f32).item()
        dot_diff_via_subtraction = (dot_v1 - dot_v2)
        
        # Get top-2 from logits
        top2_logits, top2_indices_final = torch.topk(logits, k=2)
        probs = F.softmax(logits, dim=-1)
        top2_probs = probs[top2_indices_final]
        
        token1_final = tokenizer.decode([top2_indices_final[0].item()])
        token2_final = tokenizer.decode([top2_indices_final[1].item()])
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS (float32 precision)")
        print(f"{'='*80}")
        print(f"Geometric constraint:")
        print(f"  h·(v1-v2) = {dot_diff:.10f} (should be ~0)")
        print(f"  h·v1 - h·v2 = {dot_diff_via_subtraction:.10f}")
        print(f"  |difference| = {abs(dot_diff - dot_diff_via_subtraction):.10e}")
        print(f"  h·v1 = {dot_v1:.10f}")
        print(f"  h·v2 = {dot_v2:.10f}")
        print(f"  Difference: {dot_v1-dot_v2:.10f}")
        print(f"\nLogits (verification):")
        print(f"  Logit diff: {(top2_logits[0] - top2_logits[1]).item():.8f}")
        print(f"  Top-2: '{token1_final}' (p={top2_probs[0].item():.6f}) vs '{token2_final}' (p={top2_probs[1].item():.6f})")
    
    # Save optimized embedding
    optimized_last_token_embedding = embeddings[0, last_token_idx, :].detach().cpu()
    last_hidden_cpu = h_final.detach().cpu()
    
    return optimized_last_token_embedding, last_hidden_cpu, embeddings, last_token_idx

if __name__ == "__main__":
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp6_part1_geom_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Saving results to: {exp_dir}\n")

    # Clear cache before starting
    torch.cuda.empty_cache()

    model, tokenizer = load_model(MODEL_PATH)
    
    input_text = "The capital of France is"
    
    opt_embedding, last_hidden, full_embeddings, last_idx = optimize_for_equal_top2_geometric(
        model, tokenizer, input_text,
        num_iterations=500,
        lr=0.001,
        tolerance=1e-8
    )
    
    # Save results
    np.save(os.path.join(exp_dir, "optimized_last_token_embedding_geometric.npy"), opt_embedding.numpy())
    np.save(os.path.join(exp_dir, "optimized_last_hidden_geometric.npy"), last_hidden.numpy())
    torch.save({
        'full_embeddings': full_embeddings.detach().cpu(),
        'last_token_idx': last_idx,
        'input_text': input_text
    }, os.path.join(exp_dir, "optimized_state_geometric.pt"))

    print("\nSaved:")
    print(f"  - {os.path.join(exp_dir, 'optimized_last_token_embedding_geometric.npy')}")
    print(f"  - {os.path.join(exp_dir, 'optimized_last_hidden_geometric.npy')}")
    print(f"  - {os.path.join(exp_dir, 'optimized_state_geometric.pt')}")