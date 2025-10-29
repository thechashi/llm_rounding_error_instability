import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def load_model(model_path):
    """Load model and tokenizer with memory optimization"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "22GB", 1: "22GB"}
    )
    model.eval()
    return model, tokenizer

def optimize_for_equal_top2_probabilities(
    model, 
    tokenizer, 
    input_text,
    num_iterations=500,
    lr=0.001
):
    """
    Optimize last token embedding to make top-2 probabilities equal
    """
    device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get initial embeddings - keep in bfloat16
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    
    last_token_idx = inputs["input_ids"].shape[1] - 1
    
    # Make embeddings trainable - KEEP IN BFLOAT16
    embeddings = embeddings.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([embeddings], lr=lr)
    
    print(f"Input: '{input_text}'")
    print(f"Last token index: {last_token_idx}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"\nOptimizing...")
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        logits = outputs.logits[0, last_token_idx, :].float()
        
        # Get top-2
        top2_values, top2_indices = torch.topk(logits, k=2)
        
        # Loss: minimize difference between top-2 logits
        loss = (top2_values[0] - top2_values[1]).pow(2)
        
        # Check for NaN or inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf detected at iteration {iteration}")
            print(f"  top2_values: {top2_values}")
            print(f"  logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            break
        
        loss.backward()
        
        # Update only last token embedding
        with torch.no_grad():
            # Zero out gradients for all tokens except last
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
                probs = F.softmax(logits, dim=-1)
                top2_probs = probs[top2_indices]
                token1 = tokenizer.decode([top2_indices[0].item()])
                token2 = tokenizer.decode([top2_indices[1].item()])
                
                print(f"Iter {iteration}: Loss={loss.item():.8f}")
                print(f"  Logit diff: {(top2_values[0] - top2_values[1]).item():.8f}")
                print(f"  Top-2: '{token1}' (p={top2_probs[0].item():.6f}) vs '{token2}' (p={top2_probs[1].item():.6f})")
    
    # Final check
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        logits = outputs.logits[0, last_token_idx, :].float()
        last_hidden = outputs.hidden_states[-1][0, last_token_idx, :].float()
        
        top2_values, top2_indices = torch.topk(logits, k=2)
        probs = F.softmax(logits, dim=-1)
        top2_probs = probs[top2_indices]
        
        token1 = tokenizer.decode([top2_indices[0].item()])
        token2 = tokenizer.decode([top2_indices[1].item()])
        
        print(f"\nFinal Results:")
        print(f"  Logit diff: {(top2_values[0] - top2_values[1]).item():.8f}")
        print(f"  Top-2: '{token1}' (p={top2_probs[0].item():.6f}) vs '{token2}' (p={top2_probs[1].item():.6f})")
    
    # Save optimized embedding
    optimized_last_token_embedding = embeddings[0, last_token_idx, :].detach().cpu()
    last_hidden_cpu = last_hidden.detach().cpu()
    
    return optimized_last_token_embedding, last_hidden_cpu, embeddings, last_token_idx

if __name__ == "__main__":
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    
    model, tokenizer = load_model(MODEL_PATH)
    
    input_text = "The capital of France is"
    
    opt_embedding, last_hidden, full_embeddings, last_idx = optimize_for_equal_top2_probabilities(
        model, tokenizer, input_text,
        num_iterations=500,
        lr=0.001
    )
    
    # Save results
    np.save("optimized_last_token_embedding.npy", opt_embedding.float().numpy())
    np.save("optimized_last_hidden.npy", last_hidden.float().numpy())
    torch.save({
        'full_embeddings': full_embeddings.detach().cpu(),
        'last_token_idx': last_idx,
        'input_text': input_text
    }, "optimized_state.pt")
    
    print("\nSaved:")
    print("  - optimized_last_token_embedding.npy")
    print("  - optimized_last_hidden.npy")
    print("  - optimized_state.pt")