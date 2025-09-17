import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Load model and tokenizer
# -----------------------------
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Force everything to CPU, slow but avoids GPU OOM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "cpu"}
)

# -----------------------------
# Get embeddings + logits
# -----------------------------
def get_embeddings_and_logits(model, tokenizer, input_text):
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Token embeddings
        embeddings = model.model.embed_tokens(inputs["input_ids"])
        # Forward pass
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[0, -1, :]  # last-token logits
    
    return embeddings, logits, inputs["input_ids"]

# -----------------------------
# Apply perturbation to ALL tokens
# -----------------------------
def perturb_embeddings_all(embeddings, perturbation_vector):
    perturbed = embeddings.clone()
    perturbed[0] += perturbation_vector  # add to all tokens in sequence
    diff = perturbed - embeddings
    diff_norm = torch.norm(diff).item()
    return perturbed, diff_norm, diff

# -----------------------------
# Main test function
# -----------------------------
def test_perturbation_effects(model, tokenizer, input_text, perturbation_powers=None):
    if perturbation_powers is None:
        perturbation_powers = list(range(-10, 6))  # 2^-10 to 2^5
    
    print(f"\nTesting input: '{input_text}'")
    device = next(model.parameters()).device
    
    # Original embeddings + logits
    original_embeddings, original_logits, input_ids = get_embeddings_and_logits(model, tokenizer, input_text)
    original_token_id = torch.argmax(original_logits).item()
    original_token = tokenizer.decode([original_token_id])
    original_prob = F.softmax(original_logits, dim=-1)[original_token_id].item()
    print(f"Original prediction: '{original_token}' (prob={original_prob:.4f})")
    
    embed_dim = original_embeddings.shape[-1]
    results = []
    
    for power in perturbation_powers:
        magnitude = 2.0 ** power
        
        # perturbation types
        perturbation_types = [
            ("random_uniform", torch.rand(embed_dim, device=device) - 0.5),
            ("random_normal", torch.randn(embed_dim, device=device)),
            ("constant_positive", torch.ones(embed_dim, device=device)),
            ("single_dimension", torch.zeros(embed_dim, device=device))
        ]
        perturbation_types[3][1][0] = 1.0  # single dim
        
        for pert_type, base_vec in perturbation_types:
            vec = base_vec * magnitude
            
            # Apply perturbation to ALL token embeddings
            perturbed_embeddings, diff_norm, diff = perturb_embeddings_all(original_embeddings, vec)
            
            with torch.no_grad():
                pert_outputs = model(inputs_embeds=perturbed_embeddings)
                pert_logits = pert_outputs.logits[0, -1, :]
            
            new_token_id = torch.argmax(pert_logits).item()
            new_token = tokenizer.decode([new_token_id])
            new_prob = F.softmax(pert_logits, dim=-1)[new_token_id].item()
            
            token_changed = (new_token_id != original_token_id)
            logit_diff = torch.norm(pert_logits - original_logits).item()
            
            results.append({
                "power": power,
                "magnitude": magnitude,
                "perturbation_type": pert_type,
                "original_token": original_token,
                "new_token": new_token,
                "token_changed": token_changed,
                "original_prob": original_prob,
                "new_prob": new_prob,
                "logit_diff": logit_diff,
                "embedding_diff": diff_norm,
            })
            
            if token_changed:
                print(f"  2^{power} {pert_type}: '{original_token}' → '{new_token}' (prob={new_prob:.4f})")
    
    return pd.DataFrame(results)
