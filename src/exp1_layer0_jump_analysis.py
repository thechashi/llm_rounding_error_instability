import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def get_layer0_output(model, embeddings, last_token_idx):
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        return outputs.hidden_states[0][0, last_token_idx, :].cpu().numpy()

if __name__ == "__main__":
    text = "The capital of France is"
    e1 = 1e-6 + 1815*2e-13
    step_size = 3*2e-14
    singular_idx = 0

    print("Loading model...")
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    print("Tokenizing input...")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])

    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()

    print("Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    direction = Vt[singular_idx, :]

    print(f"Getting layer 0 output for e1={e1:.15e}...")
    perturbed_emb1 = original_input_emb + e1 * direction
    embeddings1 = embeddings.clone()
    embeddings1[0, last_idx, :] = perturbed_emb1
    layer0_e1 = get_layer0_output(model, embeddings1, last_idx)

    print("Analyzing jumps 1 to 1000...")
    jumps = list(range(1, 1001))
    counts = []
    max_diffs = []

    for jump in jumps:
        e2 = e1 + step_size * jump
        perturbed_emb2 = original_input_emb + e2 * direction
        embeddings2 = embeddings.clone()
        embeddings2[0, last_idx, :] = perturbed_emb2
        layer0_e2 = get_layer0_output(model, embeddings2, last_idx)

        diff = np.abs(layer0_e1 - layer0_e2)
        count_changed = np.sum(diff > 0)
        max_diff = np.max(diff)

        counts.append(count_changed)
        max_diffs.append(max_diff)

        if jump % 100 == 0:
            print(f"Jump {jump}: {count_changed} changed, max_diff={max_diff:.6e}")

    print("\nPlotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(jumps, counts, linewidth=1.5)
    ax1.set_xlabel('Jump')
    ax1.set_ylabel('Number of Changed Values')
    ax1.set_title('Count of Changed Values vs Jump')
    ax1.grid(True, alpha=0.3)

    ax2.plot(jumps, max_diffs, linewidth=1.5, color='red')
    ax2.set_xlabel('Jump')
    ax2.set_ylabel('Max Absolute Difference')
    ax2.set_title('Max Absolute Difference vs Jump')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp1_layer0_jump_analysis.pdf', dpi=300, bbox_inches='tight')
    print("Saved: exp1_layer0_jump_analysis.pdf")
    plt.show()
