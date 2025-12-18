import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import csv

def load_model(model_path="/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"):
    """Loads the model on CPU with float64 precision."""
    print("Loading model on CPU with float64 precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float64
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

def get_layer0_hidden_state(model, embeddings, last_token_idx):
    """Gets the hidden state of the first layer (layer 0) for the last token."""
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        # hidden_states[0] is the input embedding, so hidden_states[1] is the output of the first layer.
        return outputs.hidden_states[1][0, last_token_idx, :].numpy()

def analyze_layer0_jumps():
    """Analyzes the effect of small perturbations on the layer 0 hidden state."""
    model, tokenizer = load_model()

    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"]).to(torch.float64)

    last_idx = inputs["input_ids"].shape[1] - 1
    original_input_emb = embeddings[0, last_idx, :].clone()

    print("Computing Jacobian SVD...")
    U, S, Vt = compute_jacobian_svd(model, embeddings, last_idx)
    direction = Vt[0, :]

    # Define perturbation parameters
    e1 = 1e-6 + 1815 * 2e-13
    step_size = 3 * 2e-14
    jumps = list(range(1, 1001))

    counts = []
    max_diffs = []
    emb_counts = []
    emb_max_diffs = []

    print("Generating embeddings for e1...")
    perturbed_emb1 = original_input_emb + e1 * direction
    embeddings1 = embeddings.clone()
    embeddings1[0, last_idx, :] = perturbed_emb1
    rep1 = get_layer0_hidden_state(model, embeddings1, last_idx)

    print("Analyzing jumps...")
    for i, jump in enumerate(jumps):
        e2 = e1 + step_size * jump
        perturbed_emb2 = original_input_emb + e2 * direction
        embeddings2 = embeddings.clone()
        embeddings2[0, last_idx, :] = perturbed_emb2
        rep2 = get_layer0_hidden_state(model, embeddings2, last_idx)

        # Layer 0 hidden state difference
        diff = np.abs(rep1 - rep2)
        count = np.sum(diff > 0)  # Use a small threshold for floating point comparison
        max_diff = np.max(diff)
        counts.append(count)
        max_diffs.append(max_diff)

        # Input embedding difference
        emb_diff = np.abs(perturbed_emb2.cpu().numpy() - perturbed_emb1.cpu().numpy())
        emb_count = np.sum(emb_diff > 0)
        emb_max_diff = np.max(emb_diff)
        emb_counts.append(emb_count)
        emb_max_diffs.append(emb_max_diff)

        if (i + 1) % 100 == 0:
            print(f"Jump {jump}:")
            print(f"  Layer0 - count: {count}, max_diff: {max_diff:.4e}")
            print(f"  Embedding - count: {emb_count}, max_diff: {emb_max_diff:.4e}")
            print(f"  Progress: {i + 1}/{len(jumps)}")

    return np.array(jumps), np.array(counts), np.array(max_diffs), np.array(emb_counts), np.array(emb_max_diffs)

def plot_results(jumps, counts, max_diffs, emb_counts, emb_max_diffs):
    """Plots the results of the analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Layer 0 hidden state plots
    ax1.plot(jumps, counts, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    ax1.set_xlabel('Jump', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count of Changed Values', fontsize=12, fontweight='bold')
    ax1.set_title('Layer 0 Hidden State: Count of Changed Values vs Jump (CPU, float64)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(jumps, max_diffs, 'o-', linewidth=2, markersize=4, color='#A23B72')
    ax2.set_xlabel('Jump', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Absolute Difference', fontsize=12, fontweight='bold')
    ax2.set_title('Layer 0 Hidden State: Max Absolute Difference vs Jump (CPU, float64)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Embedding difference plots
    ax3.plot(jumps, emb_counts, 'o-', linewidth=2, markersize=4, color='#F18F01')
    ax3.set_xlabel('Jump', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count of Changed Values', fontsize=12, fontweight='bold')
    ax3.set_title('Input Embedding: Count of Changed Values vs Jump (CPU, float64)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4.plot(jumps, emb_max_diffs, 'o-', linewidth=2, markersize=4, color='#C73E1D')
    ax4.set_xlabel('Jump', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Max Absolute Difference', fontsize=12, fontweight='bold')
    ax4.set_title('Input Embedding: Max Absolute Difference vs Jump (CPU, float64)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = '../results/exp1_layer0_count_and_maxdiff_vs_jump_cpu_fp64.pdf'
    png_path = '../results/exp1_layer0_count_and_maxdiff_vs_jump_cpu_fp64.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {png_path}")
    plt.show()

def save_results(jumps, counts, max_diffs, emb_counts, emb_max_diffs):
    """Saves the results to JSON and CSV files."""
    data_dict = {
        'jumps': jumps.tolist(),
        'layer0_counts': counts.tolist(),
        'layer0_max_diffs': max_diffs.tolist(),
        'embedding_counts': emb_counts.tolist(),
        'embedding_max_diffs': emb_max_diffs.tolist()
    }

    json_path = '../results/exp1_layer0_results_cpu_fp64.json'
    with open(json_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved results: {json_path}")

    csv_path = '../results/exp1_layer0_results_cpu_fp64.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Jump', 'Layer0_Count', 'Layer0_Max_Diff', 'Embedding_Count', 'Embedding_Max_Diff'])
        writer.writerows(zip(jumps, counts, max_diffs, emb_counts, emb_max_diffs))
    print(f"Saved results: {csv_path}")

if __name__ == "__main__":
    jumps, counts, max_diffs, emb_counts, emb_max_diffs = analyze_layer0_jumps()
    plot_results(jumps, counts, max_diffs, emb_counts, emb_max_diffs)
    save_results(jumps, counts, max_diffs, emb_counts, emb_max_diffs)
