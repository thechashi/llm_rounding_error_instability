
import csv

def run_iterative_perturbation_analysis(num_steps, e1, step_size, singular_idx, text, threshold, save_prefix):
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
            
            # Perform perturbation calculation in float64 for higher precision
            perturbed_emb2_64 = original_input_emb.double() + float(e2) * direction.double()
            
            # Cast back to float32 before passing to the model
            perturbed_emb2 = perturbed_emb2_64.float()
            
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
    NUM_STEPS = 20000
    E1 = 1e-6 
    STEP_SIZE = 3e-14
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
        save_prefix=os.path.join(exp_dir, "iterative_perturbation")
    )

- experiment5: Computes separate Jacobian/SVD for each layer, tests layer-
  specific directions