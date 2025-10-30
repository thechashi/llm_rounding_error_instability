"""
Experiment 8, Part 1: Submodule Output Generation

This experiment captures intermediate representations from specific submodules
during token-by-token generation. It is designed to provide a granular view
of the model's internal state at each step.

Purpose:
--------
- To collect a dataset of internal model representations for analysis.
- To investigate how representations evolve through specific submodules.
- To provide data for analyzing numerical stability and information flow.

Methodology:
------------
1.  Set seeds for reproducibility.
2.  Load the Llama model.
3.  For each test question:
    a.  Generate text token-by-token.
    b.  Use forward hooks to capture the output of specific submodules at each step.
    c.  Save the captured outputs to .npy files.

Data Captured for each generated token:
---------------------------------------
- Input embeddings.
- Output of the first layer's input layernorm.
- Output of the first layer's post-attention layernorm.
- Output of the last layer (before the final norm).
- Output of the final norm.

Output:
-------
- A timestamped results directory.
- For each question, a subdirectory containing:
  - `input_embeddings.npy`
  - `layer0_input_layernorm_outputs.npy`
  - `layer0_post_attention_layernorm_outputs.npy`
  - `last_layer_before_norm_outputs.npy`
  - `final_norm_outputs.npy`
  - `words.json` (generated tokens and other metadata)
  - `metadata.json` (generation metadata)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path: str):
    """Load model and tokenizer with proper configuration"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer

def generate_and_save_data(
    model,
    tokenizer,
    input_text: str,
    output_dir: Path,
    question_id: int,
    num_tokens: int = 1000,
    seed: int = 42
):
    """
    Generate tokens one by one and collect submodule outputs using hooks.
    """
    set_seed(seed)
    device = next(model.parameters()).device

    # --- Hook setup ---
    activations = {}
    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook

    hooks = []
    # Hook for input embeddings (output of embedding layer)
    hooks.append(model.model.embed_tokens.register_forward_hook(get_hook('input_embeddings')))
    # Hooks for layer 0
    hooks.append(model.model.layers[0].input_layernorm.register_forward_hook(get_hook('layer0_input_layernorm')))
    hooks.append(model.model.layers[0].post_attention_layernorm.register_forward_hook(get_hook('layer0_post_attention_layernorm')))
    # Hook for last layer output (input to final norm)
    hooks.append(model.model.layers[-1].register_forward_hook(get_hook('last_layer_before_norm')))
    # Hook for final norm output
    hooks.append(model.model.norm.register_forward_hook(get_hook('final_norm')))
    
    # --- Storage initialization ---
    data_storage = {
        'input_embeddings': [],
        'layer0_input_layernorm_outputs': [],
        'layer0_post_attention_layernorm_outputs': [],
        'last_layer_before_norm_outputs': [],
        'final_norm_outputs': [],
        'generated_words': [],
        'top_5_words': [],
        'top_10_token_ids': [],
    }

    current_text = input_text
    
    print(f"Generating {num_tokens} tokens...")
    
    for step in range(num_tokens):
        if step % 100 == 0:
            print(f"  Generated {step}/{num_tokens} tokens...")
        
        inputs = tokenizer(
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

        # --- Data extraction from hooks ---
        # The hooks store the full sequence output, we need the one for the last token
        for name, activation in activations.items():
            data_storage[f'{name}_outputs' if name not in data_storage else name].append(activation[0, -1, :].float().cpu().numpy())

        # --- Standard generation data ---
        next_token_logits = outputs.logits[0, -1, :]
        top_10_vals, top_10_indices = torch.topk(next_token_logits, 10)
        
        top_5_indices = top_10_indices[:5]
        top_5_tokens = [tokenizer.decode([idx.item()]) for idx in top_5_indices]
        
        next_token_id = top_10_indices[0].item()
        next_token = tokenizer.decode([next_token_id])

        data_storage['generated_words'].append(next_token)
        data_storage['top_5_words'].append(top_5_tokens)
        data_storage['top_10_token_ids'].append(top_10_indices.cpu().numpy())
        
        current_text += next_token
        
        if next_token_id == tokenizer.eos_token_id:
            print(f"  Stopped early at token {step+1} (EOS token)")
            break

    # --- Remove hooks ---
    for hook in hooks:
        hook.remove()

    print(f"Generation complete! Generated {len(data_storage['generated_words'])} tokens.")

    # --- Save data ---
    question_dir = output_dir / f"question_{question_id:02d}"
    question_dir.mkdir(parents=True, exist_ok=True)

    for key, data_list in data_storage.items():
        if key not in ['generated_words', 'top_5_words', 'top_10_token_ids']:
            np.save(question_dir / f"{key}.npy", np.array(data_list))

    words_data = {
        'top_5_words': data_storage['top_5_words'],
        'generated_words': data_storage['generated_words'],
        'top_10_token_ids': [ids.tolist() for ids in data_storage['top_10_token_ids']]
    }
    with open(question_dir / "words.json", 'w') as f:
        json.dump(words_data, f, indent=2)

    metadata = {
        'question_id': question_id,
        'input_text': input_text,
        'num_tokens_generated': len(data_storage['generated_words']),
        'full_generated_text': current_text,
    }
    with open(question_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved all data to {question_dir}")
    return current_text


def main():
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp8_part1_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    OUTPUT_DIR = Path(exp_dir)
    NUM_TOKENS = 1000
    SEED = 42
    
    questions = [
        "Explain the process of photosynthesis in plants, including the light-dependent and light-independent reactions, and describe how this process is crucial for life on Earth.",
        "Describe the major events and consequences of World War II, including the key battles, political changes, and the impact on global geopolitics in the post-war era.",
        "Explain how the internet works, from the physical infrastructure of fiber optic cables to the protocols like TCP/IP, DNS, and HTTP that enable communication between devices.",
        "Discuss the theory of evolution by natural selection as proposed by Charles Darwin, including evidence from fossil records, comparative anatomy, and modern genetic studies.",
        "Explain the structure and function of the human brain, including the roles of different regions like the cerebral cortex, cerebellum, hippocampus, and how neurons communicate through synapses.",
        "What are the traditional customs and festivals celebrated on the planet Zorblax, and how do the three-headed Zorblaxians prepare their famous crystal soup?",
        "Describe the architectural features of the Underwater City of Atlantis 2.0 that was discovered in 2023, and explain how the ancient magnetic levitation system still works.",
        "What were the main findings of the 2024 expedition to the center of the Earth, and how did scientists manage to survive the temperatures in the crystal caverns?",
        "Explain the quantum telepathy communication system that was invented by Dr. Xenophon Quark in 2022, and how it uses entangled consciousness particles to transmit thoughts.",
        "Describe the historical peace treaty signed between humans and the underground mole civilization in 1847, and what led to the Great Tunneling War that preceded it."
    ]
    
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_model(MODEL_PATH)

    for i, question in enumerate(questions, 1):
        print("\n" + "="*80)
        print(f"PROCESSING QUESTION {i}/{len(questions)}")
        print("="*80)
        print(f"Question: {question[:100]}...")
        print()
        
        full_text = generate_and_save_data(
            model=model,
            tokenizer=tokenizer,
            input_text=question,
            output_dir=OUTPUT_DIR,
            question_id=i,
            num_tokens=NUM_TOKENS,
            seed=SEED
        )
        
        print(f"\nGenerated text preview:")
        print(full_text[:500] + "...")
        print()
    
    print("\n" + "="*80)
    print("ALL QUESTIONS PROCESSED!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    
    print("\nSummary of saved files for each question:")
    print("  - input_embeddings.npy")
    print("  - layer0_input_layernorm_outputs.npy")
    print("  - layer0_post_attention_layernorm_outputs.npy")
    print("  - last_layer_before_norm_outputs.npy")
    print("  - final_norm_outputs.npy")
    print("  - words.json")
    print("  - metadata.json")

if __name__ == "__main__":
    main()
