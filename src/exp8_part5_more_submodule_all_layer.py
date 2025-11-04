"""
Experiment 8, Part 5: Comprehensive Submodule Output Generation

This experiment captures intermediate representations from ALL submodules
across ALL layers during token-by-token generation.

Purpose:
--------
- To collect a comprehensive dataset of internal model representations.
- To investigate how representations evolve through each submodule in each layer.
- To provide data for analyzing numerical stability and information flow.

Methodology:
------------
1.  Set seeds for reproducibility.
2.  Load the Llama model.
3.  For each test question:
    a.  Generate text token-by-token.
    b.  Use forward hooks to capture the output of ALL submodules at each step.
    c.  Save the captured outputs to .npy files.

Data Captured for each generated token:
---------------------------------------
For each layer (0 to N-1):
- input_layernorm output
- self_attn output
- attn.o_proj output
- post_attention_layernorm output
- mlp.gate_proj output
- mlp.up_proj output
- mlp.act_fn output (intermediate activation)
- mlp.down_proj output
- mlp (full MLP block) output

Plus:
- Input embeddings
- Last layer output (before final norm)
- Final norm output

Output:
-------
- A timestamped results directory.
- For each question, a subdirectory containing:
  - .npy files for each submodule at each layer
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
        torch_dtype=torch.float32,
        trust_remote_code=True,
        output_hidden_states=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer

def setup_hooks(model):
    """
    Set up hooks for all submodules across all layers.
    Returns: activations dict, hooks list, data_storage dict
    """
    activations = {}
    hooks = []
    data_storage = {}
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook
    
    # Hook for input embeddings
    hooks.append(model.model.embed_tokens.register_forward_hook(get_hook('input_embeddings')))
    data_storage['input_embeddings'] = []
    
    # Get number of layers
    num_layers = len(model.model.layers)
    print(f"Setting up hooks for {num_layers} layers...")
    
    # Hook each layer's submodules
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        # Input layernorm
        hook_name = f'layer{layer_idx}_input_layernorm'
        hooks.append(layer.input_layernorm.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Self attention (full block)
        hook_name = f'layer{layer_idx}_self_attn'
        hooks.append(layer.self_attn.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Attention output projection
        hook_name = f'layer{layer_idx}_attn_o_proj'
        hooks.append(layer.self_attn.o_proj.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Post-attention layernorm
        hook_name = f'layer{layer_idx}_post_attention_layernorm'
        hooks.append(layer.post_attention_layernorm.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # MLP submodules
        # Gate projection
        hook_name = f'layer{layer_idx}_mlp_gate_proj'
        hooks.append(layer.mlp.gate_proj.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Up projection
        hook_name = f'layer{layer_idx}_mlp_up_proj'
        hooks.append(layer.mlp.up_proj.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Activation function (this is tricky - we'll hook the act_fn if it exists as a module)
        # Note: In some architectures, act_fn is just a function, not a module
        # We'll try to hook it, but may need special handling
        if hasattr(layer.mlp, 'act_fn') and isinstance(layer.mlp.act_fn, torch.nn.Module):
            hook_name = f'layer{layer_idx}_mlp_act_fn'
            hooks.append(layer.mlp.act_fn.register_forward_hook(get_hook(hook_name)))
            data_storage[hook_name] = []
        
        # Down projection
        hook_name = f'layer{layer_idx}_mlp_down_proj'
        hooks.append(layer.mlp.down_proj.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
        
        # Full MLP block
        hook_name = f'layer{layer_idx}_mlp'
        hooks.append(layer.mlp.register_forward_hook(get_hook(hook_name)))
        data_storage[hook_name] = []
    
    # Hook for last layer output (before final norm)
    hook_name = 'last_layer_before_norm'
    hooks.append(model.model.layers[-1].register_forward_hook(get_hook(hook_name)))
    data_storage[hook_name] = []
    
    # Hook for final norm
    hook_name = 'final_norm'
    hooks.append(model.model.norm.register_forward_hook(get_hook(hook_name)))
    data_storage[hook_name] = []
    
    # Add metadata storage
    data_storage['generated_words'] = []
    data_storage['top_5_words'] = []
    data_storage['top_10_token_ids'] = []
    
    print(f"Total hooks registered: {len(hooks)}")
    
    return activations, hooks, data_storage

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
    Generate tokens one by one and collect ALL submodule outputs using hooks.
    """
    set_seed(seed)
    device = next(model.parameters()).device

    # --- Hook setup ---
    activations, hooks, data_storage = setup_hooks(model)
    
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
            if name in data_storage:
                data_storage[name].append(activation[0, -1, :].float().cpu().numpy())

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

    print(f"Saving data to {question_dir}...")
    saved_files = []
    
    for key, data_list in data_storage.items():
        if key not in ['generated_words', 'top_5_words', 'top_10_token_ids']:
            filename = f"{key}.npy"
            np.save(question_dir / filename, np.array(data_list))
            saved_files.append(filename)

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
        'saved_files': saved_files,
        'num_layers': len([k for k in saved_files if 'layer0_' in k])
    }
    with open(question_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {len(saved_files)} .npy files to {question_dir}")
    return current_text


def main():
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp8_part5_comprehensive_float32_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    OUTPUT_DIR = Path(exp_dir)
    NUM_TOKENS = 200
    SEED = 42
    
    questions = [
        "What were the main findings of the 2024 expedition to the center of the Earth, and how did scientists manage to survive the temperatures in the crystal caverns?",
        "Explain the quantum telepathy communication system that was invented by Dr. Xenophon Quark in 2022, and how it uses entangled consciousness particles to transmit thoughts.",
        "Describe the historical peace treaty signed between humans and the underground mole civilization in 1847, and what led to the Great Tunneling War that preceded it."
    ]
    
    print("="*80)
    print("COMPREHENSIVE SUBMODULE DATA COLLECTION")
    print("="*80)
    print("This will capture outputs from ALL submodules in ALL layers")
    print("="*80)
    print("\nLOADING MODEL")
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
    
    print("\nFor each question, saved files include:")
    print("  - input_embeddings.npy")
    print("  For each layer L (0 to N-1):")
    print("    - layerL_input_layernorm.npy")
    print("    - layerL_self_attn.npy")
    print("    - layerL_attn_o_proj.npy")
    print("    - layerL_post_attention_layernorm.npy")
    print("    - layerL_mlp_gate_proj.npy")
    print("    - layerL_mlp_up_proj.npy")
    print("    - layerL_mlp_act_fn.npy (if available)")
    print("    - layerL_mlp_down_proj.npy")
    print("    - layerL_mlp.npy")
    print("  - last_layer_before_norm.npy")
    print("  - final_norm.npy")
    print("  - words.json")
    print("  - metadata.json")

if __name__ == "__main__":
    main()