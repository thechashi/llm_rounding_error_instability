"""
Experiment 4: GPU Hardware Comparison for Numerical Instability

This experiment tests whether different GPU hardware produces different outputs
for the SAME model and input, revealing hardware-specific numerical differences
due to floating-point arithmetic implementation variations.

Purpose:
--------
Empirically tests the hypothesis that identical models running on different GPUs
can produce different outputs due to:
1. Hardware-specific floating-point implementations
2. Different rounding modes in GPU arithmetic units
3. Variations in matrix multiplication algorithms (cuBLAS implementations)
4. Compiler optimizations that affect numerical precision


Relationship:
-------------
This is the DATA GENERATION script for experiment4 series:
1. experiment4_GPU_comaprison.py (this file): Generates outputs on different GPUs
2. experiment4_part2: Loads and compares the generated outputs
3. experiment4_part3: Identifies divergence points in generation
4. experiment4_part4: Creates visualization plots
5. experiment4_part5-6: Analyzes input embeddings around divergence

Workflow:
---------
1. Run THIS script on GPU 0 → saves results to exp4_gpu0_TIMESTAMP/
2. Run THIS script on GPU 1 → saves results to exp4_gpu1_TIMESTAMP/
3. Run part2 to compare outputs → identifies divergences
4. Run part3-6 for detailed analysis

The comparison reveals whether hardware affects numerical stability, answering:
- Do different GPUs produce identical outputs? (They should, but often don't)
- Where do outputs first diverge?
- How significant are the hardware-induced differences?

Methodology:
------------
1. Set seeds for reproducibility (torch, numpy, CUDA)
2. Load Llama model on specified GPU with bfloat16
3. For each test question:
   a. Generate token-by-token with greedy decoding
   b. Save hidden representations at each step
   c. Save top-10 logits and probabilities
   d. Save generated tokens and top-5 predictions
   e. Save metadata (prompt, full generation, etc.)
4. Save all data to GPU-specific directory

Test Questions:
---------------
Uses diverse prompts to test different generation scenarios:
- Factual questions (e.g., "What is the capital of France?")
- Counterfactual questions (e.g., "What is the capital of Moon?")
- Different topics and complexity levels

Use Case:
---------
Use this experiment to:
- Test if GPU hardware affects model outputs
- Understand hardware-specific numerical behavior
- Identify if instability is partly hardware-dependent
- Establish whether results are reproducible across GPUs

Key Differences from experiment3:
- experiment3: Tests SOFTWARE precision (float32 vs bfloat16 vs float16)
- experiment4: Tests HARDWARE variation (GPU 0 vs GPU 1, same software precision)

Dependencies:
-------------
- torch, transformers (HuggingFace)
- numpy, json, pathlib
- Llama-3.1-8B-Instruct model
- Multiple GPUs for comparison

Key Functions:
--------------
- set_seed(): Ensure reproducibility (critical for GPU comparison)
- load_model(): Load model on specified GPU with bfloat16
- generate_and_save_data(): Token-by-token generation with full logging
- (main): Run experiments and save GPU-specific results

Output:
-------
- Timestamped results directory: results/exp4_gpu{N}_YYYY-MM-DD_HH-MM-SS/
- Per-question subdirectories with:
  * representations.npy: Hidden states at each generation step
  * top_10_logits.npy: Top-10 logit values
  * top_10_probs.npy: Top-10 probabilities
  * words.json: Generated tokens and top-5 predictions
  * metadata.json: Prompt, full generation, GPU info

Note:
-----
CRITICAL: Use same seed, model, and prompts across GPUs for valid comparison.
Any differences in outputs indicate hardware-specific numerical behavior.
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
    # For deterministic behavior
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

def generate_tokens_with_full_data(
    model, 
    tokenizer, 
    input_text: str, 
    num_tokens: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate tokens one by one and collect all required data for each step
    
    Returns:
        Dictionary with:
        - representations: List of final normalized representations [num_tokens, hidden_size]
        - top_10_logits: List of top 10 logits [num_tokens, 10]
        - top_10_probs: List of top 10 probabilities [num_tokens, 10]
        - top_10_token_ids: List of top 10 token IDs [num_tokens, 10]
        - top_5_words: List of top 5 words [num_tokens, 5]
        - generated_words: List of actual generated words [num_tokens]
        - full_text: Complete generated text
    """
    set_seed(seed)
    device = next(model.parameters()).device
    
    # Initialize storage
    representations = []
    top_10_logits = []
    top_10_probs = []
    top_10_token_ids = []
    top_5_words = []
    generated_words = []
    
    # Start with the input
    current_text = input_text
    
    print(f"Generating {num_tokens} tokens...")
    
    for step in range(num_tokens):
        if step % 100 == 0:
            print(f"  Generated {step}/{num_tokens} tokens...")
        
        # Tokenize current text
        inputs = tokenizer(
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Increased for longer contexts
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get next token logits
            next_token_logits = outputs.logits[0, -1, :].cpu().detach().float()
            next_token_probs = F.softmax(next_token_logits, dim=0)
            
            # Get final normalized representation (what goes to lm_head)
            last_hidden_states = outputs.hidden_states[-1]
            normalized_states = model.model.norm(last_hidden_states)
            final_representation = normalized_states[0, -1, :].cpu().detach().float()
            
            # Get top 10 logits and probabilities
            top_10_vals, top_10_indices = torch.topk(next_token_logits, 10)
            top_10_prob_vals = next_token_probs[top_10_indices]
            
            # Get top 5 words
            top_5_indices = top_10_indices[:5]
            top_5_tokens = [tokenizer.decode([idx.item()]) for idx in top_5_indices]
            
            # Get the most likely next token (greedy selection)
            next_token_id = top_10_indices[0].item()
            next_token = tokenizer.decode([next_token_id])
            
        # Store data
        representations.append(final_representation.numpy())
        top_10_logits.append(top_10_vals.numpy())
        top_10_probs.append(top_10_prob_vals.numpy())
        top_10_token_ids.append(top_10_indices.numpy())
        top_5_words.append(top_5_tokens)
        generated_words.append(next_token)
        
        # Update current text
        current_text += next_token
        
        # Stop if we hit end of sequence
        if next_token_id == tokenizer.eos_token_id:
            print(f"  Stopped early at token {step+1} (EOS token)")
            break
    
    print(f"Generation complete! Generated {len(generated_words)} tokens.")
    
    return {
        'input_text': input_text,
        'representations': np.array(representations),  # [num_tokens, hidden_size]
        'top_10_logits': np.array(top_10_logits),      # [num_tokens, 10]
        'top_10_probs': np.array(top_10_probs),        # [num_tokens, 10]
        'top_10_token_ids': np.array(top_10_token_ids), # [num_tokens, 10]
        'top_5_words': top_5_words,                     # List[List[str]]
        'generated_words': generated_words,             # List[str]
        'full_text': current_text
    }

def save_generation_data(data: Dict, output_dir: Path, question_id: int):
    """Save all generation data to separate files"""
    # Create question-specific directory
    question_dir = output_dir / f"question_{question_id:02d}"
    question_dir.mkdir(parents=True, exist_ok=True)
    
    # Save representations
    np.save(
        question_dir / "representations.npy",
        data['representations']
    )
    
    # Save logits
    np.save(
        question_dir / "top_10_logits.npy",
        data['top_10_logits']
    )
    
    # Save probabilities
    np.save(
        question_dir / "top_10_probs.npy",
        data['top_10_probs']
    )
    
    # Save words (JSON for readability)
    words_data = {
        'top_5_words': data['top_5_words'],
        'generated_words': data['generated_words'],
        'top_10_token_ids': data['top_10_token_ids'].tolist()
    }
    with open(question_dir / "words.json", 'w') as f:
        json.dump(words_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'question_id': question_id,
        'input_text': data['input_text'],
        'num_tokens_generated': len(data['generated_words']),
        'full_generated_text': data['full_text'],
        'representation_shape': data['representations'].shape,
        'hidden_size': data['representations'].shape[1] if len(data['representations']) > 0 else 0
    }
    with open(question_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved all data to {question_dir}")

def main():
    # Configuration
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct"

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("../results", f"exp4_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    OUTPUT_DIR = Path(exp_dir)

    NUM_TOKENS = 1000
    SEED = 42
    
    # Questions: 5 factual + 5 hallucination-inducing
    questions = [
        # Factual questions (model can answer confidently)
        "Explain the process of photosynthesis in plants, including the light-dependent and light-independent reactions, and describe how this process is crucial for life on Earth.",
        
        "Describe the major events and consequences of World War II, including the key battles, political changes, and the impact on global geopolitics in the post-war era.",
        
        "Explain how the internet works, from the physical infrastructure of fiber optic cables to the protocols like TCP/IP, DNS, and HTTP that enable communication between devices.",
        
        "Discuss the theory of evolution by natural selection as proposed by Charles Darwin, including evidence from fossil records, comparative anatomy, and modern genetic studies.",
        
        "Explain the structure and function of the human brain, including the roles of different regions like the cerebral cortex, cerebellum, hippocampus, and how neurons communicate through synapses.",
        
        # Hallucination-inducing questions (imaginary/impossible scenarios)
        "What are the traditional customs and festivals celebrated on the planet Zorblax, and how do the three-headed Zorblaxians prepare their famous crystal soup?",
        
        "Describe the architectural features of the Underwater City of Atlantis 2.0 that was discovered in 2023, and explain how the ancient magnetic levitation system still works.",
        
        "What were the main findings of the 2024 expedition to the center of the Earth, and how did scientists manage to survive the temperatures in the crystal caverns?",
        
        "Explain the quantum telepathy communication system that was invented by Dr. Xenophon Quark in 2022, and how it uses entangled consciousness particles to transmit thoughts.",
        
        "Describe the historical peace treaty signed between humans and the underground mole civilization in 1847, and what led to the Great Tunneling War that preceded it."
    ]
    
    # Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_model(MODEL_PATH)

    # Process each question
    for i, question in enumerate(questions, 1):
        print("\n" + "="*80)
        print(f"PROCESSING QUESTION {i}/{len(questions)}")
        print("="*80)
        print(f"Question: {question[:100]}...")
        print()
        
        # Generate tokens with all data
        data = generate_tokens_with_full_data(
            model=model,
            tokenizer=tokenizer,
            input_text=question,
            num_tokens=NUM_TOKENS,
            seed=SEED
        )
        
        # Save data
        save_generation_data(data, OUTPUT_DIR, i)
        
        print(f"\nGenerated text preview:")
        print(data['full_text'][:500] + "...")
        print()
    
    print("\n" + "="*80)
    print("ALL QUESTIONS PROCESSED!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    
    # Print summary
    print("\nSummary of saved files for each question:")
    print("  - representations.npy: [num_tokens, hidden_size] final normalized representations")
    print("  - top_10_logits.npy: [num_tokens, 10] top 10 logits")
    print("  - top_10_probs.npy: [num_tokens, 10] top 10 probabilities")
    print("  - words.json: top 5 words, generated words, and token IDs")
    print("  - metadata.json: question info and generation metadata")

if __name__ == "__main__":
    main()