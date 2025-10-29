"""
Core Utility Module for LLM Rounding Error Instability Research

This module provides essential utility functions for loading and interacting with
Large Language Models (LLMs) in the context of numerical precision and instability
research.

Purpose:
--------
Provides standardized functions for:
- Loading LLMs with specific precision configurations (bfloat16)
- Extracting hidden representations and logits from models
- Generating text with controlled parameters
- Token-by-token generation with logit tracking

Related Files:
--------------
- notebooks/utils.py: Identical copy of this file for notebook experiments
- notebooks/project_utils.py: Extended version with JSON config loading (depends on this file's functions)
- Most experiment files (experiment*.py) use these utilities for model loading
- experiment6_part1_optimize_equal_probs.py: Uses model loading functions
- experiment7_logit_maps.py: Uses model loading and hidden state extraction

Dependencies:
-------------
- torch, transformers (HuggingFace)
- Used by nearly all experiment scripts in the project

Key Functions:
--------------
- load_model(): Loads model and tokenizer with bfloat16 precision
- get_final_representation(): Extracts normalized hidden states for analysis
- get_first_token_logits(): Retrieves next token predictions and probabilities
- generate_response_with_params(): Full-featured text generation with control
- generate_tokens_with_logits(): Step-by-step generation with logit tracking

Note:
-----
This file is duplicated in notebooks/utils.py for convenience in notebook environments.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path: str):
    """Load model and tokenizer with proper configuration"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"  # Important for batch processing
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,  # This is crucial!
        low_cpu_mem_usage=True
    )
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully on device: {next(model.parameters()).device}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer

def generate_response_with_params(
    model, 
    tokenizer, 
    input_text: str,
    max_length: int = 512,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1
) -> Dict[str, Any]:
    """
    Generate response with full parameter control
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        input_text: Input text to generate from
        max_length: Maximum total sequence length
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        do_sample: Whether to use sampling vs greedy decoding
        repetition_penalty: Penalty for repeating tokens
        num_return_sequences: Number of sequences to return
    
    Returns:
        Dictionary with generated text and metadata
    """
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length - max_new_tokens
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    
    # Decode responses
    input_length = inputs['input_ids'].shape[1]
    generated_sequences = []
    
    for i in range(num_return_sequences):
        generated_tokens = outputs.sequences[i][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response = tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
        
        generated_sequences.append({
            'generated_text': generated_text.strip(),
            'full_response': full_response,
            'generated_token_count': len(generated_tokens)
        })
    
    return {
        'input_text': input_text,
        'input_token_count': input_length,
        'generated_sequences': generated_sequences,
        'generation_params': {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'repetition_penalty': repetition_penalty
        }
    }


def get_final_representation(model, tokenizer, input_text: str) -> torch.Tensor:
    """
    Get the final hidden representation that goes into the LM head (with layer norm)
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        input_text: Input text to get representation for
    
    Returns:
        Final hidden state tensor [hidden_size] (NORMALIZED - same as what goes to lm_head)
    """
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get the last hidden state (final layer)
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_states = outputs.hidden_states[-1]
        
        # IMPORTANT: Apply the final layer normalization (this is what actually goes to lm_head!)
        # This was the missing piece - the lm_head receives normalized representations
        normalized_states = model.model.norm(last_hidden_states)
        
        # Get representation of the last token
        final_representation = normalized_states[0, -1, :].cpu().detach().float()
    
    return final_representation


def get_first_token_logits(
    model, 
    tokenizer, 
    input_text: str, 
    return_probabilities: bool = False
) -> Dict[str, Any]:
    """
    Get logits for the first token that would be generated
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        input_text: Input text to get next token logits for
        return_probabilities: Whether to also return probabilities
    
    Returns:
        Dictionary with logits, top tokens, and optionally probabilities
    """
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get logits for the last position (next token prediction)
        # Shape: [batch_size, sequence_length, vocab_size]
        next_token_logits = outputs.logits[0, -1, :].cpu().detach().float()
    
    # Get top tokens and their logits
    top_k = 10
    top_logits, top_indices = torch.topk(next_token_logits, top_k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    
    result = {
        'full_logits': next_token_logits,  # All vocab logits
        'top_logits': top_logits,          # Top k logits
        'top_token_ids': top_indices,      # Top k token IDs
        'top_tokens': top_tokens,          # Top k decoded tokens
        'vocab_size': len(next_token_logits)
    }
    
    if return_probabilities:
        # Convert logits to probabilities
        probabilities = F.softmax(next_token_logits, dim=0)
        top_probs = probabilities[top_indices]
        
        result.update({
            'full_probabilities': probabilities,
            'top_probabilities': top_probs
        })
    
    return result

def generate_tokens_with_logits(model, tokenizer, input_text: str, num_tokens: int = 50) -> Dict[str, Any]:
    """
    Generate tokens one by one and collect logits/probabilities for each step
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        input_text: Input text to generate from
        num_tokens: Number of tokens to generate
    
    Returns:
        Dictionary with tokens, logits, and probabilities for each generation step
    """
    device = next(model.parameters()).device
    
    # Start with the input
    current_text = input_text
    all_logits = []
    all_probabilities = []
    generated_tokens = []
    
    for step in range(num_tokens):
        # Tokenize current text
        inputs = tokenizer(
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get next token logits
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :].cpu().detach().float()
            next_token_probs = F.softmax(next_token_logits, dim=0)
        
        # Store logits and probabilities
        all_logits.append(next_token_logits)
        all_probabilities.append(next_token_probs)
        
        # Get the most likely next token
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id])
        generated_tokens.append(next_token)
        
        # Update current text
        current_text += next_token
        
        # Stop if we hit end of sequence
        if next_token_id == tokenizer.eos_token_id:
            break
    
    return {
        'input_text': input_text,
        'generated_tokens': generated_tokens,
        'all_logits': torch.stack(all_logits),  # [num_generated_tokens, vocab_size]
        'all_probabilities': torch.stack(all_probabilities),  # [num_generated_tokens, vocab_size]
        'final_text': current_text
    }

# Example usage:
if __name__ == "__main__":
    MODEL_PATH = "/home/chashi/Desktop/Research/My Projects/AnyDoor-Center-Attack/models/Llama-3.1-8B-Instruct"
    model, tokenizer = load_model(MODEL_PATH)
    
    # 1. Generate response with custom parameters
    response = generate_response_with_params(
        model, tokenizer,
        "What is the capital of Moon?",
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        top_k=40
    )
    print("Generated:", response['generated_sequences'][0]['generated_text'])
    
    # 2. Get final representation
    representation = get_final_representation(
        model, tokenizer,
        "What is the capital of Moon?"
    )
    print(f"Representation shape: {representation.shape}")
    print(f"Representation norm: {torch.norm(representation):.4f}")
    
    # 3. Get first token logits
    logits_info = get_first_token_logits(
        model, tokenizer,
        "What is the capital of Moon?",
        return_probabilities=True
    )
    print("Top next tokens:")
    for token, logit, prob in zip(
        logits_info['top_tokens'][:5], 
        logits_info['top_logits'][:5],
        logits_info['top_probabilities'][:5]
    ):
        print(f"  '{token}' -> logit: {logit:.4f}, prob: {prob:.4f}")

    # 4. Test the difference (optional - for debugging)
    print("\n=== Testing Layer Norm Effect ===")
    device = next(model.parameters()).device
    inputs = tokenizer("What is the capital of France?", return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Raw representation (old way)
        raw_repr = outputs.hidden_states[-1][0, -1, :].cpu().detach().float()
        
        # Normalized representation (new way)
        normalized_repr = model.model.norm(outputs.hidden_states[-1])[0, -1, :].cpu().detach().float()
        
        print(f"Raw representation norm: {torch.norm(raw_repr):.4f}")
        print(f"Normalized representation norm: {torch.norm(normalized_repr):.4f}")
        print(f"Cosine similarity between them: {torch.cosine_similarity(raw_repr.unsqueeze(0), normalized_repr.unsqueeze(0)).item():.4f}")
        
        # Compare top similarities
        lm_head_weights = model.lm_head.weight.detach().cpu().float().numpy()
        
        # Raw similarities
        raw_sims = torch.cosine_similarity(raw_repr.unsqueeze(0), torch.from_numpy(lm_head_weights))
        raw_top5 = torch.topk(raw_sims, 5)
        
        # Normalized similarities  
        norm_sims = torch.cosine_similarity(normalized_repr.unsqueeze(0), torch.from_numpy(lm_head_weights))
        norm_top5 = torch.topk(norm_sims, 5)
        
        print("\nTop 5 tokens by similarity:")
        print("Raw representation:")
        for i, (sim, idx) in enumerate(zip(raw_top5.values, raw_top5.indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' -> {sim:.4f}")
            
        print("Normalized representation (NEW - should match generation better):")
        for i, (sim, idx) in enumerate(zip(norm_top5.values, norm_top5.indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' -> {sim:.4f}")