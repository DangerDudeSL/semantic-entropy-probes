
import os
import pickle
import torch
import numpy as np
import argparse
import faulthandler
import psutil
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging

# Enable C-level traceback on segfault
faulthandler.enable()

# Enable verbose logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# --- Configuration ---
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_PATH = os.path.join(SCRIPT_DIR, "models", "Llama2-7b_inference.pkl")

def check_memory():
    vm = psutil.virtual_memory()
    print(f"[DEBUG] System RAM - Total: {vm.total/1e9:.2f}GB, Available: {vm.available/1e9:.2f}GB")

def load_probes(path):
    """Load the trained probes from the pickle file."""
    print(f"Loading probes from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probe file not found at {path}")
        
    with open(path, 'rb') as f:
        probes = pickle.load(f)
    
    print(f"Loaded probes trained on {len(probes)} datasets.")
    return probes

def get_hidden_states(model, inputs, token_index=-1):
    """
    Extract hidden states from the model.
    Returns a numpy array of shape (num_layers+1, hidden_size).
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    # Stack hidden states: list of (batch, seq, dim) -> (num_layers, batch, seq, dim)
    extracted_states = []
    for layer_state in hidden_states:
        # layer_state shape: (batch_size, seq_len, hidden_size)
        vec = layer_state[0, token_index, :].cpu().numpy()
        extracted_states.append(vec)
        
    return np.stack(extracted_states)

def run_probe(probe_model, hidden_states, layer_range):
    """
    Run the probe model on the hidden states.
    hidden_states: (num_layers, hidden_size)
    layer_range: tuple (start, end)
    """
    start, end = layer_range
    if start >= hidden_states.shape[0] or end > hidden_states.shape[0]:
        pass

    selected_states = hidden_states[start:end] 
    input_vec = selected_states.flatten().reshape(1, -1)
    
    # Predict (Binary classification: return probability of class 1)
    # class 1 corresponds to "High Entropy" (Uncertainty)
    probs = probe_model.predict_proba(input_vec)
    return probs[0, 1]

def main():
    parser = argparse.ArgumentParser(description="Live LLM Interface with Semantic Entropy Probes.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name or path of the HF model.")
    parser.add_argument("--probe_path", type=str, default=PROBE_PATH, help="Path to the probe pickle file.")
    parser.add_argument("--dataset_index", type=int, default=0, help="Index of the dataset probe to use.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision.")
    
    args = parser.parse_args()
    
    check_memory()

    # 1. Load Probes
    try:
        probes = load_probes(args.probe_path)
    except FileNotFoundError:
        print(f"Error: Could not find probe file at {args.probe_path}")
        return

    if args.dataset_index >= len(probes):
        print(f"Error: dataset_index {args.dataset_index} out of range.")
        return

    selected_probe = probes[args.dataset_index]
    print(f"Using probe trained on: {selected_probe['name']}")
    
    # 2. Load Model
    print(f"Loading model {args.model_name}...")
    
    quantization_config = None
    if args.load_in_4bit:
        print("Using 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.load_in_8bit:
        print("Using 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        offload_folder = os.path.join(SCRIPT_DIR, "live_offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        max_memory = {0: "10GB", "cpu": "1.5GB"}
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\n" + "="*50)
    print(" LIVE LLM INTERFACE READY")
    print(" Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n[USER]: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
                
            # Process Prompt - Wrap in template to force answer behavior
            prompt = f"Question: {user_input}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            print("[LLM]: Generating...", end="", flush=True)
            
            # Generate with sampling and repetition penalty
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True,          # Enable sampling for natural text
                    temperature=0.7,         # Creativity
                    top_p=0.9,               # Nucleus sampling
                    repetition_penalty=1.2   # Prevent loops
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract just the new response part
            response_text = generated_text[len(prompt):]
            
            print(f"\r[LLM]: {response_text.strip()}")

            # Calculate Uncertainty (SLT - Second Last Token / Last Token)
            seq_len = generated_ids.shape[1]
            last_token_id = generated_ids[0, -1].item()
            if last_token_id == tokenizer.eos_token_id:
                slt_index = seq_len - 2
            else:
                slt_index = seq_len - 1
            
            # If the response is empty or just EOS (slt_index points to prompt), we might have issues
            prompt_len = inputs['input_ids'].shape[1]
            if slt_index < prompt_len:
                 # Try to fallback to last token of prompt if generation failed, but usually better to skip
                 print(" (Response too short to measure uncertainty)")
                 continue

            # Get hidden states for the full sequence
            full_inputs = {'input_ids': generated_ids, 'attention_mask': torch.ones_like(generated_ids)}
            slt_states = get_hidden_states(model, full_inputs, token_index=slt_index)
            
            uncertainty_score = 0.0
            accuracy_score = 0.0
            
            if 's_bmodel' in selected_probe:
                uncertainty_score = run_probe(selected_probe['s_bmodel'], slt_states, selected_probe['sep_layer_range'])
                
            if 's_amodel' in selected_probe:
                accuracy_score = run_probe(selected_probe['s_amodel'], slt_states, selected_probe['ap_layer_range'])

            print(f"\n[ANALYSIS]")
            print(f" Uncertainty (Semantic Entropy): {uncertainty_score:.4f}  (>0.5 = High Uncertainty)")
            # print(f" Predicted Correctness:          {accuracy_score:.4f}") 

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during inference: {e}")

if __name__ == "__main__":
    main()
