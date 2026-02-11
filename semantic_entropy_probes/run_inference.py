
import os
import pickle
import torch
import numpy as np
import argparse
import faulthandler
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging

# Enable C-level traceback on segfault
faulthandler.enable()

# Enable verbose logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# --- Configuration ---
# Default paths - adjust if necessary
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Updated to base model per user info
# Get the directory where this script is located
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
        # The pickle contains a tuple of dictionaries, one for each dataset the probe was trained on.
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
    # We iterate and pick the token.
    extracted_states = []
    for layer_state in hidden_states:
        # layer_state shape: (batch_size, seq_len, hidden_size)
        # We assume batch_size=1
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
    # Select layers
    # Ensure hidden_states has enough layers
    if start >= hidden_states.shape[0] or end > hidden_states.shape[0]:
        # Fallback or warn? 
        # The probe range might be trained on specific indices.
        # hidden_states[0] is embeddings.
        pass

    selected_states = hidden_states[start:end] # (n_selected, hidden_size)
    
    # Flatten/Concatenate
    input_vec = selected_states.flatten().reshape(1, -1)
    
    # Predict (Binary classification: return probability of class 1)
    # The probe model is a sklearn LogisticRegression
    # class 1 corresponds to "High Entropy" (Uncertainty) usually, or whatever the label was.
    # In the notebook: b_entropy is trained. 1 = High SE.
    probs = probe_model.predict_proba(input_vec)
    return probs[0, 1]

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained Semantic Entropy Probes.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name or path of the HF model.")
    parser.add_argument("--probe_path", type=str, default=PROBE_PATH, help="Path to the probe pickle file.")
    parser.add_argument("--dataset_index", type=int, default=0, help="Index of the dataset probe to use (0=bioasq coverage, etc. depends on save order).")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision (requires bitsandbytes).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision (requires bitsandbytes).")
    
    args = parser.parse_args()
    
    check_memory()

    # 1. Load Probes
    probes = load_probes(args.probe_path)
    if args.dataset_index >= len(probes):
        print(f"Error: dataset_index {args.dataset_index} out of range. Only {len(probes)} probes avail.")
        print("Available probes trained on:", [p['name'] for p in probes])
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

    # Use slow tokenizer to avoid Rust crashes
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Check if GPU available for device_map
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    # Create offload folder for low-RAM loading
    offload_folder = os.path.join(SCRIPT_DIR, "offload")
    os.makedirs(offload_folder, exist_ok=True)
    
    # Constrain memory to force offloading (Critical for 12GB GPU + Low System RAM)
    # Re-check memory before setting constraints
    check_memory()
    max_memory = {
        0: "10GB",      # Reserve ample space on GPU (12GB total)
        "cpu": "1.5GB"    # Lower limit to 1.5GB to be ultra-safe
    }
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=quantization_config,
            # Use strict types to avoid conversion overhead
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # model.eval() # 8-bit/4-bit models are already formatted for inference, calling eval() is sometimes redundant or errored if casted. 
    # But generally safe.
    
    # 3. Process Prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    
    # --- Inference on Prompt (TBG Probe) ---
    # TBG: Token Before Generation (Last token of prompt)
    print("\n--- Running TBG Probe (on Prompt) ---")
    tbg_states = get_hidden_states(model, inputs, token_index=-1)
    
    # Predict Uncertainty (Binary Entropy)
    if 't_bmodel' in selected_probe:
        tbg_se_prob = run_probe(selected_probe['t_bmodel'], tbg_states, selected_probe['sep_layer_range'])
        print(f"Predicted Probability of High Semantic Entropy (Uncertainty): {tbg_se_prob:.4f}")
    
    # Predict Accuracy
    if 't_amodel' in selected_probe:
        tbg_acc_prob = run_probe(selected_probe['t_amodel'], tbg_states, selected_probe['ap_layer_range'])
        print(f"Predicted Probability of Correctness: {tbg_acc_prob:.4f}")


    # --- Inference on Generation (SLT Probe) ---
    # SLT: Second Last Token (Token before EOS)
    print("\n--- Generating Response for SLT Probe ---")
    
    # Generate one token (or more)
    # We generate until EOS or max length.
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False) # Greedy for reproducibility
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: \"{generated_text}\"")
    
    # To get SLT states, we pass the full generated sequence through the model
    # and extract the specific token state.
    # The SLT is the token before the EOS token.
    # If the model didn't emit EOS, it's the last token.
    
    # Find EOS position
    if tokenizer.eos_token_id in generated_ids[0]:
        # Get index of first EOS
        # Note: (generated_ids[0] == tokenizer.eos_token_id).nonzero() ...
        # Simplified: take the last token before EOS.
        # But wait, we need to run the model on the generated_ids to get hidden states.
        pass
    
    # Re-run model on full generated output to get hidden states
    # We want the state of the last generated token (before EOS).
    # If generated_ids includes prompt + new tokens.
    
    seq_len = generated_ids.shape[1]
    
    # Determine SLT index
    # If the last token is EOS, we want the one before it (index -2).
    # If not EOS, we want the last one (index -1).
    last_token_id = generated_ids[0, -1].item()
    if last_token_id == tokenizer.eos_token_id:
        slt_index = seq_len - 2
    else:
         slt_index = seq_len - 1
         
    # Ensure slt_index is valid (e.g. if prompt was empty?)
    
    print(f"Extracting hidden states at token index {slt_index}...")
    full_inputs = {'input_ids': generated_ids, 'attention_mask': torch.ones_like(generated_ids)}
    slt_states = get_hidden_states(model, full_inputs, token_index=slt_index)
    
    # Predict Uncertainty (Binary Entropy)
    if 's_bmodel' in selected_probe:
        slt_se_prob = run_probe(selected_probe['s_bmodel'], slt_states, selected_probe['sep_layer_range'])
        print(f"Predicted Probability of High Semantic Entropy (Uncertainty): {slt_se_prob:.4f}")

    # Predict Accuracy
    if 's_amodel' in selected_probe:
        slt_acc_prob = run_probe(selected_probe['s_amodel'], slt_states, selected_probe['ap_layer_range'])
        print(f"Predicted Probability of Correctness: {slt_acc_prob:.4f}")

if __name__ == "__main__":
    main()
