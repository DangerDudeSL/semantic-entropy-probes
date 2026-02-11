
import pickle
import numpy as np
import os
import argparse
from sklearn.linear_model import LogisticRegression

def train_probes(generations_file, output_path, layer_index=-1):
    print(f"Loading generations from {generations_file}...")
    
    # Handle potential list of files if needed, but for now single file
    with open(generations_file, 'rb') as f:
        generations = pickle.load(f)
    
    print(f"Loaded {len(generations)} examples.")
    
    X_tbg = []
    X_slt = []
    y_acc = []
    
    # Inspect first item to determine structure/layers
    first_key = next(iter(generations))
    first_gen = generations[first_key]
    
    if 'most_likely_answer' not in first_gen:
        print("Error: 'most_likely_answer' key missing in generations.")
        return

    # Check embedding shapes
    # In generate_answers.py:
    # 'emb_tok_before_eos' -> TBG (Token Before Generation)
    # 'emb_last_tok_before_gen' -> SLT (Second Last Token / Token Before EOS)
    
    # They are tensors, possibly on CPU.
    # Shape: (num_layers, hidden_size) or (num_layers, 1, hidden_size)?
    
    # Let's flatten for safety if unknown, or use specific layer.
    
    count = 0
    for uid, data in generations.items():
        if 'most_likely_answer' not in data:
            continue
            
        ans = data['most_likely_answer']
        
        # Extract Embeddings
        # Note: variable naming in generate_answers might be confused, but positionally:
        # 3rd return (emb_before_eos) -> TBG
        # 2nd return (emb_last_before_gen) -> SLT
        
        # Keys in dict:
        tbg_tensor = ans.get('emb_tok_before_eos') # This corresponds to TBG
        slt_tensor = ans.get('emb_last_tok_before_gen') # This corresponds to SLT
        acc = ans.get('accuracy', 0.0)
        
        if tbg_tensor is None or slt_tensor is None:
            continue
            
        # Convert to numpy and flatten/select layer
        # If we want to use ALL layers flattened:
        tbg_flat = tbg_tensor.float().numpy().flatten()
        slt_flat = slt_tensor.float().numpy().flatten()
        
        X_tbg.append(tbg_flat)
        X_slt.append(slt_flat)
        y_acc.append(1 if acc > 0.5 else 0)
        count += 1
        
    print(f"Collected {count} valid samples.")
    
    X_tbg = np.array(X_tbg)
    X_slt = np.array(X_slt)
    y_acc = np.array(y_acc)
    y_uncertainty = 1 - y_acc # 1 for Incorrect/Uncertain
    
    # Train Models
    print("Training TBG Accuracy Probe...")
    t_amodel = LogisticRegression(max_iter=1000).fit(X_tbg, y_acc)
    print(f"TBG Accuracy Train Score: {t_amodel.score(X_tbg, y_acc):.4f}")
    
    print("Training SLT Accuracy Probe...")
    s_amodel = LogisticRegression(max_iter=1000).fit(X_slt, y_acc)
    print(f"SLT Accuracy Train Score: {s_amodel.score(X_slt, y_acc):.4f}")
    
    print("Training TBG Uncertainty Probe...")
    t_bmodel = LogisticRegression(max_iter=1000).fit(X_tbg, y_uncertainty)
    print(f"TBG Uncertainty Train Score: {t_bmodel.score(X_tbg, y_uncertainty):.4f}")
    
    print("Training SLT Uncertainty Probe...")
    s_bmodel = LogisticRegression(max_iter=1000).fit(X_slt, y_uncertainty)
    print(f"SLT Uncertainty Train Score: {s_bmodel.score(X_slt, y_uncertainty):.4f}")
    
    # Determine layer count for metadata
    # Assuming standard shape (num_layers, ...)
    # If flattened, we just say range is full.
    # Ideally we should store the shape to unflatten if needed, but run_inference flattens too.
    
    # run_inference expects specific keys
    probe_dict = {
        'name': 'custom_llama3_probe',
        't_amodel': t_amodel,
        's_amodel': s_amodel,
        't_bmodel': t_bmodel,
        's_bmodel': s_bmodel,
        'sep_layer_range': (0, 1000), # Covers all
        'ap_layer_range': (0, 1000)
    }
    
    # Save as list of dicts
    probes_list = [probe_dict]
    
    print(f"Saving probes to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(probes_list, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations_file", type=str, required=True, help="Path to generations.pkl file")
    parser.add_argument("--output_path", type=str, default="probes.pkl", help="Path to save trained probes")
    args = parser.parse_args()
    
    train_probes(args.generations_file, args.output_path)
