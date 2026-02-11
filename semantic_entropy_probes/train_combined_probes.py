
import pickle
import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression

# Define the runs and their corresponding probe names
RUNS = [
    {
        "id": "jb5cn370",
        "name": "llama3-triviaqa",
        "base_pattern": "run-*jb5cn370*" 
    },
    {
        "id": "wbjigu7q",
        "name": "llama3-squad",
        "base_pattern": "run-*wbjigu7q*"
    }
]

WANDB_DIR = r"d:\Github Repositories\semantic-entropy-probes\diren\uncertainty\wandb"

def find_generations_file(run_config):
    # Find the run directory
    search_path = os.path.join(WANDB_DIR, run_config["base_pattern"])
    dirs = glob.glob(search_path)
    
    if not dirs:
        print(f"Could not find directory for {run_config['base_pattern']}")
        return None
        
    run_dir = dirs[0]
    
    # Try typical paths
    # 1. Direct files/train_generations.pkl
    p1 = os.path.join(run_dir, "files", "train_generations.pkl")
    if os.path.exists(p1): return p1
    
    # 2. Deep nested structure
    # Look for any train_generations.pkl inside this folder recursively
    for root, _, files in os.walk(run_dir):
        if "train_generations.pkl" in files:
            return os.path.join(root, "train_generations.pkl")
            
    return None

def train_single_probe(generations_file, probe_name):
    print(f"Training probe '{probe_name}' from {generations_file}...")
    
    with open(generations_file, 'rb') as f:
        generations = pickle.load(f)
        
    X_slt = []
    y_acc = []
    
    count = 0
    for uid, data in generations.items():
        if 'most_likely_answer' not in data:
            continue
            
        ans = data['most_likely_answer']
        slt_tensor = ans.get('emb_last_tok_before_gen') # SLT
        acc = ans.get('accuracy', 0.0)
        
        if slt_tensor is None:
            continue
            
        # Convert to numpy and flatten
        try:
             slt_flat = slt_tensor.float().numpy().flatten()
        except:
             # Handle case where it might already be numpy or on GPU
             slt_flat = np.array(slt_tensor).flatten()

        X_slt.append(slt_flat)
        y_acc.append(1 if acc > 0.5 else 0)
        count += 1
        
    print(f"  Collected {count} samples.")
    
    if count == 0:
        return None

    X_slt = np.array(X_slt)
    y_acc = np.array(y_acc)
    y_uncertainty = 1 - y_acc
    
    # Train Models
    print(f"  Training {probe_name} Accuracy Probe...")
    s_amodel = LogisticRegression(max_iter=1000).fit(X_slt, y_acc)
    print(f"  Score: {s_amodel.score(X_slt, y_acc):.4f}")
    
    print(f"  Training {probe_name} Uncertainty Probe...")
    s_bmodel = LogisticRegression(max_iter=1000).fit(X_slt, y_uncertainty)
    print(f"  Score: {s_bmodel.score(X_slt, y_uncertainty):.4f}")
    
    return {
        'name': probe_name,
        's_amodel': s_amodel,
        's_bmodel': s_bmodel,
        'sep_layer_range': (0, 1000), 
        'ap_layer_range': (0, 1000)
    }

def main():
    probes_list = []
    
    for run in RUNS:
        gen_file = find_generations_file(run)
        if gen_file:
            probe = train_single_probe(gen_file, run["name"])
            if probe:
                probes_list.append(probe)
        else:
            print(f"Skipping {run['name']} (File not found)")
            
    if not probes_list:
        print("No probes trained!")
        return

    output_path = r"d:\Github Repositories\semantic-entropy-probes\semantic_entropy_probes\models\Llama3-8b_inference.pkl"
    print(f"Saving {len(probes_list)} probes to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(probes_list, f)
        
    print("Done.")

if __name__ == "__main__":
    main()
