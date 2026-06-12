"""
Fix probe pickle files that were saved with numpy 2.x but need to load under numpy 1.x.

The issue: numpy 2.x moved `numpy.core` → `numpy._core`. Pickle files store the full
module path, so a file saved with numpy 2.x references `numpy._core.numeric` which
doesn't exist in numpy 1.x.

The fix: We patch the unpickler to redirect `numpy._core` → `numpy.core` and then
re-save the file so it works permanently.
"""
import pickle
import io
import sys
import os
import shutil


class NumpyCompatUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps numpy._core -> numpy.core for numpy 1.x compat."""
    def find_class(self, module, name):
        # Remap numpy 2.x internal paths to numpy 1.x paths
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def fix_pickle_file(filepath):
    backup = filepath + ".bak"
    
    print(f"Fixing: {filepath}")
    print(f"  Creating backup at: {backup}")
    shutil.copy2(filepath, backup)
    
    # Load with compatibility shim
    print(f"  Loading with numpy._core -> numpy.core remapping...")
    with open(filepath, 'rb') as f:
        data = NumpyCompatUnpickler(f).load()
    
    # Re-save — this writes with the CURRENT numpy's module paths
    print(f"  Re-saving with current numpy ({__import__('numpy').__version__})...")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    # Verify it loads cleanly now
    print(f"  Verifying clean load...")
    with open(filepath, 'rb') as f:
        _ = pickle.load(f)
    
    print(f"  ✓ Fixed successfully!")


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "semantic_entropy_probes", "models")
    
    target = os.path.join(models_dir, "Llama3-8b_inference.pkl")
    
    if not os.path.exists(target):
        print(f"ERROR: File not found: {target}")
        sys.exit(1)
    
    fix_pickle_file(target)
    print("\nDone! You can now switch to Llama 3 in the web app without errors.")
