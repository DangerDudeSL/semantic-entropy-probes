# Windows Setup Guide for Semantic Entropy Probes

This guide will help you set up and run the Semantic Entropy Probes repository on Windows.

## Prerequisites

1. **Python 3.11** - Download from [python.org](https://www.python.org/downloads/)
2. **Conda/Miniconda** - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
3. **Git** - For cloning the repository (if not already cloned)
4. **GPU with CUDA support** (optional but recommended) - The code is designed for GPU inference

## Step 1: Install Conda

1. Download Miniconda for Windows from: https://docs.conda.io/en/latest/miniconda.html
2. Run the installer and follow the prompts
3. During installation, check "Add Miniconda3 to PATH" if available

## Step 2: Set Up the Environment

### Open Anaconda Prompt or PowerShell

Press `Win + X` and select "Windows PowerShell" or "Terminal", or search for "Anaconda Prompt".

### Navigate to the Repository

```powershell
cd "D:\Github Repositories\semantic-entropy-probes"
```

### Create/Update Conda Environment

**Important**: The original `sep_enviroment.yaml` contains Linux-specific package builds that won't work on Windows. Use the Windows-compatible version instead:

```powershell
conda env update -f sep_enviroment_windows.yaml
```

**Alternative**: If you prefer to create from scratch:

```powershell
conda env create -f sep_enviroment_windows.yaml
```

This will create the `se_probes` conda environment with all necessary dependencies optimized for Windows.

**Note**: The Windows environment file uses `conda-forge` channel which has better Windows support and omits Linux-specific packages that aren't needed on Windows.

### Activate the Environment

```powershell
conda activate se_probes
```

## Step 3: Set Up Environment Variables

You need to set some environment variables for the code to work properly:

**Required:**
- `HUGGING_FACE_HUB_TOKEN` - For downloading models from Hugging Face

**Optional but recommended:**
- `WANDB_ENT` or `WANDB_SEM_UNC_ENTITY` - Your Weights & Biases entity/username (or use `--entity` command line argument)
- `OPENAI_API_KEY` - Only if using GPT models (not needed for basic usage)

### Option A: Set via PowerShell (Session-specific)

Open PowerShell and run:

```powershell
$env:HUGGING_FACE_HUB_TOKEN = "your-huggingface-token"
$env:WANDB_ENT = "your-wandb-username"  # Optional: or use --entity argument
$env:OPENAI_API_KEY = "your-openai-api-key"  # Optional: only if using GPT models
```

### Option B: Set Permanently (Recommended)

1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables" (or "System variables"), click "New"
5. Add the following variables:
   - `HUGGING_FACE_HUB_TOKEN` = your Hugging Face token (REQUIRED - get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
   - `WANDB_ENT` = your Weights & Biases username (OPTIONAL - or use `--entity` command line argument)
   - `OPENAI_API_KEY` = your OpenAI API key (OPTIONAL - only if using GPT models)

### Option C: Create a `.env` File (Not used by default, but you can modify scripts)

You can also create a `.env` file in the root directory, but you'll need to load it manually or use a library like `python-dotenv`.

### Getting Your Tokens

1. **Weights & Biases (W&B)**:
   - Sign up at [wandb.ai](https://wandb.ai)
   - Your entity name is usually your username
   - Get your API key from [wandb.ai/settings](https://wandb.ai/settings)

2. **Hugging Face**:
   - Sign up at [huggingface.co](https://huggingface.co)
   - Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - For LLaMA models, you need to [request access](https://huggingface.co/meta-llama) first

3. **OpenAI** (Optional - NOT required for basic usage):
   - **You do NOT need this** for the basic example (`--model_name=Llama-2-7b-chat`)
   - Only needed if you want to use GPT models for long-form generation or specific metrics
   - If you want to use it: Sign up at [platform.openai.com](https://platform.openai.com)
   - Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - **Cost warning**: Using GPT models costs money (estimated $10-100 for full experiments)

## Step 4: Prepare Dataset Files (if needed)

Some datasets require manual download:

### BioASQ Dataset

1. Download from: http://participants-area.bioasq.org/datasets/
   - Use training 11b: `training11b.json`
2. Create the directory structure:
   ```powershell
   mkdir $env:USERPROFILE\uncertainty\data\bioasq
   ```
3. Copy `training11b.json` to: `%USERPROFILE%\uncertainty\data\bioasq\training11b.json`

### ReCoRD Dataset

1. Download from the ReCoRD dataset repository
2. Create the directory:
   ```powershell
   mkdir $env:USERPROFILE\uncertainty\data\record
   ```
3. Place `train.json` and `dev.json` in: `%USERPROFILE%\uncertainty\data\record\`

**Note**: The paths use `~` which expands to your user home directory on Windows (e.g., `C:\Users\YourUsername\`).

## Step 5: Install Visual C++ Redistributable (Required for PyTorch)

**Important**: PyTorch on Windows requires Visual C++ Redistributable libraries. Install this BEFORE testing PyTorch:

1. Download "Microsoft Visual C++ Redistributable for Visual Studio 2015-2022" (64-bit)
   - Direct link: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search for "VC++ Redistributable" on Microsoft's website
2. Run the installer and follow the prompts
3. Restart your terminal/PowerShell after installation

## Step 6: Verify Installation

Test that everything is working:

```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see:
- PyTorch version: 2.1.1 (or similar)
- CUDA available: True (if you have a compatible GPU)

**If you get an error about missing DLLs (fbgemm.dll, etc.)**: See the troubleshooting section below for the Visual C++ Redistributable fix.

## Step 6: Run Your First Experiment

### Basic Command (from repository root)

```powershell
python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

**If you haven't set WANDB_ENT environment variable**, you can pass it as a command-line argument:

```powershell
python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa --entity=your-wandb-username
```

**Note**: You may need to adjust the path to `generate_answers.py` based on your working directory. If you're in the root directory, use the path above. If you're in `semantic_uncertainty`, use:

```powershell
python generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

### Using the Windows Batch Script

A Windows-compatible script is provided at `windows/run_experiments.bat`. See the file for usage.

## Step 7: Training Semantic Entropy Probes

After generating answers, you can train probes using the Jupyter notebook:

1. Start Jupyter:
   ```powershell
   jupyter notebook
   ```
2. Open `semantic_entropy_probes/train-latent-probe.ipynb`
3. Update the `wandb_run_ids` with the IDs from your runs
4. Execute all cells

## Troubleshooting

### Issue: "USER" environment variable not found

**Fixed**: The code now automatically uses `USERNAME` on Windows. This should no longer be an issue.

### Issue: PackagesNotFoundError (e.g., "gmp=6.2.1" not available)

**Cause**: The original `sep_enviroment.yaml` contains Linux-specific package builds that don't exist on Windows.

**Solution**: Use `sep_enviroment_windows.yaml` instead, which:
- Uses `conda-forge` channel (better Windows support)
- Removes Linux-specific packages (like `gmp`, `libgcc-ng`, etc.)
- Focuses on core dependencies that work cross-platform
- Lets conda resolve Windows-compatible builds automatically

If you still encounter package errors:
1. Try installing from conda-forge explicitly:
   ```powershell
   conda install -c conda-forge <package-name>
   ```
2. Some packages may be available via pip instead - check if pip installation works
3. Some low-level system libraries may not be needed on Windows

### Issue: Path not found errors

**Solution**: Ensure you're using forward slashes `/` or `os.path.join()` in paths. The code has been updated to handle Windows paths properly using `os.path.expanduser()`.

### Issue: OSError [WinError 182] loading fbgemm.dll or other PyTorch DLLs

**Cause**: Missing Visual C++ Redistributable libraries or incompatible PyTorch build.

**Solution**:
1. **Install Visual C++ Redistributable** (Most common fix):
   - Download and install "Microsoft Visual C++ Redistributable for Visual Studio 2015-2022" (64-bit)
   - Get it from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - After installation, restart your terminal and try again

2. **Reinstall PyTorch** (if step 1 doesn't work):
   ```powershell
   conda activate se_probes
   conda uninstall pytorch torchvision torchaudio pytorch-cuda -y
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Try CPU-only version first** (to test if it's a CUDA issue):
   ```powershell
   conda activate se_probes
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

4. **Check your system architecture**: Ensure you're using 64-bit Python and PyTorch:
   ```powershell
   python -c "import sys; print(sys.maxsize > 2**32)"
   ```
   Should output `True` for 64-bit.

### Issue: CUDA/GPU not detected

**Solutions**:
1. Verify you have an NVIDIA GPU with CUDA support
2. Check CUDA installation: `nvidia-smi` in PowerShell
3. Reinstall PyTorch with CUDA support if needed:
   ```powershell
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### Issue: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'

**Cause**: Version mismatch between `pyarrow` and `datasets`. The `datasets` library requires a newer version of `pyarrow` (>=11.0.0) that includes `PyExtensionType`, but either pyarrow isn't installed or an older version is installed.

**Solution**: Install or upgrade pyarrow to a compatible version.

**In Git Bash (which you're using)**:

The issue is that newer pyarrow versions (like 22.0.0) may have removed or changed `PyExtensionType`. Use the exact version from the original environment:

```bash
conda activate se_probes
conda install pyarrow=11.0.0 -c conda-forge -y
```

**Note**: We pin to `pyarrow=11.0.0` specifically because `datasets 2.12.0` was tested with this version. Newer versions of pyarrow may have API changes.

**Or use the provided fix script**:
```bash
bash windows/fix_pyarrow.sh
```

**Verify it worked**:
```bash
python -c "import pyarrow as pa; print('PyExtensionType available:', hasattr(pa, 'PyExtensionType'))"
```
Should output: `PyExtensionType available: True`

**Note**: The updated `sep_enviroment_windows.yaml` now includes `pyarrow>=11.0.0`, so if you recreate the environment, this should be automatic.

### Issue: Module not found errors

**Solution**: Ensure the conda environment is activated:
```powershell
conda activate se_probes
```

### Issue: Permission errors when writing files

**Solution**: Run PowerShell as Administrator, or ensure you have write permissions to the directories.

### Issue: Hugging Face model access denied

**Solution**: 
1. Make sure you've requested access to LLaMA models if needed
2. Login to Hugging Face:
   ```powershell
   huggingface-cli login
   ```
   Enter your token when prompted.

### Issue: W&B authentication errors

**Solution**: Login to W&B:
```powershell
wandb login
```
Enter your API key when prompted.

### Issue: W&B Error 403: Permission denied

**Cause**: Entity mismatch - the code is trying to use a different entity name than your logged-in wandb account.

**Solution**: Make sure the entity matches your wandb username. You can:
1. Set the correct entity:
   ```bash
   export WANDB_ENT="your-actual-wandb-username"
   ```
2. Or use the `--entity` argument:
   ```bash
   python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa --entity=your-actual-wandb-username
   ```
3. The entity should match what you see when you run `wandb login` (e.g., if it says "Currently logged in as: direndrakavindu", use `direndrakavindu` as the entity)

### Issue: Git ownership warnings (non-fatal)

**Cause**: Git detects the repository is owned by a different user (Windows/Git Bash quirk).

**Solution**: Add the repository to git's safe directory list:
```bash
git config --global --add safe.directory "$(pwd)"
```

Or use the fix script:
```bash
bash windows/fix_git_ownership.sh
```

This is just a warning and won't stop the code from running, but it's annoying.

## Differences from Linux Setup

1. **Environment Variables**: Windows uses `USERNAME` instead of `USER` (now handled automatically)
2. **Path Separators**: Forward slashes `/` work in Python on Windows, but `os.path.expanduser()` is used for cross-platform compatibility
3. **Shell Scripts**: Use `.bat` files or PowerShell scripts instead of bash scripts
4. **Home Directory**: `~` expands to `%USERPROFILE%` (usually `C:\Users\YourUsername\`)

## Additional Resources

- [Original README](./README.md) - General project documentation
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Windows Guide](https://pytorch.org/get-started/locally/)

## Need Help?

If you encounter issues not covered here:
1. Check the error message carefully
2. Verify all environment variables are set correctly
3. Ensure the conda environment is activated
4. Check that all dataset files are in the correct locations
5. Review the original README.md for general usage instructions
