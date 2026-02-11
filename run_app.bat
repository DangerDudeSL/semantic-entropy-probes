@echo off
echo Activating Conda Environment se_probes_fixed (Python 3.10 with CUDA) ...
call C:\ProgramData\miniconda3\Scripts\activate.bat se_probes_fixed

echo Checking Python version:
python --version

echo Checking Torch version (should have +cu121):
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, CUDNN: {torch.backends.cudnn.version()}')"

echo Starting Application...
cd /d "d:\Github Repositories\semantic-entropy-probes"
python start_app.py

echo Application stopped.
pause
