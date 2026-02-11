# 🧠 Semantic Entropy Probes — Web Application Guide

A real-time hallucination detection interface powered by Semantic Entropy Probes (SEPs). Ask questions to an LLM and see live uncertainty metrics that indicate whether the model is confident or hallucinating.

![Architecture Overview](#architecture-overview)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Interface](#using-the-interface)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This web application provides an interactive chat interface where you can:

- **Ask questions** to a Llama LLM (Llama 2 7B or Llama 3.1 8B)
- **See real-time uncertainty scores** — Semantic Entropy and Accuracy Probability — for every response
- **Visualize sentence-level hallucination highlighting** — individual sentences are color-coded by their uncertainty
- **Track metrics over time** with a live chart showing entropy and accuracy trends across your conversation
- **Switch between models** on the fly from the navbar

The system uses trained linear probes that read the model's internal hidden states to predict whether the model is hallucinating, without needing to generate multiple samples.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Browser                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            React Frontend (Vite, Port 5173)            │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │  │
│  │  │    Chat       │ │  Uncertainty │ │   Guidance     │  │  │
│  │  │  Interface    │ │    Chart     │ │    Modal       │  │  │
│  │  └──────┬───────┘ └──────┬───────┘ └───────────────┘  │  │
│  └─────────┼────────────────┼────────────────────────────┘  │
└────────────┼────────────────┼────────────────────────────────┘
             │  HTTP/JSON     │
             ▼                │
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Inference Engine                      │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │  LLM     │  │  Trained     │  │  Sentence     │  │   │
│  │  │ (Llama)  │──│  SEP Probes  │──│  Analysis     │  │   │
│  │  │ 4-bit Q  │  │  (.pkl)      │  │  (NLTK)       │  │   │
│  │  └──────────┘  └──────────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 8GB+ VRAM | NVIDIA RTX 3060 12GB or better |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB free (for model downloads) | 50 GB free |
| **CUDA** | CUDA 11.8+ | CUDA 12.1+ |

> [!IMPORTANT]
> An NVIDIA GPU with CUDA support is **required**. The model is loaded in 4-bit quantization, which needs a minimum of ~6GB VRAM for a 7B model. CPU-only inference is supported but will be extremely slow.

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10 or 3.11 | Backend runtime |
| **Node.js** | 18+ (LTS recommended) | Frontend build & dev server |
| **npm** | 9+ (comes with Node.js) | Frontend package management |
| **Conda** (Miniconda or Anaconda) | Latest | Python environment management |
| **Git** | Latest | Repository cloning |
| **NVIDIA Drivers** | 525+ | GPU support |
| **CUDA Toolkit** | 11.8+ | PyTorch CUDA backend |

### Accounts & Tokens

| Account | Required? | Purpose |
|---------|-----------|---------|
| [Hugging Face](https://huggingface.co) | ✅ **Required** | Download Llama models |
| [Meta Llama Access](https://huggingface.co/meta-llama) | ✅ **Required** | Gated model access approval |
| [Weights & Biases](https://wandb.ai) | ❌ Optional | Only for data generation/training (not needed for the web app) |

> [!NOTE]
> You must request and receive access to the Llama model on Hugging Face **before** running the app. Go to [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) and click "Request Access". Approval is usually instant.

### Trained Probe File

The application requires a pre-trained probe file (`.pkl`) to function. This file contains the trained logistic regression models that predict semantic entropy from the LLM's hidden states.

The probe file should be placed at:
```
semantic_entropy_probes/models/Llama2-7b_inference.pkl
```

> [!TIP]
> If you don't have a trained probe file, you can train one yourself using the `semantic_entropy_probes/train-latent-probe.ipynb` notebook. See the main [README.md](./README.md) for data generation instructions.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/DangerDudeSL/semantic-entropy-probes.git
cd semantic-entropy-probes
```

### Step 2: Set Up the Python Environment

**Option A: Using the provided Conda environment file (Recommended)**

```bash
# For Windows
conda env create -f sep_enviroment_windows.yaml
conda activate se_probes

# For Linux/macOS
conda env create -f sep_enviroment.yaml
conda activate se_probes
```

**Option B: Manual setup with pip**

```bash
# Create a new conda environment
conda create -n se_probes python=3.10 -y
conda activate se_probes

# Install PyTorch with CUDA (adjust cuda version as needed)
# Visit https://pytorch.org/get-started/locally/ for the correct command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install backend dependencies
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

Set your Hugging Face token so the app can download Llama models:

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_HUB_TOKEN = "hf_your_token_here"
```

**Windows (Permanent):**
1. Press `Win + X` → "System" → "Advanced system settings" → "Environment Variables"
2. Add `HUGGING_FACE_HUB_TOKEN` with your token value

**Linux/macOS:**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

Or log in via the CLI:
```bash
huggingface-cli login
```

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 5: Download NLTK Data

The backend uses NLTK for sentence tokenization. It will auto-download on first run, but you can pre-download:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 6: Verify Installation

```bash
# Check Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check Node.js
node --version
npm --version
```

Expected output:
```
PyTorch: 2.x.x, CUDA: True
v18.x.x (or higher)
9.x.x (or higher)
```

---

## Running the Application

### Option 1: Quick Start (Recommended)

Use the provided launcher script that starts both backend and frontend together:

```bash
python start_app.py
```

This will:
1. Start the **FastAPI backend** on `http://localhost:8000`
2. Start the **Vite frontend** on `http://localhost:5173`
3. Load the LLM model and probes on startup

> [!NOTE]
> The first run will take longer as it downloads the Llama model (~4GB for 4-bit). Subsequent runs use the cached model.

### Option 2: Windows Batch File

If you're on Windows and using Conda:

```
run_app.bat
```

> [!IMPORTANT]
> Edit `run_app.bat` first and update the conda environment name and path to match your setup.

### Option 3: Manual Start (Two Terminals)

**Terminal 1 — Backend:**
```bash
conda activate se_probes
cd backend
python main.py
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Accessing the Application

Once both servers are running, open your browser and navigate to:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | [http://localhost:5173](http://localhost:5173) | Main chat interface |
| **Backend API** | [http://localhost:8000](http://localhost:8000) | REST API |
| **API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger documentation |

### Stopping the Application

- If using `start_app.py`: Press `Ctrl+C` in the terminal
- If running manually: Press `Ctrl+C` in each terminal

---

## Using the Interface

### Chat Interface (Left Panel)

1. **Type a question** in the input box at the bottom and press Enter or click Send
2. The model will generate a response with uncertainty analysis
3. **Sentence highlighting** (toggle via the eye icon in the navbar):
   - 🟢 **Green** sentences — Low uncertainty, high confidence
   - 🟡 **Yellow** sentences — Moderate uncertainty
   - 🔴 **Red** sentences — High uncertainty, likely hallucination
4. **Hover over highlighted sentences** to see their individual entropy and accuracy scores

### Uncertainty Chart (Right Panel)

- Tracks **Semantic Entropy** (red line) and **Accuracy Probability** (blue line) across your conversation
- Each data point corresponds to one question-answer pair
- Hover over points to see the original question

### Metrics Card (Bottom Right)

Shows the scores for the **last response**:
- **Semantic Entropy**: How confused the model is (lower is better, > 0.5 suggests hallucination)
- **Accuracy Probability**: How likely the answer is correct (higher is better)

### Navbar Controls

| Control | Description |
|---------|-------------|
| **Model selector** | Switch between Llama 2 (7B) and Llama 3.1 (8B) |
| **Status indicator** | Green dot = model loaded, Red = not loaded |
| **Highlights toggle** | Enable/disable sentence-level uncertainty coloring |
| **Help button** (?) | Opens the Guidance modal explaining the metrics |
| **Clear History** | Resets chat and chart data |

### Understanding the Metrics

| Scenario | Entropy | Accuracy | Interpretation |
|----------|---------|----------|----------------|
| ✅ Reliable | Low (< 0.3) | High (> 0.7) | Model is confident and consistent |
| ⚠️ Uncertain | Medium (0.3–0.6) | Medium (0.4–0.7) | Treat with caution, verify independently |
| ❌ Likely Hallucination | High (> 0.5) | Low (< 0.5) | Model is guessing, do not trust |

---

## API Reference

The backend exposes a RESTful API. Full interactive docs are available at `http://localhost:8000/docs`.

### `GET /status`

Check the current status of the engine.

**Response:**
```json
{
  "model_loaded": true,
  "probes_loaded": true,
  "model_name": "meta-llama/Llama-2-7b-hf",
  "probe_name": "llama3-triviaqa"
}
```

### `POST /infer`

Send a question and get a response with uncertainty metrics.

**Request:**
```json
{
  "question": "What is the capital of France?"
}
```

**Response:**
```json
{
  "answer": "The capital of France is Paris.",
  "entropy": 0.1234,
  "accuracy_prob": 0.8765,
  "sentence_details": [
    {
      "text": "The capital of France is Paris.",
      "entropy": 0.1234,
      "accuracy_prob": 0.8765
    }
  ]
}
```

### `POST /set_model`

Switch the loaded LLM model.

**Request:**
```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
```

---

## Troubleshooting

### Backend won't start

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Ensure conda environment is activated: `conda activate se_probes` |
| `FileNotFoundError: Probe file not found` | Place your trained `.pkl` probe file in `semantic_entropy_probes/models/` |
| `CUDA out of memory` | Close other GPU applications, or ensure 4-bit quantization is enabled (default) |
| `torch not compiled with CUDA` | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

### Frontend won't start

| Problem | Solution |
|---------|----------|
| `npm: command not found` | Install Node.js from [nodejs.org](https://nodejs.org) |
| `node_modules missing` | Run `cd frontend && npm install` |
| Port 5173 already in use | Kill the existing process or change the port in `frontend/vite.config.js` |

### Model loading is slow

- **First run**: The model needs to be downloaded (~4GB). This is a one-time cost.
- **Subsequent runs**: Model loads from the Hugging Face cache (`~/.cache/huggingface/`). Should take 1–3 minutes depending on your hardware.
- Ensure you have sufficient RAM (16GB minimum) and VRAM (6GB minimum for 4-bit quantization).

### "Backend offline" in the frontend

- Ensure the backend is running on port 8000
- Check the backend terminal for errors
- The frontend polls the backend every 2 seconds — the status indicator (green/red dot) shows connectivity

### Sentence highlighting not working

- Click the **eye icon** in the navbar to toggle highlights on
- Very short answers (1 sentence) may not show sentence-level variation
- Check the backend logs for `[WARNING] Sentence-level analysis failed`

---

## Project Structure

```
semantic-entropy-probes/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx              # Main app component
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx    # Chat UI with message rendering
│   │   │   ├── Navbar.jsx           # Top navigation with model selector
│   │   │   ├── UncertaintyChart.jsx # Chart.js line chart for metrics
│   │   │   └── Guidance.jsx         # Help modal explaining metrics
│   │   └── index.css            # Global styles (Tailwind CSS)
│   └── package.json
├── backend/
│   ├── main.py                  # FastAPI server with endpoints
│   ├── engine.py                # Inference engine (model + probes)
│   └── requirements.txt         # Backend Python dependencies
├── semantic_entropy_probes/
│   └── models/
│       └── Llama2-7b_inference.pkl  # Trained probe models
├── start_app.py                 # Launcher (starts both servers)
├── run_app.bat                  # Windows batch launcher
├── requirements.txt             # Root-level Python dependencies
└── README.md                    # Main project README
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.
