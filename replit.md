# BitNet LLM Demo Project

## Project Overview
This project was created to demonstrate Microsoft's BitNet b1.58 2B LLM, a 1-bit Large Language Model.

## Current Status & Known Issues

### BitNet Model Compatibility Issue
**Problem:** The Microsoft BitNet models require the `accelerate` Python package to load, but `accelerate` has dependency conflicts in the current Replit environment and cannot be installed.

**Attempted Solutions:**
1. ✗ Install accelerate directly - dependency resolution fails
2. ✗ Use BF16 variant of BitNet model - still requires accelerate
3. ✗ Use compatibility mode with `_fast_init=False` - accelerate still required at a deeper level

**Root Cause:** The `shumingma/transformers` fork (which adds BitNet support) enforces the accelerate dependency for BitNet quantized models, and the dependency cannot be satisfied in this environment.

### Recommended Solutions

**Option 1: Run BitNet Locally with bitnet.cpp**
For actual BitNet performance benefits, use the official C++ implementation:
```bash
git clone https://github.com/microsoft/BitNet.git
cd BitNet
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```

**Option 2: Use Alternative Small Model**
Replace BitNet with another small, efficient model that doesn't require special dependencies:
- `microsoft/phi-2` (2.7B parameters)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)  
- `google/gemma-2b` (2B parameters)

## Project Structure

- `app.py` - Main Streamlit application
- `pyproject.toml` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Features Implemented

### MVP Features (Attempted)
- [x] Download model from Hugging Face
- [x] Interactive chat interface
- [x] Adjustable generation parameters (temperature, max tokens)
- [x] Model loading progress indicators
- [x] UI showing model information
- [x] Example prompts
- [ ] **BLOCKED:** Actual text generation (requires accelerate package)

### Planned Next-Phase Features
- [ ] Conversation history persistence across sessions
- [ ] Model response streaming
- [ ] Advanced parameter controls (top-p, top-k)
- [ ] Multi-turn conversation memory
- [ ] Performance metrics display

## Technical Details

- **Framework:** Streamlit 1.51.0+
- **ML Library:** PyTorch (CPU-only)
- **Transformers:** Custom fork from shumingma/transformers (BitNet support)
- **Python:** 3.11
- **Deployment:** Replit workflow on port 5000

## Running the Project

The project automatically runs via the configured Streamlit workflow:
```bash
streamlit run app.py --server.port 5000
```

## Notes for Future Development

If you want to make BitNet work:
1. Install in an environment where `accelerate` can be installed (e.g., standard Python venv, not Replit)
2. Or use a different deployment platform that supports all dependencies
3. Or switch to using bitnet.cpp for optimal performance
