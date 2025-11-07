# Quantized LLM Comparison Demo

## Overview
This project is a FastAPI web application with a JavaScript frontend that demonstrates multiple quantized and efficient Large Language Models (LLMs), including Microsoft's BitNet b1.58 2B (1-bit LLM), SmolLM2, and Qwen2.5 models. It provides an interactive chat interface with streaming responses, advanced generation parameters, and performance metrics. The application also includes Named Entity Recognition (NER) and Optical Character Recognition (OCR) capabilities. The business vision is to showcase the potential of highly efficient LLMs for real-world applications, highlighting their low memory footprint and potential for deployment on resource-constrained devices.

## User Preferences
I want the agent to use a direct and concise communication style. When making changes, prioritize iterative development and ask for confirmation before implementing major architectural shifts. Ensure detailed explanations are provided for complex technical decisions or new feature implementations. Do not modify files in the `conversations/` directory, and avoid making changes to `replit.md` itself. Do not use emojis in communication.

## System Architecture

### UI/UX Decisions
The application features a modern JavaScript frontend (HTML/CSS/JS) with a tabbed interface for LLM Chat, NER, and OCR. It boasts a fully responsive mobile design with a hamburger menu and off-canvas sidebar for an optimal user experience across devices. The design prioritizes clean layouts and smooth animations (300ms CSS transitions).

### Technical Implementations
The backend is built with FastAPI, providing REST APIs and Server-Sent Events (SSE) for real-time token streaming during LLM inference. Model loading is handled automatically from Hugging Face, with CPU-only inference for BitNet, SmolLM2, and Qwen2.5 models. The chat interface uses `TextIteratorStreamer` for token-by-token generation. Advanced generation parameters like temperature, top-p, top-k, and max tokens are configurable. Performance metrics (time, tokens, tokens/second) are tracked and displayed.

**Inference Optimization**: The application now supports dual inference backends for maximum performance:
- **llama.cpp backend** (via llama-cpp-python): Optimized GGUF models deliver 2-6x faster inference compared to standard PyTorch. Uses specialized CPU kernels with SIMD optimizations for extremely efficient computation. **Status: Integration complete, awaiting llama-cpp-python installation.**
- **Transformers backend**: Standard Hugging Face transformers for maximum compatibility with all models. Currently active for all models.

### Feature Specifications
- **LLM Chat:** Interactive chat with multiple model selection including optimized GGUF models (BitNet b1.58 2B GGUF - Fast) and standard transformers models (SmolLM2 1.7B, Qwen2.5 1.5B/0.5B). Features streaming responses and configurable generation parameters.
- **Named Entity Recognition (NER):** Extracts Person, Organization, Location, and Miscellaneous entities from text using `dslim/bert-base-NER`.
- **Optical Character Recognition (OCR):** Extracts text from uploaded images using EasyOCR, providing bounding boxes and confidence scores.
- **Conversation Persistence:** Conversations are saved and loaded, primarily using JSON file storage as a robust fallback to a PostgreSQL database.
- **Tabbed Interface:** Separates LLM Chat, NER, and OCR functionalities into distinct tabs.
- **Performance Optimization:** Automatically downloads and caches GGUF models for accelerated inference when using llama.cpp backend.

### System Design Choices
The project uses a clear separation of concerns with `server.py` for FastAPI logic, `static/` for frontend assets, and dedicated modules for database (`database.py`) and JSON storage (`json_storage.py`). The system is designed for robustness, including graceful error handling, automatic fallback mechanisms (e.g., database to JSON storage), and optional dependency loading for features like NER/OCR.

## External Dependencies
- **FastAPI:** Web framework for the backend.
- **Uvicorn:** ASGI server for running FastAPI.
- **sse-starlette:** For Server-Sent Events implementation.
- **PyTorch:** Underlying deep learning framework (CPU-only build).
- **Hugging Face Transformers:** For loading and interacting with LLM models. A custom fork (`git+https://github.com/shumingma/transformers.git`) is used for BitNet compatibility.
- **llama-cpp-python:** Python bindings for llama.cpp, enabling optimized GGUF model inference with 2-6x performance improvements.
- **huggingface-hub:** For downloading GGUF models from Hugging Face repositories.
- **accelerate:** Required by BitNet models for efficient computation.
- **EasyOCR:** Used for Optical Character Recognition.
- **Pillow:** Image processing library, a dependency for EasyOCR.
- **SQLAlchemy:** ORM for database interactions (intended for PostgreSQL, currently uses JSON fallback).
- **psycopg2-binary:** PostgreSQL adapter (declared, but JSON fallback is active).
- **dslim/bert-base-NER:** Hugging Face model used for Named Entity Recognition.