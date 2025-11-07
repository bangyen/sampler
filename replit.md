# Quantized LLM Comparison Demo

## Overview
This project is a FastAPI web application with a JavaScript frontend that demonstrates multiple quantized and efficient Large Language Models (LLMs), including Microsoft's BitNet b1.58 2B (1-bit LLM), SmolLM2, and Qwen2.5 models. It provides an interactive chat interface with streaming responses, advanced generation parameters, and performance metrics. The application also includes Named Entity Recognition (NER) and Optical Character Recognition (OCR) capabilities. The business vision is to showcase the potential of highly efficient LLMs for real-world applications, highlighting their low memory footprint and potential for deployment on resource-constrained devices.

## User Preferences
I want the agent to use a direct and concise communication style. When making changes, prioritize iterative development and ask for confirmation before implementing major architectural shifts. Ensure detailed explanations are provided for complex technical decisions or new feature implementations. Do not modify files in the `conversations/` directory, and avoid making changes to `replit.md` itself. Do not use emojis in communication.

## System Architecture

### UI/UX Decisions
The application features a modern JavaScript frontend (HTML/CSS/JS) with a four-tab interface: LLM Chat, NER, OCR, and Layout Analysis. It boasts a fully responsive mobile design with a hamburger menu and off-canvas sidebar for an optimal user experience across devices. The design prioritizes clean layouts and smooth animations (300ms CSS transitions). Each tab has its own sidebar for configuration and history management.

### Technical Implementations
The backend is built with FastAPI, providing REST APIs and Server-Sent Events (SSE) for real-time token streaming during LLM inference. Model loading is handled automatically from Hugging Face, with CPU-only inference for BitNet, SmolLM2, and Qwen2.5 models. The chat interface uses `TextIteratorStreamer` for token-by-token generation. Advanced generation parameters like temperature, top-p, top-k, and max tokens are configurable. Performance metrics (time, tokens, tokens/second) are tracked and displayed.

### Feature Specifications
- **LLM Chat:** Interactive chat with multiple model selection (BitNet b1.58 2B, SmolLM2 1.7B, Qwen2.5 1.5B/0.5B), streaming responses, and configurable generation parameters.
- **Named Entity Recognition (NER):** Extracts Person, Organization, Location, and Miscellaneous entities from text using multiple models (BERT Base NER, RoBERTa Large NER). Supports history tracking with MD5-hashed IDs.
- **Optical Character Recognition (OCR):** Extracts text from uploaded images using EasyOCR with multiple language configurations (English Only, English + Spanish, English + Chinese, Multi-Language). Provides bounding boxes and confidence scores. Supports history tracking with MD5-hashed IDs.
- **Layout Analysis:** Advanced document layout analysis using PaddleOCR for detecting text regions, tables, and document structure. Supports history tracking with MD5-hashed IDs.
- **History Management:** All analysis types (conversations, NER, OCR, Layout) have persistent history stored in JSON files with MD5-based content hashing. Limited to 50 most recent items per type.
- **Tabbed Interface:** Four tabs (LLM Chat, NER, OCR, Layout Analysis) with dynamic sidebars that adapt to the active tab.

### System Design Choices
The project uses a clear separation of concerns with `server.py` for FastAPI logic, `static/` for frontend assets, and dedicated modules for database (`database.py`) and JSON storage (`json_storage.py`, `ner_storage.py`, `ocr_storage.py`, `layout_storage.py`). The system is designed for robustness, including graceful error handling, automatic fallback mechanisms (e.g., database to JSON storage), lazy imports to avoid dependency conflicts (transformers pipeline only imported when NER is used), and optional dependency loading for features like NER/OCR/Layout Analysis.

## External Dependencies
- **FastAPI:** Web framework for the backend.
- **Uvicorn:** ASGI server for running FastAPI.
- **sse-starlette:** For Server-Sent Events implementation.
- **PyTorch:** Underlying deep learning framework (CPU-only build). Note: torchvision is installed but not directly imported to avoid compatibility issues.
- **Hugging Face Transformers:** For loading and interacting with LLM models. A custom fork (`git+https://github.com/shumingma/transformers.git`) is used for BitNet compatibility. The `pipeline` function is lazy-loaded only when NER is used.
- **accelerate:** Required by BitNet models for efficient computation.
- **EasyOCR:** Used for Optical Character Recognition with configurable language support.
- **PaddleOCR:** Used for advanced document layout analysis (CPU-optimized).
- **Pillow:** Image processing library, required for OCR and Layout Analysis.
- **SQLAlchemy:** ORM for database interactions (intended for PostgreSQL, currently uses JSON fallback).
- **psycopg2-binary:** PostgreSQL adapter (declared, but JSON fallback is active).
- **dslim/bert-base-NER and Jean-Baptiste/roberta-large-ner-english:** Hugging Face models used for Named Entity Recognition.