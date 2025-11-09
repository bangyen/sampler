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

**Inference Optimization**: The application supports three high-performance inference backends:
- **bitnet.cpp backend** (compiled binary): Microsoft's custom 1.58-bit kernels via subprocess. Delivers fastest inference with ~400MB memory. Includes prompt stripping and UTF-8 streaming. **Status: Active, binary compiled.**
- **llama.cpp backend** (via llama-cpp-python): Optimized GGUF models deliver 2-6x faster inference compared to standard PyTorch. Uses specialized CPU kernels with SIMD optimizations.
- **Transformers backend**: Standard Hugging Face transformers (hidden by default, available as fallback).

### Feature Specifications
- **LLM Chat:** Interactive chat with optimized GGUF models (BitNet b1.58 2B GGUF - Fastest, SmolLM2 1.7B GGUF - Fast). Slower transformers models hidden by default. Features streaming responses and configurable generation parameters.
- **Build Automation:** `build.sh` script automates BitNet.cpp binary compilation for easy deployment.
- **Named Entity Recognition (NER):** Extracts Person, Organization, Location, and Miscellaneous entities from text using `dslim/bert-base-NER`.
- **Optical Character Recognition (OCR):** Extracts text from uploaded images using EasyOCR, providing bounding boxes and confidence scores.
- **Database Persistence:** All application data (conversations, NER analyses, OCR results, zero-shot classifications, layout analyses) is persisted in PostgreSQL database with proper schema design and transactional safety.
- **Tabbed Interface:** Separates LLM Chat, NER, and OCR functionalities into distinct tabs.
- **Performance Optimization:** Automatically downloads and caches GGUF models for accelerated inference when using llama.cpp backend.

### System Design Choices
The project uses a clear separation of concerns with modular organization:
- **`server.py`**: Main FastAPI application entry point
- **`storage/`**: Persistence layer modules using PostgreSQL database exclusively
  - `database.py`: SQLAlchemy models and core database functions
  - `ner_storage.py`, `ocr_storage.py`, `zero_shot_storage.py`, `layout_storage.py`: Domain-specific storage with consistent API
- **`inference/`**: LLM inference backends (bitnet_inference, bitnet_cpp_bridge)
- **`static/`**: Frontend assets (HTML, CSS, JavaScript)
- **`tests/`**: Pytest test suite with comprehensive coverage

**Database Architecture:**
- PostgreSQL database required (no fallback to JSON)
- Five tables: conversations, messages, ner_analyses, ocr_analyses, zero_shot_analyses, layout_analyses
- All tables use auto-increment primary keys and unique indexed analysis_id columns
- Hash-based IDs (MD5) for NER/OCR/Layout ensure idempotent saves
- Timestamp-based IDs for zero-shot classifications prevent collisions
- Transactional safety with rollback on errors
- JSON columns for complex data (entities, results, candidate_labels)

The system is designed for robustness, including graceful error handling, proper transaction management with rollback support, and optional dependency loading for features like NER/OCR.

### Testing Infrastructure
The project includes a comprehensive pytest testing suite that validates backend modules and API endpoints. Tests are organized in the `tests/` directory with the following coverage:
- **JSON Storage Tests** (`test_json_storage.py`): Validates conversation persistence via JSON files, including save/load operations, error handling, and edge cases.
- **Database Tests** (`test_database.py`): Verifies PostgreSQL database operations with mocked connections, testing conversation CRUD operations.
- **Inference Tests** (`test_bitnet_inference.py`): Tests llama.cpp inference wrapper functionality, including model initialization, text generation, and streaming.
- **API Endpoint Tests** (`test_server.py`): Validates FastAPI endpoints using test client, covering conversation management, model listing, and NER/OCR configurations.

**Running Tests:**
- `just test` - Run all tests with verbose output
- `just all` - Run linting and tests together
- `python -m pytest tests/` - Direct pytest execution

The test suite uses fixtures for consistent test data and mocking for external dependencies, ensuring fast, isolated tests that don't require actual models or database connections.

## External Dependencies
- **FastAPI:** Web framework for the backend.
- **Uvicorn:** ASGI server for running FastAPI.
- **sse-starlette:** For Server-Sent Events implementation.
- **PyTorch:** Underlying deep learning framework (CPU-only build).
- **Hugging Face Transformers:** For loading and interacting with LLM models. A custom fork (`git+https://github.com/shumingma/transformers.git`) is used for BitNet compatibility.
- **llama-cpp-python:** Python bindings for llama.cpp, enabling optimized GGUF model inference with 2-6x performance improvements.
- **huggingface-hub:** For downloading GGUF models from Hugging Face repositories.
- **bitnet.cpp:** Compiled C++ binary (bin/BitNet/build/bin/llama-cli) for 1.58-bit quantization inference. Build with `./build.sh`.
- **accelerate:** Required by BitNet models for efficient computation.
- **EasyOCR:** Used for Optical Character Recognition.
- **Pillow:** Image processing library, a dependency for EasyOCR.
- **SQLAlchemy:** ORM for database interactions with PostgreSQL.
- **psycopg2-binary:** PostgreSQL adapter for database connectivity.
- **dslim/bert-base-NER:** Hugging Face model used for Named Entity Recognition.