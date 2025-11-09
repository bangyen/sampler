from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.pipelines import pipeline
from threading import Thread
import json
import time
from sse_starlette.sse import EventSourceResponse
import io

try:
    import easyocr  # noqa: F401

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR  # noqa: F401

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from PIL import Image  # noqa: F401

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import pytesseract  # noqa: F401

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Import database storage functions
from storage.database import (
    save_conversation,
    load_conversation,
    get_all_conversations,
    delete_conversation,
    clear_all_conversations,
    get_engine,
)
from sqlalchemy import text


# Test database connection on startup
def test_database_connection():
    try:
        engine = get_engine()
        if engine is None:
            return False
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False


if not test_database_connection():
    raise RuntimeError(
        "PostgreSQL database is not available. Please ensure DATABASE_URL is set correctly."
    )

DATABASE_AVAILABLE = True
PERSISTENCE_TYPE = "PostgreSQL"


from storage.ner_storage import (  # noqa: E402
    save_ner_analysis,
    load_ner_analysis,
    get_all_ner_analyses,
    delete_ner_analysis,
    clear_all_ner_analyses,
)

from storage.ocr_storage import (  # noqa: E402
    save_ocr_analysis,
    load_ocr_analysis,
    get_all_ocr_analyses,
    delete_ocr_analysis,
    clear_all_ocr_analyses,
)

from storage.layout_storage import (  # noqa: E402
    save_layout_analysis,
    load_layout_analysis,
    get_all_layout_analyses,
    delete_layout_analysis,
    clear_all_layout_analyses,
)

from storage.zero_shot_storage import (  # noqa: E402
    save_zero_shot_analysis,
    load_zero_shot_analysis,
    get_all_zero_shot_analyses,
    delete_zero_shot_analysis,
    clear_all_zero_shot_analyses,
)

# Import zero-shot classifier
try:
    from inference.zero_shot_classifier import (
        LLMZeroShotClassifier,
        ZeroShotResult,
        build_zero_shot_prompt,
    )

    ZERO_SHOT_AVAILABLE = True
except ImportError:
    ZERO_SHOT_AVAILABLE = False
    LLMZeroShotClassifier = None
    ZeroShotResult = None
    build_zero_shot_prompt = None

# Import bitnet.cpp inference module (llama-cpp-python wrapper)
try:
    from inference.bitnet_inference import (
        BitNetInference,
        is_available as llama_cpp_available,
        download_gguf_model,
    )

    LLAMA_CPP_AVAILABLE = llama_cpp_available()
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    BitNetInference = None
    download_gguf_model = None

# Import BitNet compiled binary bridge
try:
    from inference.bitnet_cpp_bridge import BitNetCppBridge, load_bitnet_model

    BITNET_CPP_AVAILABLE = True
except ImportError:
    BITNET_CPP_AVAILABLE = False
    BitNetCppBridge = None
    load_bitnet_model = None


app = FastAPI(title="Quantized LLM Comparison API")

AVAILABLE_MODELS = {
    # Qwen 0.5B always available as lightweight option
    "Qwen 2.5 0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "quantization": "FP16",
        "memory": "~1GB",
        "description": "Alibaba's smallest model, great for quick responses",
        "backend": "transformers",
    }
}

# Add GGUF models if backends are available

# SmolLM2 GGUF with llama-cpp-python
if LLAMA_CPP_AVAILABLE:
    AVAILABLE_MODELS["SmolLM2 1.7B"] = {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "params": "1.7B",
        "quantization": "Q4_K_M (GGUF)",
        "memory": "~1.1GB",
        "description": "Fast, efficient model optimized with llama.cpp - 2-3x faster inference",
        "backend": "llamacpp",
        "gguf_repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "gguf_file": "smollm2-1.7b-instruct-q4_k_m.gguf",
    }

    # Qwen 2.5 7B - Testing for improved accuracy
    AVAILABLE_MODELS["Qwen 2.5 7B"] = {
        "id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "params": "7B",
        "quantization": "Q4_K_M (GGUF)",
        "memory": "~4.7GB",
        "description": "Larger model with enhanced reasoning - testing for accuracy improvement",
        "backend": "llamacpp",
        "gguf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    }
else:
    # Fallback to transformers version if llama.cpp not available
    AVAILABLE_MODELS["SmolLM2 1.7B"] = {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "quantization": "FP16",
        "memory": "~3.4GB",
        "description": "HuggingFace's efficient model (transformers fallback)",
        "backend": "transformers",
    }

# BitNet GGUF with compiled bitnet.cpp binary
if BITNET_CPP_AVAILABLE:
    AVAILABLE_MODELS["BitNet b1.58 2B"] = {
        "id": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "params": "2B",
        "quantization": "i2_s (1.58-bit GGUF)",
        "memory": "~1.2GB",
        "description": "Fastest model with ultra-efficient 1.58-bit quantization",
        "backend": "bitnet_cpp",
        "gguf_repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "gguf_file": "ggml-model-i2_s.gguf",
    }

NER_MODELS = {
    "BERT Base NER": {
        "id": "dslim/bert-base-NER",
        "params": "110M",
        "memory": "~420MB",
        "description": "Fast and accurate BERT-based NER (Person, Organization, Location, Misc)",
    },
    "BERT Large NER": {
        "id": "dslim/bert-large-NER",
        "params": "340M",
        "memory": "~1.3GB",
        "description": "More accurate large BERT model for entity recognition",
    },
    "RoBERTa Large NER": {
        "id": "Jean-Baptiste/roberta-large-ner-english",
        "params": "355M",
        "memory": "~1.4GB",
        "description": "RoBERTa-based NER with high accuracy on English text",
    },
}

OCR_CONFIGS = {
    "EasyOCR English": {
        "engine": "easyocr",
        "languages": ["en"],
        "description": "Fast and accurate OCR with deep learning",
    },
    "Tesseract English": {
        "engine": "tesseract",
        "languages": ["eng"],
        "description": "Lightweight and fast traditional OCR engine",
    },
    "PaddleOCR English": {
        "engine": "paddleocr",
        "languages": ["en"],
        "description": "Advanced OCR with superior accuracy (CPU-optimized)",
    },
}

LAYOUT_CONFIG = {
    "PaddleOCR": {
        "engine": "paddleocr",
        "languages": ["en"],
        "description": "Advanced layout analysis with PaddleOCR (CPU-optimized)",
    },
}

loaded_models = {}
loaded_llama_models = {}
loaded_bitnet_models = {}
ner_pipelines = {}
ocr_readers = {}


class NERRequest(BaseModel):
    text: str
    model: str = "BERT Base NER"


class ZeroShotRequest(BaseModel):
    text: str
    candidate_labels: List[str]
    model: str = "Qwen 2.5 7B"
    hypothesis_template: str = "This example is {}."
    multi_label: bool = False
    use_logprobs: bool = False
    abstain_threshold: Optional[float] = None


class OCRResponse(BaseModel):
    text: str
    bounding_boxes: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: str
    metrics: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model_name: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9
    top_k: int = 50


class ConversationRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]


def load_model(model_id: str):
    """Load the selected model and tokenizer"""
    if model_id in loaded_models:
        return loaded_models[model_id], None

    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
        except (ImportError, OSError) as e:
            if "accelerate" in str(e).lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    _fast_init=False,
                )
            else:
                raise

        model.to("cpu")
        model.eval()

        load_time = time.time() - start_time

        loaded_models[model_id] = {"model": model, "tokenizer": tokenizer}
        return loaded_models[model_id], load_time
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def load_llama_model(model_name: str):
    """Load GGUF model for llama.cpp backend"""
    if not LLAMA_CPP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="llama.cpp backend not available. llama-cpp-python is not installed.",
        )

    if download_gguf_model is None or BitNetInference is None:
        raise HTTPException(
            status_code=503,
            detail="llama.cpp functions not available. bitnet_inference module not properly loaded.",
        )

    if model_name in loaded_llama_models:
        return loaded_llama_models[model_name], None

    try:
        start_time = time.time()
        model_config = AVAILABLE_MODELS[model_name]
        gguf_repo = model_config.get("gguf_repo")
        gguf_file = model_config.get("gguf_file")

        if not gguf_repo:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not have GGUF configuration",
            )

        if not gguf_file:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} missing GGUF file specification",
            )

        # Download GGUF model from Hugging Face
        model_path = download_gguf_model(gguf_repo, gguf_file)

        # Initialize BitNet inference engine
        inference = BitNetInference(model_path, n_ctx=2048)

        load_time = time.time() - start_time

        loaded_llama_models[model_name] = inference
        return inference, load_time
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading GGUF model: {str(e)}"
        )


def load_bitnet_cpp_model(model_name: str):
    """Load GGUF model for bitnet.cpp compiled binary backend"""
    if not BITNET_CPP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="BitNet.cpp backend not available. Binary not compiled or bitnet_cpp_bridge not found.",
        )

    if load_bitnet_model is None:
        raise HTTPException(
            status_code=503,
            detail="BitNet.cpp functions not available. bitnet_cpp_bridge module not properly loaded.",
        )

    if model_name in loaded_bitnet_models:
        return loaded_bitnet_models[model_name], None

    try:
        start_time = time.time()
        model_config = AVAILABLE_MODELS[model_name]
        gguf_repo = model_config.get("gguf_repo")
        gguf_file = model_config.get("gguf_file")

        if not gguf_repo:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not have GGUF configuration",
            )

        if not gguf_file:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} missing GGUF file specification",
            )

        # Load BitNet model (downloads if needed)
        bridge = load_bitnet_model(gguf_repo, gguf_file)

        load_time = time.time() - start_time

        loaded_bitnet_models[model_name] = bridge
        return bridge, load_time
    except FileNotFoundError as e:
        # Binary not found - give clear guidance
        raise HTTPException(
            status_code=503,
            detail=f"BitNet binary not compiled. Please compile it first: cd bin/BitNet/build && make. Error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading BitNet.cpp model: {str(e)}"
        )


def load_ner_model(model_name="BERT Base NER"):
    """Load NER model using transformers pipeline"""
    global ner_pipelines

    if model_name not in NER_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid NER model: {model_name}")

    model_id = NER_MODELS[model_name]["id"]

    if model_id in ner_pipelines:
        return ner_pipelines[model_id], None

    try:
        start_time = time.time()
        ner_pipelines[model_id] = pipeline(
            "ner", model=model_id, aggregation_strategy="simple", device=-1
        )
        load_time = time.time() - start_time
        return ner_pipelines[model_id], load_time
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading NER model: {str(e)}"
        )


def load_ocr_model(config_name="English Only"):
    """Load OCR engine (EasyOCR or PaddleOCR) with specified configuration"""
    global ocr_readers

    if not PILLOW_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PIL/Pillow not installed. Please install: pip install Pillow",
        )

    if config_name not in OCR_CONFIGS:
        raise HTTPException(
            status_code=400, detail=f"Invalid OCR configuration: {config_name}"
        )

    config = OCR_CONFIGS[config_name]
    engine = config["engine"]

    if engine == "easyocr":
        if not EASYOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="EasyOCR not installed. Please install: pip install easyocr",
            )

        languages = tuple(config["languages"])

        if languages in ocr_readers:
            return ocr_readers[languages], None

        try:
            import easyocr as easyocr_module

            start_time = time.time()
            ocr_readers[languages] = easyocr_module.Reader(list(languages), gpu=False)
            load_time = time.time() - start_time
            return ocr_readers[languages], load_time
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error loading EasyOCR: {str(e)}"
            )

    elif engine == "paddleocr":
        if not PADDLEOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR not installed. Please install: pip install paddleocr",
            )

        if "paddleocr" in ocr_readers:
            return ocr_readers["paddleocr"], None

        try:
            from paddleocr import PaddleOCR as PaddleOCREngine

            start_time = time.time()
            ocr_readers["paddleocr"] = PaddleOCREngine(lang="en")
            load_time = time.time() - start_time
            return ocr_readers["paddleocr"], load_time
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"PaddleOCR initialization failed in this environment. Please use EasyOCR instead. Error: {str(e)[:100]}",
            )

    elif engine == "tesseract":
        if not PYTESSERACT_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Tesseract not installed. Please install: pip install pytesseract",
            )

        # Tesseract doesn't need preloading, just return a placeholder
        # The actual OCR will be done per-request
        if "tesseract" not in ocr_readers:
            ocr_readers["tesseract"] = "initialized"
        return ocr_readers["tesseract"], None

    else:
        raise HTTPException(status_code=400, detail=f"Unknown OCR engine: {engine}")


def format_prompt(messages: List[Dict], tokenizer):
    """Format messages into a prompt string"""
    try:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError):
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)
        return prompt
    except Exception:
        return None


async def generate_response_streaming(
    model,
    tokenizer,
    messages,
    temperature,
    max_tokens,
    top_p=0.9,
    top_k=50,
    request=None,
    load_time=None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response from the model"""
    try:
        # Send model loading events if this was a fresh load
        if load_time is not None:
            print(f"[DEBUG] Model loaded in {load_time:.2f}s")
            yield json.dumps({"model_loading_end": True, "load_time": load_time})

        print("[DEBUG] Starting transformers streaming generation")
        prompt = format_prompt(messages, tokenizer)
        if prompt is None:
            print("[ERROR] Could not format prompt")
            yield json.dumps({"error": "Could not format prompt"})
            return

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": top_k,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        token_count = 0
        for new_text in streamer:
            # Check if client disconnected
            if request and await request.is_disconnected():
                thread.join(timeout=0.1)  # Try to cleanup thread
                return

            if new_text.startswith("Assistant:"):
                new_text = new_text[len("Assistant:") :].strip()
            token_count += 1
            yield json.dumps({"text": new_text})

        thread.join()
        print(f"[DEBUG] Transformers generation complete, {token_count} tokens")
        yield json.dumps({"done": True})
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception in generate_response_streaming: {e}")
        print(traceback.format_exc())
        yield json.dumps({"error": str(e)})


async def generate_response_streaming_llama(
    inference,
    messages,
    temperature,
    max_tokens,
    top_p=0.9,
    top_k=50,
    request=None,
    load_time=None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using llama.cpp backend"""
    try:
        # Send model loading events if this was a fresh load
        if load_time is not None:
            print(f"[DEBUG] Model loaded in {load_time:.2f}s")
            yield json.dumps({"model_loading_end": True, "load_time": load_time})

        print("[DEBUG] Starting llama.cpp streaming generation")
        token_count = 0

        # Use the BitNetInference generate method
        for token in inference.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
        ):
            # Check if client disconnected
            if request and await request.is_disconnected():
                print("[DEBUG] Client disconnected, stopping generation")
                return

            token_count += 1
            yield json.dumps({"text": token})

        print(f"[DEBUG] llama.cpp generation complete, {token_count} tokens")
        yield json.dumps({"done": True})
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception in generate_response_streaming_llama: {e}")
        print(traceback.format_exc())
        yield json.dumps({"error": str(e)})


async def generate_response_streaming_bitnet_cpp(
    bridge,
    messages,
    temperature,
    max_tokens,
    top_p=0.9,
    top_k=50,
    request=None,
    load_time=None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using bitnet.cpp compiled binary backend"""
    try:
        # Send model loading events if this was a fresh load
        if load_time is not None:
            print(f"[DEBUG] Model loaded in {load_time:.2f}s")
            yield json.dumps({"model_loading_end": True, "load_time": load_time})

        print("[DEBUG] Starting bitnet.cpp streaming generation")
        token_count = 0

        # Extract prompt from messages
        # Combine all messages into a single prompt for the binary
        prompt_parts = []
        system_msg = None
        last_role = None

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
                last_role = "user"
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
                last_role = "assistant"

        prompt = "\n".join(prompt_parts)

        # Always add "Assistant:" suffix when last message is from user
        # This signals the model to start generating the assistant's response
        if last_role == "user":
            prompt += "\nAssistant:"

        # Use the BitNetCppBridge generate method
        for token in bridge.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_message=system_msg,
        ):
            # Check if client disconnected
            if request and await request.is_disconnected():
                print("[DEBUG] Client disconnected, stopping generation")
                return

            token_count += 1
            yield json.dumps({"text": token})

        print(f"[DEBUG] bitnet.cpp generation complete, {token_count} tokens")
        yield json.dumps({"done": True})
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception in generate_response_streaming_bitnet_cpp: {e}")
        print(traceback.format_exc())
        yield json.dumps({"error": str(e)})


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
async def get_models():
    """Get list of available LLM models"""
    return {
        "models": AVAILABLE_MODELS,
        "persistence_type": PERSISTENCE_TYPE,
    }


@app.get("/api/ner/models")
async def get_ner_models():
    """Get list of available NER models"""
    return {"models": NER_MODELS}


@app.get("/api/ocr/configs")
async def get_ocr_configs():
    """Get list of available OCR configurations"""
    return {"configs": OCR_CONFIGS}


@app.get("/api/layout/config")
async def get_layout_config():
    """Get layout analysis configuration"""
    return {"config": LAYOUT_CONFIG}


@app.get("/api/zero-shot/available")
async def check_zero_shot_available():
    """Check if zero-shot classification is available"""
    return {
        "available": ZERO_SHOT_AVAILABLE,
        "models": list(AVAILABLE_MODELS.keys()) if ZERO_SHOT_AVAILABLE else [],
    }


@app.get("/api/models/status")
async def get_models_status():
    """Get loading status of all LLM models"""
    status = {}
    for model_name, model_config in AVAILABLE_MODELS.items():
        backend = model_config.get("backend", "transformers")

        if backend == "llamacpp":
            status[model_name] = {
                "loaded": model_name in loaded_llama_models,
                "backend": backend,
            }
        elif backend == "bitnet_cpp":
            status[model_name] = {
                "loaded": model_name in loaded_bitnet_models,
                "backend": backend,
            }
        else:
            model_id = model_config["id"]
            status[model_name] = {
                "loaded": model_id in loaded_models,
                "backend": backend,
            }

    return {"status": status}


class LoadModelRequest(BaseModel):
    model_name: str


@app.post("/api/models/load")
async def load_model_endpoint(request: LoadModelRequest):
    """Preemptively load a model"""
    try:
        if request.model_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail="Invalid model name")

        model_config = AVAILABLE_MODELS[request.model_name]
        backend = model_config.get("backend", "transformers")

        if backend == "llamacpp":
            if request.model_name in loaded_llama_models:
                return {
                    "success": True,
                    "already_loaded": True,
                    "model_name": request.model_name,
                    "backend": backend,
                }

            inference, load_time = load_llama_model(request.model_name)
            return {
                "success": True,
                "already_loaded": False,
                "model_name": request.model_name,
                "backend": backend,
                "load_time": load_time,
            }

        elif backend == "bitnet_cpp":
            if request.model_name in loaded_bitnet_models:
                return {
                    "success": True,
                    "already_loaded": True,
                    "model_name": request.model_name,
                    "backend": backend,
                }

            bridge, load_time = load_bitnet_cpp_model(request.model_name)
            return {
                "success": True,
                "already_loaded": False,
                "model_name": request.model_name,
                "backend": backend,
                "load_time": load_time,
            }

        else:
            model_id = model_config["id"]
            if model_id in loaded_models:
                return {
                    "success": True,
                    "already_loaded": True,
                    "model_name": request.model_name,
                    "backend": backend,
                }

            model_data, load_time = load_model(model_id)
            return {
                "success": True,
                "already_loaded": False,
                "model_name": request.model_name,
                "backend": backend,
                "load_time": load_time,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/api/ner/models/status")
async def get_ner_models_status():
    """Get loading status of all NER models"""
    status = {}
    for model_name, model_config in NER_MODELS.items():
        model_id = model_config["id"]
        status[model_name] = {"loaded": model_id in ner_pipelines}

    return {"status": status}


class LoadNERModelRequest(BaseModel):
    model_name: str


@app.post("/api/ner/models/load")
async def load_ner_model_endpoint(request: LoadNERModelRequest):
    """Preemptively load a NER model"""
    try:
        if request.model_name not in NER_MODELS:
            raise HTTPException(status_code=400, detail="Invalid NER model name")

        model_id = NER_MODELS[request.model_name]["id"]

        if model_id in ner_pipelines:
            return {
                "success": True,
                "already_loaded": True,
                "model_name": request.model_name,
            }

        ner_model, load_time = load_ner_model(request.model_name)

        return {
            "success": True,
            "already_loaded": False,
            "model_name": request.model_name,
            "load_time": load_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading NER model: {str(e)}"
        )


@app.get("/api/ocr/configs/status")
async def get_ocr_configs_status():
    """Get loading status of all OCR configurations"""
    status = {}
    for config_name, config in OCR_CONFIGS.items():
        engine = config["engine"]

        if engine == "easyocr":
            cache_key = tuple(config["languages"])
            status[config_name] = {"loaded": cache_key in ocr_readers, "engine": engine}
        elif engine == "paddleocr":
            status[config_name] = {
                "loaded": "paddleocr" in ocr_readers,
                "engine": engine,
            }
        elif engine == "tesseract":
            status[config_name] = {
                "loaded": "tesseract" in ocr_readers,
                "engine": engine,
            }
        else:
            status[config_name] = {"loaded": False, "engine": engine}

    return {"status": status}


class LoadOCRConfigRequest(BaseModel):
    config_name: str


@app.post("/api/ocr/configs/load")
async def load_ocr_config_endpoint(request: LoadOCRConfigRequest):
    """Preemptively load an OCR configuration"""
    try:
        if request.config_name not in OCR_CONFIGS:
            raise HTTPException(
                status_code=400, detail="Invalid OCR configuration name"
            )

        config = OCR_CONFIGS[request.config_name]
        engine = config["engine"]

        # Check if already loaded
        if engine == "easyocr":
            cache_key = tuple(config["languages"])
            already_loaded = cache_key in ocr_readers
        elif engine == "paddleocr":
            already_loaded = "paddleocr" in ocr_readers
        elif engine == "tesseract":
            already_loaded = "tesseract" in ocr_readers
        else:
            already_loaded = False

        if already_loaded:
            return {
                "success": True,
                "already_loaded": True,
                "config_name": request.config_name,
                "engine": engine,
            }

        ocr_model, load_time = load_ocr_model(request.config_name)

        return {
            "success": True,
            "already_loaded": False,
            "config_name": request.config_name,
            "engine": engine,
            "load_time": load_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading OCR config: {str(e)}"
        )


async def stream_with_loading_wrapper_transformers(
    model_id, messages, temperature, max_tokens, top_p, top_k, request
) -> AsyncGenerator[str, None]:
    """Wrapper generator that handles model loading and emits loading events"""
    # Check if model needs to be loaded
    is_cached = model_id in loaded_models

    if not is_cached:
        # Emit model loading start event
        yield json.dumps({"model_loading_start": True})

    # Load the model (returns cached model if already loaded)
    model_data, load_time = load_model(model_id)

    # Stream the actual generation
    async for chunk in generate_response_streaming(
        model_data["model"],
        model_data["tokenizer"],
        messages,
        temperature,
        max_tokens,
        top_p,
        top_k,
        request,
        load_time,
    ):
        yield chunk


async def stream_with_loading_wrapper_llama(
    model_name, messages, temperature, max_tokens, top_p, top_k, request
) -> AsyncGenerator[str, None]:
    """Wrapper generator that handles model loading and emits loading events"""
    # Check if model needs to be loaded
    is_cached = model_name in loaded_llama_models

    if not is_cached:
        # Emit model loading start event
        yield json.dumps({"model_loading_start": True})

    # Load the model (returns cached model if already loaded)
    inference, load_time = load_llama_model(model_name)

    # Stream the actual generation
    async for chunk in generate_response_streaming_llama(
        inference,
        messages,
        temperature,
        max_tokens,
        top_p,
        top_k,
        request,
        load_time,
    ):
        yield chunk


async def stream_with_loading_wrapper_bitnet(
    model_name, messages, temperature, max_tokens, top_p, top_k, request
) -> AsyncGenerator[str, None]:
    """Wrapper generator that handles model loading and emits loading events"""
    # Check if model needs to be loaded
    is_cached = model_name in loaded_bitnet_models

    if not is_cached:
        # Emit model loading start event
        yield json.dumps({"model_loading_start": True})

    # Load the model (returns cached model if already loaded)
    bridge, load_time = load_bitnet_cpp_model(model_name)

    # Stream the actual generation
    async for chunk in generate_response_streaming_bitnet_cpp(
        bridge,
        messages,
        temperature,
        max_tokens,
        top_p,
        top_k,
        request,
        load_time,
    ):
        yield chunk


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """Stream chat completions"""
    try:
        if request.model_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail="Invalid model name")

        model_config = AVAILABLE_MODELS[request.model_name]
        backend = model_config.get("backend", "transformers")

        print(f"[DEBUG] Using backend: {backend} for model: {request.model_name}")

        messages_dict = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ] + [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Route to appropriate backend with loading wrappers
        if backend == "llamacpp":
            # Use llama.cpp backend for GGUF models
            print(f"[DEBUG] Using llama.cpp backend for: {request.model_name}")
            return EventSourceResponse(
                stream_with_loading_wrapper_llama(
                    request.model_name,
                    messages_dict,
                    request.temperature,
                    request.max_tokens,
                    request.top_p,
                    request.top_k,
                    raw_request,
                )
            )
        elif backend == "bitnet_cpp":
            # Use bitnet.cpp compiled binary backend
            print(f"[DEBUG] Using bitnet.cpp backend for: {request.model_name}")
            return EventSourceResponse(
                stream_with_loading_wrapper_bitnet(
                    request.model_name,
                    messages_dict,
                    request.temperature,
                    request.max_tokens,
                    request.top_p,
                    request.top_k,
                    raw_request,
                )
            )
        else:
            # Use transformers backend for standard models
            print(f"[DEBUG] Using transformers backend for: {request.model_name}")
            model_id = model_config["id"]
            return EventSourceResponse(
                stream_with_loading_wrapper_transformers(
                    model_id,
                    messages_dict,
                    request.temperature,
                    request.max_tokens,
                    request.top_p,
                    request.top_k,
                    raw_request,
                )
            )
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception in chat_stream: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


async def stream_ner_extraction(request: NERRequest) -> AsyncGenerator[str, None]:
    """Stream NER extraction with model loading events"""
    try:
        # Check if model needs to be loaded
        model_id = NER_MODELS[request.model]["id"]
        is_cached = model_id in ner_pipelines

        if not is_cached:
            # Emit model loading start event
            yield json.dumps({"model_loading_start": True})

        # Load the model (returns cached model if already loaded)
        ner_model, load_time = load_ner_model(request.model)

        # Emit model loading end event if this was a fresh load
        if load_time is not None:
            yield json.dumps({"model_loading_end": True, "load_time": load_time})

        start_time = time.time()

        entities = ner_model(request.text)

        end_time = time.time()
        processing_time = end_time - start_time

        formatted_entities = []
        for entity in entities:
            formatted_entities.append(
                {
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "score": float(entity["score"]),
                    "start": entity["start"],
                    "end": entity["end"],
                }
            )

        # Save to history
        ner_id = save_ner_analysis(
            request.text, formatted_entities, request.model, processing_time
        )

        # Send the results
        yield json.dumps(
            {
                "done": True,
                "entities": formatted_entities,
                "processing_time": processing_time,
                "text_length": len(request.text),
                "model": request.model,
                "id": ner_id,
            }
        )
    except HTTPException as e:
        yield json.dumps({"error": str(e.detail)})
    except Exception as e:
        yield json.dumps({"error": f"Error during NER: {str(e)}"})


@app.post("/api/ner")
async def extract_entities(request: NERRequest):
    """Extract named entities from text (streaming)"""
    return EventSourceResponse(stream_ner_extraction(request))


async def stream_zero_shot_classification(
    request: ZeroShotRequest,
) -> AsyncGenerator[str, None]:
    """Stream zero-shot classification with schema-locked JSON outputs and logprob scoring"""
    try:
        if not ZERO_SHOT_AVAILABLE:
            yield json.dumps({"error": "Zero-shot classification not available"})
            return

        if request.model not in AVAILABLE_MODELS:
            yield json.dumps({"error": f"Invalid model: {request.model}"})
            return

        model_config = AVAILABLE_MODELS[request.model]
        backend = model_config.get("backend", "transformers")

        # Route to appropriate backend for zero-shot classification
        if backend == "llamacpp":
            # Use GGUF model via llama.cpp backend (2-3x faster)
            is_cached = request.model in loaded_llama_models

            if not is_cached:
                yield json.dumps({"model_loading_start": True})

            model_instance, load_time = load_llama_model(request.model)

            if load_time is not None:
                yield json.dumps({"model_loading_end": True, "load_time": load_time})

            start_time = time.time()

            if LLMZeroShotClassifier is None:
                yield json.dumps({"error": "Zero-shot classifier not properly loaded"})
                return

            # Load tokenizer from base model (not GGUF repo) for encoding labels
            # Map GGUF models to their base tokenizer repos
            if "SmolLM2" in request.model:
                tokenizer_repo = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            elif "Qwen 2.5 7B" in request.model:
                tokenizer_repo = "Qwen/Qwen2.5-7B-Instruct"
            else:
                # Fallback to base model ID if available
                tokenizer_repo = model_config.get("id", model_config.get("gguf_repo"))

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

            # Pass BitNetInference instance with tokenizer
            classifier = LLMZeroShotClassifier(
                model=model_instance, tokenizer=tokenizer, device="cpu"
            )

        elif backend == "bitnet_cpp":
            # BitNet.cpp doesn't support logits/logprobs extraction yet
            yield json.dumps(
                {
                    "error": "BitNet.cpp models are not supported for zero-shot classification yet. Please use SmolLM2 1.7B or Qwen 2.5 0.5B."
                }
            )
            return
        else:
            # Regular transformers backend
            model_id = model_config["id"]
            is_cached = model_id in loaded_models

            if not is_cached:
                yield json.dumps({"model_loading_start": True})

            model_data, load_time = load_model(model_id)

            if load_time is not None:
                yield json.dumps({"model_loading_end": True, "load_time": load_time})

            start_time = time.time()

            if LLMZeroShotClassifier is None:
                yield json.dumps({"error": "Zero-shot classifier not properly loaded"})
                return

            classifier = LLMZeroShotClassifier(
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                device="cpu",
            )

        result = classifier.classify(
            text=request.text,
            candidate_labels=request.candidate_labels,
            hypothesis_template=request.hypothesis_template,
            use_logprobs=request.use_logprobs,
            abstain_threshold=request.abstain_threshold,
            max_tokens=100,
            temperature=0.1,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        result_dict = result.model_dump()

        zs_id = save_zero_shot_analysis(
            text=request.text,
            labels=request.candidate_labels,
            results=result_dict,
            model=request.model,
            processing_time=processing_time,
            use_logprobs=request.use_logprobs,
            abstain_threshold=request.abstain_threshold,
        )

        yield json.dumps(
            {
                "done": True,
                "result": result_dict,
                "processing_time": processing_time,
                "model": request.model,
                "id": zs_id,
            }
        )

    except HTTPException as e:
        yield json.dumps({"error": str(e.detail)})
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception in zero-shot classification: {e}")
        print(traceback.format_exc())
        yield json.dumps({"error": f"Error during classification: {str(e)}"})


@app.post("/api/zero-shot/classify")
async def classify_zero_shot(request: ZeroShotRequest):
    """Perform zero-shot classification with schema-locked JSON outputs and logprob scoring (streaming)"""
    return EventSourceResponse(stream_zero_shot_classification(request))


async def stream_ocr_extraction(
    file_contents, filename, config
) -> AsyncGenerator[str, None]:
    """Stream OCR extraction with model loading events"""
    try:
        # Check if model needs to be loaded
        ocr_config = OCR_CONFIGS[config]
        engine = ocr_config["engine"]

        # Determine cache key
        if engine == "easyocr":
            cache_key = tuple(ocr_config["languages"])
            is_cached = cache_key in ocr_readers
        elif engine == "paddleocr":
            cache_key = "paddleocr"
            is_cached = cache_key in ocr_readers
        elif engine == "tesseract":
            cache_key = "tesseract"
            is_cached = cache_key in ocr_readers
        else:
            raise HTTPException(status_code=400, detail=f"Unknown OCR engine: {engine}")

        if not is_cached:
            # Emit model loading start event
            yield json.dumps({"model_loading_start": True})

        # Load the model (returns cached model if already loaded)
        ocr_model, load_time = load_ocr_model(config)

        # Emit model loading end event if this was a fresh load
        if load_time is not None:
            yield json.dumps({"model_loading_end": True, "load_time": load_time})

        start_time = time.time()

        if not PILLOW_AVAILABLE:
            raise HTTPException(status_code=503, detail="PIL/Pillow not installed")

        from PIL import Image as PILImage
        import numpy as np

        extracted_text = ""
        bounding_boxes = []

        if engine == "easyocr":
            image = PILImage.open(io.BytesIO(file_contents))
            img_array = np.array(image)
            results = ocr_model.readtext(img_array)  # type: ignore

            extracted_text = " ".join([text for (bbox, text, conf) in results])

            bounding_boxes = []
            for bbox, text, confidence in results:
                bounding_boxes.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[int(point[0]), int(point[1])] for point in bbox],
                    }
                )

        elif engine == "paddleocr":
            image = PILImage.open(io.BytesIO(file_contents))
            img_array = np.array(image)

            results = ocr_model.ocr(img_array, cls=True)  # type: ignore

            extracted_text_parts = []
            bounding_boxes = []

            if results and results[0]:
                for line in results[0]:
                    bbox_coords = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    extracted_text_parts.append(text)
                    bounding_boxes.append(
                        {
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": [
                                [int(point[0]), int(point[1])] for point in bbox_coords
                            ],
                        }
                    )

            extracted_text = " ".join(extracted_text_parts)

        elif engine == "tesseract":
            import pytesseract
            from PIL import Image as PILImage

            image = PILImage.open(io.BytesIO(file_contents))

            # Get language code (tesseract uses different codes than easyocr)
            lang = "+".join(ocr_config["languages"])

            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(image, lang=lang)

            # Get bounding box data
            data = pytesseract.image_to_data(
                image, lang=lang, output_type=pytesseract.Output.DICT
            )

            # Filter out empty detections and create bounding boxes
            bounding_boxes = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if text:  # Only include non-empty text
                    conf = float(data["conf"][i]) / 100.0  # Convert to 0-1 range
                    if conf > 0:  # Only include confident detections
                        x, y, w, h = (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        )
                        bounding_boxes.append(
                            {
                                "text": text,
                                "confidence": conf,
                                "bbox": [
                                    [x, y],
                                    [x + w, y],
                                    [x + w, y + h],
                                    [x, y + h],
                                ],
                            }
                        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Save to history
        ocr_id = save_ocr_analysis(
            file_contents,
            filename,
            extracted_text,
            bounding_boxes,
            config,
            processing_time,
            len(bounding_boxes),
        )

        # Send the results
        yield json.dumps(
            {
                "done": True,
                "text": extracted_text,
                "bounding_boxes": bounding_boxes,
                "processing_time": processing_time,
                "num_detections": len(bounding_boxes),
                "config": config,
                "id": ocr_id,
            }
        )
    except HTTPException as e:
        yield json.dumps({"error": str(e.detail)})
    except Exception as e:
        yield json.dumps({"error": f"Error during OCR: {str(e)}"})


@app.post("/api/ocr")
async def extract_text_from_image(
    file: UploadFile = File(...), config: str = "EasyOCR English"
):
    """Extract text from uploaded image using OCR (streaming)"""
    contents = await file.read()
    return EventSourceResponse(stream_ocr_extraction(contents, file.filename, config))


@app.post("/api/conversations/save")
async def save_conv(request: ConversationRequest):
    """Save a conversation"""
    messages_dict = [
        {
            "role": msg.role,
            "content": msg.content,
            "metrics": msg.metrics,
        }
        for msg in request.messages
    ]
    success = save_conversation(request.session_id, messages_dict)
    return {"success": success}


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Load a conversation"""
    messages = load_conversation(session_id)
    return {"messages": messages}


@app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    conversations = get_all_conversations()
    return {"conversations": conversations}


@app.delete("/api/conversations/{session_id}")
async def delete_conv(session_id: str):
    """Delete a conversation"""
    success = delete_conversation(session_id)
    return {"success": success}


@app.delete("/api/conversations")
async def clear_all_convs():
    """Delete all conversations"""
    success = clear_all_conversations()
    return {"success": success}


@app.get("/api/ner/history")
async def list_ner_analyses():
    """List all NER analyses"""
    analyses = get_all_ner_analyses()
    return {"analyses": analyses}


@app.get("/api/ner/history/{ner_id}")
async def get_ner_analysis(ner_id: str):
    """Load a specific NER analysis"""
    analysis = load_ner_analysis(ner_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="NER analysis not found")
    return analysis


@app.delete("/api/ner/history/{ner_id}")
async def delete_ner(ner_id: str):
    """Delete a NER analysis"""
    success = delete_ner_analysis(ner_id)
    return {"success": success}


@app.delete("/api/ner/history")
async def clear_all_ner():
    """Delete all NER analyses"""
    success = clear_all_ner_analyses()
    return {"success": success}


@app.get("/api/ocr/history")
async def list_ocr_analyses():
    """List all OCR analyses"""
    analyses = get_all_ocr_analyses()
    return {"analyses": analyses}


@app.get("/api/ocr/history/{ocr_id}")
async def get_ocr_analysis(ocr_id: str):
    """Load a specific OCR analysis"""
    analysis = load_ocr_analysis(ocr_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="OCR analysis not found")
    return analysis


@app.delete("/api/ocr/history/{ocr_id}")
async def delete_ocr(ocr_id: str):
    """Delete an OCR analysis"""
    success = delete_ocr_analysis(ocr_id)
    return {"success": success}


@app.delete("/api/ocr/history")
async def clear_all_ocr():
    """Delete all OCR analyses"""
    success = clear_all_ocr_analyses()
    return {"success": success}


@app.post("/api/zero-shot/history")
async def save_zero_shot_analysis_endpoint(analysis: Dict[str, Any]):
    """Save a zero-shot classification analysis"""
    analysis_id = save_zero_shot_analysis(
        text=analysis.get("text", ""),
        labels=analysis.get("labels", []),
        results=analysis.get("result", {}),
        model=analysis.get("model", ""),
        processing_time=float(analysis.get("duration", 0)),
        use_logprobs=True,
        abstain_threshold=analysis.get("result", {}).get("abstain_threshold"),
    )
    return {"success": analysis_id is not None, "id": analysis_id}


@app.get("/api/zero-shot/history")
async def list_zero_shot_analyses(limit: Optional[int] = None, offset: int = 0):
    """List zero-shot classification analyses with optional pagination"""
    result = get_all_zero_shot_analyses(limit=limit, offset=offset)

    if limit is None:
        return {"analyses": result["analyses"]}

    return result


@app.get("/api/zero-shot/history/{zs_id}")
async def get_zero_shot_analysis(zs_id: str):
    """Load a specific zero-shot classification analysis"""
    analysis = load_zero_shot_analysis(zs_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Zero-shot analysis not found")
    return analysis


@app.delete("/api/zero-shot/history/{zs_id}")
async def delete_zero_shot(zs_id: str):
    """Delete a zero-shot classification analysis"""
    success = delete_zero_shot_analysis(zs_id)
    return {"success": success}


@app.delete("/api/zero-shot/history")
async def clear_all_zero_shot():
    """Delete all zero-shot classification analyses"""
    success = clear_all_zero_shot_analyses()
    return {"success": success}


@app.post("/api/layout")
async def analyze_layout(file: UploadFile = File(...)):
    """Analyze document layout using PaddleOCR"""
    try:
        # Use PaddleOCR config
        if not PADDLEOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR not installed. Please install: pip install paddleocr",
            )

        if not PILLOW_AVAILABLE:
            raise HTTPException(status_code=503, detail="PIL/Pillow not installed")

        # Load PaddleOCR model
        from paddleocr import PaddleOCR as PaddleOCREngine

        if "paddleocr" not in ocr_readers:
            try:
                ocr_readers["paddleocr"] = PaddleOCREngine(lang="en")
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"PaddleOCR initialization failed in this environment. Please use EasyOCR instead. Error: {str(e)[:100]}",
                )

        ocr_model = ocr_readers["paddleocr"]
        start_time = time.time()

        contents = await file.read()

        from PIL import Image as PILImage
        import numpy as np

        image = PILImage.open(io.BytesIO(contents))
        img_array = np.array(image)

        results = ocr_model.ocr(img_array, cls=True)

        extracted_text_parts = []
        bounding_boxes = []

        if results and results[0]:
            for line in results[0]:
                bbox_coords = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]

                extracted_text_parts.append(text)
                bounding_boxes.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [
                            [int(point[0]), int(point[1])] for point in bbox_coords
                        ],
                    }
                )

        extracted_text = " ".join(extracted_text_parts)

        end_time = time.time()
        processing_time = end_time - start_time

        # Save to history
        layout_id = save_layout_analysis(
            contents,
            file.filename,
            extracted_text,
            bounding_boxes,
            processing_time,
            len(bounding_boxes),
        )

        return {
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "processing_time": processing_time,
            "num_detections": len(bounding_boxes),
            "id": layout_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during layout analysis: {str(e)}"
        )


@app.get("/api/layout/history")
async def list_layout_analyses():
    """List all layout analyses"""
    analyses = get_all_layout_analyses()
    return {"analyses": analyses}


@app.get("/api/layout/history/{layout_id}")
async def get_layout_analysis_by_id(layout_id: str):
    """Load a specific layout analysis"""
    analysis = load_layout_analysis(layout_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Layout analysis not found")
    return analysis


@app.delete("/api/layout/history/{layout_id}")
async def delete_layout_analysis_by_id(layout_id: str):
    """Delete a layout analysis"""
    success = delete_layout_analysis(layout_id)
    return {"success": success}


@app.delete("/api/layout/history")
async def clear_all_layout():
    """Delete all layout analyses"""
    success = clear_all_layout_analyses()
    return {"success": success}


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
