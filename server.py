from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.pipelines import pipeline
from threading import Thread
import json
import uuid
import time
from sse_starlette.sse import EventSourceResponse
import io
import base64

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    from database import (
        save_conversation,
        load_conversation,
        get_all_conversations,
        delete_conversation,
        get_engine,
    )
    from sqlalchemy import text

    def test_database_connection():
        try:
            engine = get_engine()
            if engine is None:
                return False
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    if test_database_connection():
        DATABASE_AVAILABLE = True
        PERSISTENCE_TYPE = "PostgreSQL"
    else:
        raise ConnectionError("Database connection test failed")
except (ImportError, ModuleNotFoundError, ConnectionError):
    try:
        from json_storage import (
            save_conversation,
            load_conversation,
            get_all_conversations,
            delete_conversation,
        )
        DATABASE_AVAILABLE = True
        PERSISTENCE_TYPE = "JSON"
    except Exception as e:
        DATABASE_AVAILABLE = False
        PERSISTENCE_TYPE = "None"

        def save_conversation(session_id, messages):
            return False

        def load_conversation(session_id):
            return []

        def get_all_conversations():
            return []

        def delete_conversation(session_id):
            return False

from ner_storage import (
    save_ner_analysis,
    load_ner_analysis,
    get_all_ner_analyses,
    delete_ner_analysis,
)

from ocr_storage import (
    save_ocr_analysis,
    load_ocr_analysis,
    get_all_ocr_analyses,
    delete_ocr_analysis,
)

from layout_storage import (
    save_layout_analysis,
    load_layout_analysis,
    get_all_layout_analyses,
    delete_layout_analysis,
)

# Import bitnet.cpp inference module (llama-cpp-python wrapper)
try:
    from bitnet_inference import BitNetInference, is_available as llama_cpp_available, download_gguf_model
    LLAMA_CPP_AVAILABLE = llama_cpp_available()
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    BitNetInference = None
    download_gguf_model = None

# Import BitNet compiled binary bridge
try:
    from bitnet_cpp_bridge import BitNetCppBridge, load_bitnet_model
    BITNET_CPP_AVAILABLE = True
except ImportError:
    BITNET_CPP_AVAILABLE = False
    BitNetCppBridge = None
    load_bitnet_model = None


app = FastAPI(title="Quantized LLM Comparison API")

AVAILABLE_MODELS = {
    "BitNet b1.58 2B": {
        "id": "microsoft/bitnet-b1.58-2B-4T-bf16",
        "params": "2B",
        "quantization": "1.58-bit",
        "memory": "~400MB",
        "description": "Microsoft's 1-bit LLM with ternary weights {-1, 0, +1}",
        "backend": "transformers"
    },
    "SmolLM2 1.7B": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "quantization": "FP16",
        "memory": "~3.4GB",
        "description": "HuggingFace's efficient model optimized for edge/mobile",
        "backend": "transformers"
    },
    "Qwen 2.5 1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "quantization": "FP16",
        "memory": "~3GB",
        "description": "Alibaba's multilingual instruct-tuned model (29+ languages)",
        "backend": "transformers"
    },
    "Qwen 2.5 0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "quantization": "FP16",
        "memory": "~1GB",
        "description": "Alibaba's smallest model, great for quick responses",
        "backend": "transformers"
    },
}

# Add GGUF models if backends are available

# SmolLM2 GGUF with llama-cpp-python
if LLAMA_CPP_AVAILABLE:
    AVAILABLE_MODELS["SmolLM2 1.7B (GGUF - Fast)"] = {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "params": "1.7B",
        "quantization": "Q4_K_M (GGUF)",
        "memory": "~1.1GB",
        "description": "SmolLM2 optimized with llama.cpp - 2-3x faster CPU inference!",
        "backend": "llamacpp",
        "gguf_repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "gguf_file": "smollm2-1.7b-instruct-q4_k_m.gguf"
    }

# BitNet GGUF with compiled bitnet.cpp binary
if BITNET_CPP_AVAILABLE:
    AVAILABLE_MODELS["BitNet b1.58 2B (GGUF - Fastest)"] = {
        "id": "microsoft/bitnet_b1_58-large",
        "params": "2B",
        "quantization": "i2_s (1.58-bit GGUF)",
        "memory": "~400MB",
        "description": "BitNet with custom bitnet.cpp - Ultimate 1.58-bit quantization!",
        "backend": "bitnet_cpp",
        "gguf_repo": "microsoft/bitnet_b1_58-large",
        "gguf_file": "ggml-model-i2_s.gguf"
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
    "English Only": {
        "engine": "easyocr",
        "languages": ["en"],
        "description": "Fastest - English text only",
    },
    "English + Spanish": {
        "engine": "easyocr",
        "languages": ["en", "es"],
        "description": "English and Spanish text recognition",
    },
    "English + Chinese": {
        "engine": "easyocr",
        "languages": ["en", "ch_sim"],
        "description": "English and Simplified Chinese text",
    },
    "Multi-Language": {
        "engine": "easyocr",
        "languages": ["en", "es", "fr", "de", "it", "pt"],
        "description": "Common European languages (slower)",
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
        return loaded_models[model_id]

    try:
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

        loaded_models[model_id] = {"model": model, "tokenizer": tokenizer}
        return loaded_models[model_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def load_llama_model(model_name: str):
    """Load GGUF model for llama.cpp backend"""
    if not LLAMA_CPP_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="llama.cpp backend not available. llama-cpp-python is not installed."
        )
    
    if model_name in loaded_llama_models:
        return loaded_llama_models[model_name]
    
    try:
        model_config = AVAILABLE_MODELS[model_name]
        gguf_repo = model_config.get("gguf_repo")
        gguf_file = model_config.get("gguf_file")
        
        if not gguf_repo:
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not have GGUF configuration")
        
        # Download GGUF model from Hugging Face
        model_path = download_gguf_model(gguf_repo, gguf_file)
        
        # Initialize BitNet inference engine
        inference = BitNetInference(model_path, n_ctx=2048)
        
        loaded_llama_models[model_name] = inference
        return inference
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading GGUF model: {str(e)}")


def load_bitnet_cpp_model(model_name: str):
    """Load GGUF model for bitnet.cpp compiled binary backend"""
    if not BITNET_CPP_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="BitNet.cpp backend not available. Binary not compiled or bitnet_cpp_bridge not found."
        )
    
    if model_name in loaded_bitnet_models:
        return loaded_bitnet_models[model_name]
    
    try:
        model_config = AVAILABLE_MODELS[model_name]
        gguf_repo = model_config.get("gguf_repo")
        gguf_file = model_config.get("gguf_file")
        
        if not gguf_repo:
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not have GGUF configuration")
        
        # Load BitNet model (downloads if needed)
        bridge = load_bitnet_model(gguf_repo, gguf_file)
        
        loaded_bitnet_models[model_name] = bridge
        return bridge
    except FileNotFoundError as e:
        # Binary not found - give clear guidance
        raise HTTPException(
            status_code=503, 
            detail=f"BitNet binary not compiled. Please compile it first: cd bin/BitNet/build && make. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading BitNet.cpp model: {str(e)}")


def load_ner_model(model_name="BERT Base NER"):
    """Load NER model using transformers pipeline"""
    global ner_pipelines
    
    if model_name not in NER_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid NER model: {model_name}")
    
    model_id = NER_MODELS[model_name]["id"]
    
    if model_id not in ner_pipelines:
        try:
            ner_pipelines[model_id] = pipeline(
                "ner",
                model=model_id,
                aggregation_strategy="simple",
                device=-1
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading NER model: {str(e)}")
    return ner_pipelines[model_id]


def load_ocr_model(config_name="English Only"):
    """Load OCR engine (EasyOCR or PaddleOCR) with specified configuration"""
    global ocr_readers
    
    if not PILLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PIL/Pillow not installed. Please install: pip install Pillow"
        )
    
    if config_name not in OCR_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid OCR configuration: {config_name}")
    
    config = OCR_CONFIGS[config_name]
    engine = config["engine"]
    
    if engine == "easyocr":
        if not EASYOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="EasyOCR not installed. Please install: pip install easyocr"
            )
        
        languages = tuple(config["languages"])
        
        if languages not in ocr_readers:
            try:
                ocr_readers[languages] = easyocr.Reader(list(languages), gpu=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading EasyOCR: {str(e)}")
        return ocr_readers[languages]
    
    elif engine == "paddleocr":
        if not PADDLEOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR not installed. Please install: pip install paddleocr"
            )
        
        if "paddleocr" not in ocr_readers:
            try:
                ocr_readers["paddleocr"] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading PaddleOCR: {str(e)}")
        return ocr_readers["paddleocr"]
    
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
    model, tokenizer, messages, temperature, max_tokens, top_p=0.9, top_k=50, request=None
):
    """Generate a streaming response from the model"""
    try:
        print(f"[DEBUG] Starting transformers streaming generation")
        prompt = format_prompt(messages, tokenizer)
        if prompt is None:
            print(f"[ERROR] Could not format prompt")
            yield json.dumps({'error': 'Could not format prompt'})
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
            yield json.dumps({'text': new_text})

        thread.join()
        print(f"[DEBUG] Transformers generation complete, {token_count} tokens")
        yield json.dumps({'done': True})
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in generate_response_streaming: {e}")
        print(traceback.format_exc())
        yield json.dumps({'error': str(e)})


async def generate_response_streaming_llama(
    inference, messages, temperature, max_tokens, top_p=0.9, top_k=50, request=None
):
    """Generate a streaming response using llama.cpp backend"""
    try:
        print(f"[DEBUG] Starting llama.cpp streaming generation")
        token_count = 0
        
        # Use the BitNetInference generate method
        for token in inference.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True
        ):
            # Check if client disconnected
            if request and await request.is_disconnected():
                print(f"[DEBUG] Client disconnected, stopping generation")
                return
            
            token_count += 1
            yield json.dumps({'text': token})
        
        print(f"[DEBUG] llama.cpp generation complete, {token_count} tokens")
        yield json.dumps({'done': True})
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in generate_response_streaming_llama: {e}")
        print(traceback.format_exc())
        yield json.dumps({'error': str(e)})


async def generate_response_streaming_bitnet_cpp(
    bridge, messages, temperature, max_tokens, top_p=0.9, top_k=50, request=None
):
    """Generate a streaming response using bitnet.cpp compiled binary backend"""
    try:
        print(f"[DEBUG] Starting bitnet.cpp streaming generation")
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
            system_message=system_msg
        ):
            # Check if client disconnected
            if request and await request.is_disconnected():
                print(f"[DEBUG] Client disconnected, stopping generation")
                return
            
            token_count += 1
            yield json.dumps({'text': token})
        
        print(f"[DEBUG] bitnet.cpp generation complete, {token_count} tokens")
        yield json.dumps({'done': True})
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in generate_response_streaming_bitnet_cpp: {e}")
        print(traceback.format_exc())
        yield json.dumps({'error': str(e)})


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
    return {
        "models": NER_MODELS
    }


@app.get("/api/ocr/configs")
async def get_ocr_configs():
    """Get list of available OCR configurations"""
    return {
        "configs": OCR_CONFIGS
    }


@app.get("/api/layout/config")
async def get_layout_config():
    """Get layout analysis configuration"""
    return {
        "config": LAYOUT_CONFIG
    }


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

        # Route to appropriate backend
        if backend == "llamacpp":
            # Use llama.cpp backend for GGUF models
            print(f"[DEBUG] Loading llama.cpp model: {request.model_name}")
            inference = load_llama_model(request.model_name)
            return EventSourceResponse(
                generate_response_streaming_llama(
                    inference,
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
            print(f"[DEBUG] Loading bitnet.cpp model: {request.model_name}")
            bridge = load_bitnet_cpp_model(request.model_name)
            return EventSourceResponse(
                generate_response_streaming_bitnet_cpp(
                    bridge,
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
            print(f"[DEBUG] Loading transformers model: {request.model_name}")
            model_id = model_config["id"]
            model_data = load_model(model_id)
            return EventSourceResponse(
                generate_response_streaming(
                    model_data["model"],
                    model_data["tokenizer"],
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


@app.post("/api/ner")
async def extract_entities(request: NERRequest):
    """Extract named entities from text"""
    try:
        ner_model = load_ner_model(request.model)
        start_time = time.time()
        
        entities = ner_model(request.text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "text": entity["word"],
                "label": entity["entity_group"],
                "score": float(entity["score"]),
                "start": entity["start"],
                "end": entity["end"]
            })
        
        # Save to history
        ner_id = save_ner_analysis(
            request.text,
            formatted_entities,
            request.model,
            processing_time
        )
        
        return {
            "entities": formatted_entities,
            "processing_time": processing_time,
            "text_length": len(request.text),
            "model": request.model,
            "id": ner_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during NER: {str(e)}")


@app.post("/api/ocr")
async def extract_text_from_image(file: UploadFile = File(...), config: str = "English Only"):
    """Extract text from uploaded image using OCR"""
    try:
        ocr_model = load_ocr_model(config)
        start_time = time.time()
        
        contents = await file.read()
        if not PILLOW_AVAILABLE:
            raise HTTPException(status_code=503, detail="PIL/Pillow not installed")
        
        engine = OCR_CONFIGS[config]["engine"]
        
        if engine == "easyocr":
            import numpy as np
            image = Image.open(io.BytesIO(contents))
            img_array = np.array(image)
            results = ocr_model.readtext(img_array)
            
            extracted_text = " ".join([text for (bbox, text, conf) in results])
            
            bounding_boxes = []
            for (bbox, text, confidence) in results:
                bounding_boxes.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(point[0]), int(point[1])] for point in bbox]
                })
        
        elif engine == "paddleocr":
            import numpy as np
            image = Image.open(io.BytesIO(contents))
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
                    bounding_boxes.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[int(point[0]), int(point[1])] for point in bbox_coords]
                    })
            
            extracted_text = " ".join(extracted_text_parts)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown OCR engine: {engine}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save to history
        ocr_id = save_ocr_analysis(
            contents,
            file.filename,
            extracted_text,
            bounding_boxes,
            config,
            processing_time,
            len(bounding_boxes)
        )
        
        return {
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "processing_time": processing_time,
            "num_detections": len(bounding_boxes),
            "config": config,
            "id": ocr_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during OCR: {str(e)}")


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


@app.post("/api/layout")
async def analyze_layout(file: UploadFile = File(...)):
    """Analyze document layout using PaddleOCR"""
    try:
        # Use PaddleOCR config
        if not PADDLEOCR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR not installed. Please install: pip install paddleocr"
            )
        
        if not PILLOW_AVAILABLE:
            raise HTTPException(status_code=503, detail="PIL/Pillow not installed")
        
        # Load PaddleOCR model
        if "paddleocr" not in ocr_readers:
            try:
                ocr_readers["paddleocr"] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading PaddleOCR: {str(e)}")
        
        ocr_model = ocr_readers["paddleocr"]
        start_time = time.time()
        
        contents = await file.read()
        
        import numpy as np
        image = Image.open(io.BytesIO(contents))
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
                bounding_boxes.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(point[0]), int(point[1])] for point in bbox_coords]
                })
        
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
            len(bounding_boxes)
        )
        
        return {
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "processing_time": processing_time,
            "num_detections": len(bounding_boxes),
            "id": layout_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during layout analysis: {str(e)}")


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


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
