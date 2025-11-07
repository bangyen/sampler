from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import json
import uuid
import time
from sse_starlette.sse import EventSourceResponse

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


app = FastAPI(title="Quantized LLM Comparison API")

AVAILABLE_MODELS = {
    "BitNet b1.58 2B": {
        "id": "microsoft/bitnet-b1.58-2B-4T-bf16",
        "params": "2B",
        "quantization": "1.58-bit",
        "memory": "~400MB",
        "description": "Microsoft's 1-bit LLM with ternary weights {-1, 0, +1}",
    },
    "SmolLM2 1.7B": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "quantization": "FP16",
        "memory": "~3.4GB",
        "description": "HuggingFace's efficient model optimized for edge/mobile",
    },
    "Qwen 2.5 1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "quantization": "FP16",
        "memory": "~3GB",
        "description": "Alibaba's multilingual instruct-tuned model (29+ languages)",
    },
    "Qwen 2.5 0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "quantization": "FP16",
        "memory": "~1GB",
        "description": "Alibaba's smallest model, great for quick responses",
    },
}

loaded_models = {}


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
    model, tokenizer, messages, temperature, max_tokens, top_p=0.9, top_k=50
):
    """Generate a streaming response from the model"""
    try:
        prompt = format_prompt(messages, tokenizer)
        if prompt is None:
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

        for new_text in streamer:
            if new_text.startswith("Assistant:"):
                new_text = new_text[len("Assistant:") :].strip()
            yield json.dumps({'text': new_text})

        thread.join()
        yield json.dumps({'done': True})
    except Exception as e:
        yield json.dumps({'error': str(e)})


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": AVAILABLE_MODELS,
        "persistence_type": PERSISTENCE_TYPE,
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat completions"""
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    model_id = AVAILABLE_MODELS[request.model_name]["id"]
    model_data = load_model(model_id)

    messages_dict = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ] + [{"role": msg.role, "content": msg.content} for msg in request.messages]

    return EventSourceResponse(
        generate_response_streaming(
            model_data["model"],
            model_data["tokenizer"],
            messages_dict,
            request.temperature,
            request.max_tokens,
            request.top_p,
            request.top_k,
        )
    )


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


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
