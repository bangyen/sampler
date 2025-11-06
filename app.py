import streamlit as st
import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import os
import uuid

def test_database_connection():
    """Test if PostgreSQL database is accessible"""
    try:
        from database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        if engine is None:
            return False
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

try:
    from database import save_conversation, load_conversation, get_all_conversations, delete_conversation
    if test_database_connection():
        DATABASE_AVAILABLE = True
        PERSISTENCE_TYPE = "PostgreSQL"
    else:
        raise ConnectionError("Database connection test failed")
except (ImportError, ModuleNotFoundError, ConnectionError):
    try:
        from json_storage import save_conversation, load_conversation, get_all_conversations, delete_conversation
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

st.set_page_config(
    page_title="Quantized LLM Comparison Demo",
    layout="wide"
)

AVAILABLE_MODELS = {
    "BitNet b1.58 2B": {
        "id": "microsoft/bitnet-b1.58-2B-4T-bf16",
        "params": "2B",
        "quantization": "1.58-bit",
        "memory": "~400MB",
        "description": "Microsoft's 1-bit LLM with ternary weights {-1, 0, +1}"
    },
    "SmolLM2 1.7B": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "quantization": "FP16",
        "memory": "~3.4GB",
        "description": "HuggingFace's efficient model optimized for edge/mobile"
    },
    "Qwen 2.5 1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "quantization": "FP16",
        "memory": "~3GB",
        "description": "Alibaba's multilingual instruct-tuned model (29+ languages)"
    },
    "Qwen 2.5 0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "quantization": "FP16",
        "memory": "~1GB",
        "description": "Alibaba's smallest model, great for quick responses"
    }
}

st.title("Quantized LLM Comparison Demo")

st.markdown("""
Compare different quantized and efficient LLM models to see how they perform. 
Select a model from the sidebar to switch between them.
""")

if PERSISTENCE_TYPE == "JSON":
    st.info("Using JSON file-based persistence (PostgreSQL database unavailable)")
elif PERSISTENCE_TYPE == "PostgreSQL":
    st.success("Using PostgreSQL database for persistence")
elif PERSISTENCE_TYPE == "None":
    st.warning("Conversation persistence is disabled")

@st.cache_resource
def load_model(model_id):
    """Load the selected model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        except (ImportError, OSError) as e:
            if "accelerate" in str(e).lower():
                st.warning("Model requires 'accelerate' package which is not available. Loading in compatibility mode...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    _fast_init=False
                )
            else:
                raise
        
        model.to("cpu")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def format_prompt(messages):
    """Format messages into a prompt string"""
    try:
        try:
            prompt = st.session_state.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
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
    except Exception as e:
        return None

def generate_response_streaming(model, tokenizer, messages, temperature, max_tokens, top_p=0.9, top_k=50):
    """Generate a streaming response from the model"""
    try:
        prompt = format_prompt(messages)
        if prompt is None:
            yield "Error: Could not format prompt"
            return
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": top_k,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            if new_text.startswith("Assistant:"):
                new_text = new_text[len("Assistant:"):].strip()
            yield new_text
        
        thread.join()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Generation error details:\n{error_details}")
        yield f"Error generating response: {str(e)}"

with st.sidebar:
    st.subheader("Model Selection")
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "BitNet b1.58 2B"
    
    selected_model_name = st.radio(
        "Choose a model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
        help="Select a model to use for generation"
    )
    
    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
        st.rerun()
    
    selected_model_info = AVAILABLE_MODELS[selected_model_name]
    st.caption(f"**{selected_model_info['params']} params | {selected_model_info['quantization']} | {selected_model_info['memory']}**")
    st.caption(selected_model_info['description'])
    
    st.markdown("---")
    st.subheader("Conversation History")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    loaded_messages = load_conversation(st.session_state.session_id)
    st.session_state.messages = loaded_messages if loaded_messages else []

model_id = AVAILABLE_MODELS[st.session_state.selected_model]["id"]

with st.spinner(f"Loading {st.session_state.selected_model}..."):
    model, tokenizer = load_model(model_id)

if model is None or tokenizer is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

st.session_state.tokenizer = tokenizer

st.success(f"Model loaded successfully: {st.session_state.selected_model}")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with st.sidebar:
    
    if st.button("+ New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    all_conversations = get_all_conversations()
    
    if all_conversations:
        st.caption(f"Found {len(all_conversations)} saved conversation(s)")
        
        for i, conv in enumerate(all_conversations[:10]):
            is_current = conv["session_id"] == st.session_state.session_id
            
            col_a, col_b = st.columns([2.5, 1.5])
            
            with col_a:
                button_label = f"{'>' if is_current else ''} {conv['message_count']} messages"
                if st.button(button_label, key=f"conv_{i}", use_container_width=True):
                    st.session_state.session_id = conv["session_id"]
                    st.session_state.messages = load_conversation(conv["session_id"])
                    st.rerun()
            
            with col_b:
                if st.button("Delete", key=f"del_{i}", use_container_width=True):
                    delete_conversation(conv["session_id"])
                    if conv["session_id"] == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    st.rerun()
        
        if len(all_conversations) > 10:
            st.caption(f"Showing 10 of {len(all_conversations)} conversations")
    else:
        st.info("No saved conversations yet")

with col2:
    st.subheader("Generation Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=50,
        max_value=500,
        value=150,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    with st.expander("Advanced Settings"):
        top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Sample from smallest set of tokens whose cumulative probability exceeds p"
        )
        
        top_k = st.slider(
            "Top-k sampling",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Sample from top k most likely tokens"
        )
    
    st.markdown("---")
    
    st.subheader("Model Info")
    current_model = AVAILABLE_MODELS[st.session_state.selected_model]
    st.markdown(f"""
    - **Model:** {st.session_state.selected_model}
    - **Parameters:** {current_model['params']}
    - **Quantization:** {current_model['quantization']}
    - **Memory:** {current_model['memory']}
    - **Model ID:** {current_model['id']}
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_conversation(st.session_state.session_id, st.session_state.messages)
        st.rerun()

with col1:
    st.subheader("Chat Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if len(st.session_state.messages) == 0:
        st.info("Welcome! Try one of these example prompts:")
        example_prompts = [
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence",
            "What are the benefits of 1-bit LLMs?",
            "Tell me an interesting fact about space"
        ]
        
        cols = st.columns(2)
        for idx, prompt in enumerate(example_prompts):
            col_idx = idx % 2
            if cols[col_idx].button(prompt, key=f"example_{idx}"):
                st.session_state.pending_prompt = prompt
                st.rerun()
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                metrics = message.get("metrics")
                if metrics:
                    tokens_per_sec = metrics.get('tokens_per_sec', 0)
                    if tokens_per_sec > 0 and tokens_per_sec < 1:
                        speed_display = f"{1/tokens_per_sec:.1f} sec/token"
                    elif tokens_per_sec >= 1:
                        speed_display = f"{tokens_per_sec:.1f} tokens/sec"
                    else:
                        speed_display = "N/A"
                    st.caption(f"Time: {metrics['time']:.1f}s | Tokens: {metrics['tokens']} | Speed: {speed_display}")
                else:
                    st.caption("_No metrics available for this response_")
    
    user_input = st.chat_input("Type your message here...")
    
    if "pending_prompt" in st.session_state:
        user_input = st.session_state.pending_prompt
        del st.session_state.pending_prompt
    
    if user_input:
        import time
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        messages_for_model = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ] + st.session_state.messages
        
        start_time = time.time()
        
        with st.chat_message("assistant"):
            response_text = st.write_stream(
                generate_response_streaming(
                    model, 
                    tokenizer, 
                    messages_for_model, 
                    temperature, 
                    max_tokens,
                    top_p,
                    top_k
                )
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if not isinstance(response_text, str):
                response_text = ""
            num_tokens = len(tokenizer.encode(response_text)) if response_text else 0
            tokens_per_second = num_tokens / generation_time if generation_time > 0 and num_tokens > 0 else 0
            
            metrics = {
                "time": generation_time,
                "tokens": num_tokens,
                "tokens_per_sec": tokens_per_second
            }
            
            if tokens_per_second > 0 and tokens_per_second < 1:
                speed_display = f"{1/tokens_per_second:.1f} sec/token"
            elif tokens_per_second >= 1:
                speed_display = f"{tokens_per_second:.1f} tokens/sec"
            else:
                speed_display = "N/A"
            st.caption(f"Time: {generation_time:.1f}s | Tokens: {num_tokens} | Speed: {speed_display}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "metrics": metrics
        })
        
        save_conversation(st.session_state.session_id, st.session_state.messages)
        st.rerun()

st.markdown("---")
st.caption("Built with Streamlit | Model: microsoft/bitnet-b1.58-2B-4T")
