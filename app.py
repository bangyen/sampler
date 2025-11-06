import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

st.set_page_config(
    page_title="BitNet LLM Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Microsoft BitNet b1.58 2B LLM Demo")

st.markdown("""
This demo uses Microsoft's BitNet b1.58 2B model, a 1-bit Large Language Model with 2 billion parameters.
BitNet uses ternary weights {-1, 0, +1} requiring only ~400MB memory compared to 1.4-4.8GB for similar models.

**Note:** This implementation uses the Hugging Face Transformers library for compatibility. 
For production deployments with optimized performance, consider using the native bitnet.cpp framework.
""")

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

@st.cache_resource
def load_model():
    """Load the BitNet model and tokenizer"""
    with st.spinner("Downloading and loading BitNet model... This may take a few minutes on first run."):
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.to("cpu")
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None

def generate_response(model, tokenizer, messages, temperature, max_tokens):
    """Generate a response from the model"""
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

st.success("✅ Model loaded successfully!")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Generation Settings")
    
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
    
    st.markdown("---")
    
    st.subheader("📊 Model Info")
    st.markdown(f"""
    - **Model:** BitNet b1.58 2B
    - **Parameters:** 2 Billion
    - **Quantization:** 1.58-bit
    - **Memory:** ~400MB
    - **Context:** 4096 tokens
    """)
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

with col1:
    st.subheader("💬 Chat Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if len(st.session_state.messages) == 0:
        st.info("👋 Welcome! Try one of these example prompts:")
        example_prompts = [
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence",
            "What are the benefits of 1-bit LLMs?",
            "Tell me an interesting fact about space"
        ]
        
        cols = st.columns(2)
        for idx, prompt in enumerate(example_prompts):
            col_idx = idx % 2
            if cols[col_idx].button(f"📝 {prompt}", key=f"example_{idx}"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages_for_model = [
                    {"role": "system", "content": "You are a helpful AI assistant."}
                ] + st.session_state.messages
                
                response = generate_response(
                    model, 
                    tokenizer, 
                    messages_for_model, 
                    temperature, 
                    max_tokens
                )
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

st.markdown("---")
st.caption("Built with Streamlit | Model: microsoft/bitnet-b1.58-2B-4T")
