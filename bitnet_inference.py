"""
BitNet.cpp inference module using llama-cpp-python bindings.

This module provides faster inference for quantized LLM models using llama.cpp backend,
which offers 1.4-6x speedup compared to standard PyTorch CPU inference.
"""

import os
from typing import Iterator, List, Dict, Optional
import time

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class BitNetInference:
    """Wrapper for llama-cpp-python for efficient GGUF model inference"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        Initialize BitNet inference engine.
        
        Args:
            model_path: Path to GGUF format model file
            n_ctx: Context window size (default: 2048)
            n_threads: Number of threads for inference (default: auto-detect)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        
        # Initialize the model
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
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
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False
    ) -> Iterator[str]:
        """
        Generate response from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            stream: Whether to stream tokens
            
        Yields:
            Generated text tokens (if stream=True) or full response (if stream=False)
        """
        prompt = self.format_chat_prompt(messages)
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
            stream=stream
        )
        
        if stream:
            # Stream tokens one by one
            for chunk in output:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('text', '')
                    if delta:
                        yield delta
        else:
            # Return complete response
            if 'choices' in output and len(output['choices']) > 0:
                yield output['choices'][0]['text']
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "backend": "llama.cpp"
        }


def is_available() -> bool:
    """Check if llama-cpp-python is available"""
    return LLAMA_CPP_AVAILABLE


def download_gguf_model(hf_repo: str, filename: str = None, cache_dir: str = "./models") -> str:
    """
    Download a GGUF model from Hugging Face.
    
    Args:
        hf_repo: Hugging Face repository ID (e.g., "microsoft/bitnet-b1.58-2B-4T-gguf")
        filename: GGUF file name in the repo (optional, will auto-detect if not provided)
        cache_dir: Local directory to cache the model
        
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Auto-detect GGUF filename if not provided
    if filename is None:
        files = list_repo_files(repo_id=hf_repo)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if not gguf_files:
            raise ValueError(f"No GGUF files found in repository {hf_repo}")
        # Prefer i2_s quantization if available
        i2s_files = [f for f in gguf_files if 'i2_s' in f.lower()]
        filename = i2s_files[0] if i2s_files else gguf_files[0]
        print(f"Auto-detected GGUF file: {filename}")
    
    local_dir = os.path.join(cache_dir, hf_repo.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)
    
    model_path = hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
        local_dir=local_dir
    )
    
    return model_path


# Example usage
if __name__ == "__main__":
    if not is_available():
        print("llama-cpp-python not available. Install with: pip install llama-cpp-python")
    else:
        print("BitNet inference module ready!")
        print(f"Using {os.cpu_count()} CPU threads")
