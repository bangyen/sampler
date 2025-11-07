"""
BitNet.cpp subprocess bridge for GGUF inference.
Wraps the compiled bin/BitNet/build/bin/main binary.
"""

import os
import subprocess
import sys
import threading
import codecs
from pathlib import Path
from typing import Iterator, Optional
from huggingface_hub import hf_hub_download

# Path to compiled BitNet binary
BITNET_BINARY = Path("bin/BitNet/build/bin/llama-cli")

# Default model configuration
DEFAULT_BITNET_REPO = "microsoft/bitnet-b1.58-2B-4T-gguf"
DEFAULT_BITNET_FILE = "ggml-model-i2_s.gguf"

class BitNetCppBridge:
    """Bridge to BitNet.cpp compiled binary for GGUF inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize BitNet bridge with model path."""
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = None
        
        # Verify binary exists
        if not BITNET_BINARY.exists():
            raise FileNotFoundError(
                f"BitNet binary not found at {BITNET_BINARY}. "
                "Please compile it first with: cd bin/BitNet/build && make"
            )
    
    @staticmethod
    def download_model(
        repo_id: str = DEFAULT_BITNET_REPO,
        filename: str = DEFAULT_BITNET_FILE,
        cache_dir: Optional[str] = None
    ) -> str:
        """Download BitNet GGUF model from Hugging Face."""
        print(f"Downloading BitNet GGUF model from {repo_id}...")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        print(f"Model downloaded to: {model_path}")
        return model_path
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        system_message: Optional[str] = None
    ) -> Iterator[str]:
        """
        Generate text using BitNet binary and stream tokens.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            system_message: Optional system message (prepended to prompt)
        
        Yields:
            Generated tokens as strings (character chunks for real-time streaming)
        """
        if not self.model_path or not self.model_path.exists():
            raise ValueError(
                "Model path not set or doesn't exist. "
                "Call download_model() first or provide model_path."
            )
        
        # Build full prompt with system message if provided
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # Build command arguments for BitNet binary
        # Based on llama.cpp interface: ./main -m model.gguf -p "prompt" -n tokens
        # Note: Logging goes to stderr, generated tokens go to stdout
        cmd = [
            str(BITNET_BINARY),
            "-m", str(self.model_path),
            "-p", full_prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "-t", str(os.cpu_count() or 4),  # Use all CPU threads
        ]
        
        process = None
        stderr_output = []
        
        def drain_stderr(pipe, output_list):
            """Drain stderr in background thread to avoid deadlock"""
            try:
                for line in pipe:
                    # Decode bytes to string
                    output_list.append(line.decode('utf-8', errors='replace'))
            except:
                pass
        
        try:
            # Launch subprocess in binary mode for unbuffered reading
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for character-by-character reading (requires binary mode)
            )
            
            # Start background thread to drain stderr
            stderr_thread = threading.Thread(
                target=drain_stderr,
                args=(process.stderr, stderr_output),
                daemon=True
            )
            stderr_thread.start()
            
            # Stream characters from stdout
            # Use incremental UTF-8 decoder to properly handle multi-byte characters
            buffer = ""
            chunk_size = 5  # Yield every N characters for smooth streaming
            total_output = ""  # Track all output to strip prompt
            prompt_stripped = False
            
            if process.stdout:
                # Create incremental UTF-8 decoder to handle multi-byte sequences
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                
                # Read one byte at a time and decode incrementally
                for byte_char in iter(lambda: process.stdout.read(1), b''):
                    if not byte_char:  # EOF
                        break
                    
                    # Decode byte (decoder accumulates bytes until complete codepoint)
                    char = decoder.decode(byte_char, False)
                    
                    if char:  # Only add if decoder returned a character
                        buffer += char
                        total_output += char
                        
                        # Strip the prompt prefix (llama.cpp echoes the prompt)
                        if not prompt_stripped and len(total_output) >= len(full_prompt):
                            # Check if output starts with the prompt
                            if total_output.startswith(full_prompt):
                                # Remove prompt from buffer and mark as stripped
                                buffer = total_output[len(full_prompt):]
                                prompt_stripped = True
                        
                        # Yield chunks of characters for smoother streaming (only after prompt stripped)
                        if prompt_stripped and len(buffer) >= chunk_size:
                            yield buffer
                            buffer = ""
                
                # Flush any remaining bytes in decoder and buffer
                final_chars = decoder.decode(b'', True)
                if final_chars:
                    buffer += final_chars
                    total_output += final_chars
                
                # Final prompt stripping check if not done yet
                if not prompt_stripped and total_output.startswith(full_prompt):
                    buffer = total_output[len(full_prompt):]
                
                if buffer:
                    yield buffer
            
            # Wait for process to complete
            process.wait(timeout=10)
            
            # Check for errors
            if process.returncode != 0:
                stderr_thread.join(timeout=0.5)
                stderr_text = "".join(stderr_output)
                raise RuntimeError(
                    f"BitNet binary exited with code {process.returncode}. "
                    f"Error: {stderr_text}"
                )
        
        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            raise RuntimeError("BitNet binary timeout - process killed")
        
        except Exception as e:
            if process and process.poll() is None:
                process.kill()
            print(f"Error during BitNet generation: {e}", file=sys.stderr)
            raise
        
        finally:
            # Ensure cleanup
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    process.kill()


def load_bitnet_model(
    repo_id: str = DEFAULT_BITNET_REPO,
    filename: str = DEFAULT_BITNET_FILE,
    cache_dir: Optional[str] = None
) -> BitNetCppBridge:
    """
    Load BitNet GGUF model and return bridge instance.
    
    Args:
        repo_id: Hugging Face repo ID
        filename: GGUF filename in the repo
        cache_dir: Optional cache directory
    
    Returns:
        BitNetCppBridge instance ready for generation
    """
    # Download model if not cached
    model_path = BitNetCppBridge.download_model(repo_id, filename, cache_dir)
    
    # Create and return bridge
    return BitNetCppBridge(model_path=model_path)
