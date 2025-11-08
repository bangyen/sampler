"""Tests for BitNet llama.cpp inference."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os


def test_is_available():
    """Test availability check."""
    from bitnet_inference import is_available, LLAMA_CPP_AVAILABLE
    assert is_available() == LLAMA_CPP_AVAILABLE


def test_bitnet_inference_init_file_not_found():
    """Test BitNetInference raises error for non-existent model file."""
    with patch("bitnet_inference.LLAMA_CPP_AVAILABLE", True):
        from bitnet_inference import BitNetInference
        with pytest.raises(FileNotFoundError):
            BitNetInference("nonexistent-model.gguf")


def test_bitnet_inference_init_success():
    """Test BitNetInference initialization with mocked model."""
    with patch("bitnet_inference.LLAMA_CPP_AVAILABLE", True):
        with patch("bitnet_inference.Llama") as mock_llama:
            with patch("os.path.exists", return_value=True):
                from bitnet_inference import BitNetInference
                
                mock_model = Mock()
                mock_llama.return_value = mock_model
                
                inference = BitNetInference("test-model.gguf")
                assert inference.model == mock_model
                assert inference.model_path == "test-model.gguf"


def test_generate_basic():
    """Test basic text generation."""
    with patch("bitnet_inference.LLAMA_CPP_AVAILABLE", True):
        with patch("bitnet_inference.Llama") as mock_llama:
            with patch("os.path.exists", return_value=True):
                from bitnet_inference import BitNetInference
                
                mock_model = Mock()
                mock_model.return_value = {
                    "choices": [{"text": " generated text"}]
                }
                mock_llama.return_value = mock_model
                
                inference = BitNetInference("test-model.gguf")
                messages = [{"role": "user", "content": "test prompt"}]
                result = list(inference.generate(messages))
                
                assert len(result) > 0
                assert "generated text" in result[0]


def test_generate_stream():
    """Test streaming text generation."""
    with patch("bitnet_inference.LLAMA_CPP_AVAILABLE", True):
        with patch("bitnet_inference.Llama") as mock_llama:
            with patch("os.path.exists", return_value=True):
                from bitnet_inference import BitNetInference
                
                # Mock streaming response
                mock_model = Mock()
                mock_model.return_value = iter([
                    {"choices": [{"text": "chunk1"}]},
                    {"choices": [{"text": "chunk2"}]},
                    {"choices": [{"text": "chunk3"}]},
                ])
                mock_llama.return_value = mock_model
                
                inference = BitNetInference("test-model.gguf")
                messages = [{"role": "user", "content": "test prompt"}]
                chunks = []
                
                for chunk in inference.generate(messages, stream=True):
                    chunks.append(chunk)
                
                assert len(chunks) == 3
                assert "chunk1" in chunks[0]


def test_download_gguf_model_isolated():
    """Test GGUF model download function with mocked dependencies."""
    # Patch the imports that happen inside the function
    with patch("huggingface_hub.hf_hub_download", return_value="/path/to/model.gguf") as mock_download:
        with patch("huggingface_hub.list_repo_files", return_value=["model.gguf"]):
            with patch("os.makedirs"):
                from bitnet_inference import download_gguf_model
                
                # Call with explicit filename to avoid auto-detection
                result = download_gguf_model("test-repo", "model.gguf")
                
                # Should return the mocked path
                assert result == "/path/to/model.gguf"
                assert mock_download.called
