"""Pytest configuration and shared fixtures."""
import pytest
import os
import tempfile
import shutil
from pathlib import Path

# Set TESTING mode globally for all tests to prevent heavy model imports
os.environ["TESTING"] = "1"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def conversations_dir(temp_dir):
    """Create a temporary conversations directory."""
    conv_dir = os.path.join(temp_dir, "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    return conv_dir


@pytest.fixture
def mock_env_no_db(monkeypatch):
    """Mock environment with no database."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGPORT", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
        {
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
        },
    ]
