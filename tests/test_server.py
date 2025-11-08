"""Tests for FastAPI server endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Create test client with mocked backends."""
    with patch("server.LLAMA_CPP_AVAILABLE", False):
        with patch("server.BITNET_CPP_AVAILABLE", False):
            with patch("server.DATABASE_AVAILABLE", True):
                from server import app
                return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_models(client):
    """Test getting available models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "models" in data


def test_get_conversations_endpoint(client):
    """Test getting conversations endpoint structure."""
    response = client.get("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data


def test_get_conversation_by_id(client):
    """Test getting a specific conversation."""
    response = client.get("/api/conversations/test-session")
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data


def test_delete_conversation(client):
    """Test deleting a conversation."""
    with patch("server.delete_conversation", return_value=True):
        response = client.delete("/api/conversations/test-session")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data


def test_get_ner_models(client):
    """Test getting NER models."""
    response = client.get("/api/ner/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


def test_get_ocr_configs(client):
    """Test getting OCR configurations."""
    response = client.get("/api/ocr/configs")
    assert response.status_code == 200
    data = response.json()
    assert "configs" in data


def test_invalid_endpoint(client):
    """Test accessing invalid endpoint."""
    response = client.get("/api/invalid-endpoint-xyz")
    assert response.status_code == 404
