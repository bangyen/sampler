"""Tests for FastAPI server endpoints."""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Create test client with lightweight mode enabled."""
    # TESTING mode is set globally in conftest.py
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
