"""Tests for JSON-based conversation storage."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch
from storage import json_storage


def test_save_and_load_conversation(tmp_path, sample_conversation):
    """Test saving and loading a conversation."""
    session_id = "test-session-123"
    
    # Monkeypatch STORAGE_DIR to use tmp_path
    with patch.object(json_storage, 'STORAGE_DIR', tmp_path):
        # Save conversation
        result = json_storage.save_conversation(session_id, sample_conversation)
        assert result is True
        
        # Verify file exists
        file_path = tmp_path / f"{session_id}.json"
        assert file_path.exists()
        
        # Load conversation
        loaded = json_storage.load_conversation(session_id)
        assert loaded == sample_conversation


def test_load_nonexistent_conversation(tmp_path):
    """Test loading a conversation that doesn't exist."""
    with patch.object(json_storage, 'STORAGE_DIR', tmp_path):
        loaded = json_storage.load_conversation("nonexistent-id")
        assert loaded == []


def test_get_all_conversations(tmp_path, sample_conversation):
    """Test getting all conversations."""
    with patch.object(json_storage, 'STORAGE_DIR', tmp_path):
        # Save multiple conversations
        json_storage.save_conversation("session-1", sample_conversation)
        json_storage.save_conversation("session-2", sample_conversation[:2])
        
        # Get all conversations
        all_convs = json_storage.get_all_conversations()
        assert len(all_convs) == 2
        
        # Verify structure
        session_ids = [conv["session_id"] for conv in all_convs]
        assert "session-1" in session_ids
        assert "session-2" in session_ids


def test_delete_conversation(tmp_path, sample_conversation):
    """Test deleting a conversation."""
    session_id = "test-delete"
    
    with patch.object(json_storage, 'STORAGE_DIR', tmp_path):
        # Save conversation
        json_storage.save_conversation(session_id, sample_conversation)
        assert (tmp_path / f"{session_id}.json").exists()
        
        # Delete conversation
        result = json_storage.delete_conversation(session_id)
        assert result is True
        assert not (tmp_path / f"{session_id}.json").exists()


def test_delete_nonexistent_conversation(tmp_path):
    """Test deleting a conversation that doesn't exist.
    
    Note: Current implementation returns True even if file doesn't exist,
    as long as no exception occurs. This is expected behavior.
    """
    with patch.object(json_storage, 'STORAGE_DIR', tmp_path):
        result = json_storage.delete_conversation("nonexistent")
        # Implementation returns True even if file doesn't exist
        assert result is True
