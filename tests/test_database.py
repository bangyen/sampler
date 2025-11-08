"""Tests for database-based conversation storage."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from storage import database


def test_get_engine_with_url():
    """Test engine creation returns engine when DATABASE_URL is set."""
    with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
        with patch('storage.database.create_engine') as mock_create:
            with patch('storage.database.Base.metadata.create_all'):
                # Reset the global cache
                database._engine_cache = None
                
                mock_engine = Mock()
                mock_create.return_value = mock_engine
                
                # Call get_engine to test caching logic
                engine = database.get_engine()
                assert engine == mock_engine
                assert mock_create.called


def test_save_conversation_returns_false_without_db():
    """Test save conversation returns False when DATABASE_URL is None."""
    with patch('storage.database.DATABASE_URL', None):
        result = database.save_conversation("test-id", [])
        assert result is False


def test_load_conversation_returns_empty_without_db():
    """Test load conversation returns empty list when DATABASE_URL is None."""
    with patch('storage.database.DATABASE_URL', None):
        result = database.load_conversation("test-id")
        assert result == []


def test_delete_conversation_returns_false_without_db():
    """Test delete conversation returns False when DATABASE_URL is None."""
    with patch('storage.database.DATABASE_URL', None):
        result = database.delete_conversation("test-id")
        assert result is False


def test_get_all_conversations_returns_empty_without_db():
    """Test get all conversations returns empty list when DATABASE_URL is None."""
    with patch('storage.database.DATABASE_URL', None):
        result = database.get_all_conversations()
        assert result == []
