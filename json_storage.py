import json
import os
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path("conversations")
STORAGE_DIR.mkdir(exist_ok=True)

def get_conversation_file(session_id):
    """Get the file path for a conversation"""
    return STORAGE_DIR / f"{session_id}.json"

def save_conversation(session_id, messages):
    """Save conversation to JSON file"""
    try:
        conversation_data = {
            "session_id": session_id,
            "messages": messages,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        file_path = get_conversation_file(session_id)
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return False

def load_conversation(session_id):
    """Load conversation from JSON file"""
    try:
        file_path = get_conversation_file(session_id)
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get("messages", [])
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []

def get_all_conversations():
    """Get all conversation sessions"""
    try:
        conversations = []
        
        for file_path in STORAGE_DIR.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                messages = data.get("messages", [])
                first_user_message = next((msg for msg in messages if msg.get("role") == "user"), None)
                first_message_preview = first_user_message["content"][:60] if first_user_message else "No messages"
                
                conversations.append({
                    "session_id": data["session_id"],
                    "message_count": len(messages),
                    "first_message": first_message_preview,
                    "updated_at": datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
                    "created_at": datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
                })
            except Exception:
                continue
        
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def delete_conversation(session_id):
    """Delete a conversation JSON file"""
    try:
        file_path = get_conversation_file(session_id)
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False
