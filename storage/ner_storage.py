import json
from datetime import datetime
from pathlib import Path
import hashlib

STORAGE_DIR = Path("ner_history")
STORAGE_DIR.mkdir(exist_ok=True)


def generate_ner_id(text):
    """Generate a unique ID for NER analysis based on text hash"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def get_ner_file(ner_id):
    """Get the file path for a NER analysis"""
    return STORAGE_DIR / f"{ner_id}.json"


def save_ner_analysis(text, entities, model, processing_time):
    """Save NER analysis to JSON file"""
    try:
        ner_id = generate_ner_id(text)

        analysis_data = {
            "id": ner_id,
            "text": text,
            "entities": entities,
            "model": model,
            "processing_time": processing_time,
            "text_length": len(text),
            "created_at": datetime.utcnow().isoformat(),
        }

        file_path = get_ner_file(ner_id)
        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return ner_id
    except Exception as e:
        print(f"Error saving NER analysis: {e}")
        return None


def load_ner_analysis(ner_id):
    """Load NER analysis from JSON file"""
    try:
        file_path = get_ner_file(ner_id)
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading NER analysis: {e}")
        return None


def get_all_ner_analyses():
    """Get all NER analyses"""
    try:
        analyses = []

        for file_path in STORAGE_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                analyses.append(
                    {
                        "id": data["id"],
                        "text_preview": (
                            data["text"][:100] + "..."
                            if len(data["text"]) > 100
                            else data["text"]
                        ),
                        "entity_count": len(data.get("entities", [])),
                        "model": data.get("model", "Unknown"),
                        "created_at": datetime.fromisoformat(
                            data.get("created_at", datetime.utcnow().isoformat())
                        ),
                    }
                )
            except Exception:
                continue

        analyses.sort(key=lambda x: x["created_at"], reverse=True)
        return analyses[:50]  # Limit to 50 most recent
    except Exception as e:
        print(f"Error loading NER analyses: {e}")
        return []


def delete_ner_analysis(ner_id):
    """Delete a NER analysis JSON file"""
    try:
        file_path = get_ner_file(ner_id)
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting NER analysis: {e}")
        return False


def clear_all_ner_analyses():
    """Delete all NER analysis files"""
    try:
        count = 0
        for file_path in STORAGE_DIR.glob("*.json"):
            try:
                file_path.unlink()
                count += 1
            except Exception:
                continue
        return True
    except Exception as e:
        print(f"Error clearing all NER analyses: {e}")
        return False
