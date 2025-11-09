import json
from datetime import datetime
from pathlib import Path
import hashlib
import base64

STORAGE_DIR = Path("data/layout_history")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def generate_layout_id(image_data):
    """Generate a unique ID for layout analysis based on image hash"""
    return hashlib.md5(image_data).hexdigest()[:16]


def get_layout_file(layout_id):
    """Get the file path for a layout analysis"""
    return STORAGE_DIR / f"{layout_id}.json"


def save_layout_analysis(
    image_data,
    filename,
    extracted_text,
    bounding_boxes,
    processing_time,
    num_detections,
):
    """Save layout analysis to JSON file with base64 encoded image"""
    try:
        layout_id = generate_layout_id(image_data)

        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        analysis_data = {
            "id": layout_id,
            "filename": filename,
            "image_base64": image_base64,
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "processing_time": processing_time,
            "num_detections": num_detections,
            "created_at": datetime.utcnow().isoformat(),
        }

        file_path = get_layout_file(layout_id)
        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return layout_id
    except Exception as e:
        print(f"Error saving layout analysis: {e}")
        return None


def load_layout_analysis(layout_id):
    """Load layout analysis from JSON file"""
    try:
        file_path = get_layout_file(layout_id)
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading layout analysis: {e}")
        return None


def get_all_layout_analyses():
    """Get all layout analyses"""
    try:
        analyses = []

        for file_path in STORAGE_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                analyses.append(
                    {
                        "id": data["id"],
                        "filename": data.get("filename", "unknown"),
                        "text_preview": (
                            data["text"][:50] + "..."
                            if len(data["text"]) > 50
                            else data["text"]
                        ),
                        "image_base64": data.get("image_base64", ""),
                        "num_detections": data.get("num_detections", 0),
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
        print(f"Error loading layout analyses: {e}")
        return []


def delete_layout_analysis(layout_id):
    """Delete a layout analysis JSON file"""
    try:
        file_path = get_layout_file(layout_id)
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting layout analysis: {e}")
        return False


def clear_all_layout_analyses():
    """Delete all layout analysis files"""
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
        print(f"Error clearing all layout analyses: {e}")
        return False
