import json
import os
from datetime import datetime
from pathlib import Path
import hashlib
import base64

STORAGE_DIR = Path("ocr_history")
STORAGE_DIR.mkdir(exist_ok=True)

def generate_ocr_id(image_data):
    """Generate a unique ID for OCR analysis based on image hash"""
    return hashlib.md5(image_data).hexdigest()[:16]

def get_ocr_file(ocr_id):
    """Get the file path for an OCR analysis"""
    return STORAGE_DIR / f"{ocr_id}.json"

def save_ocr_analysis(image_data, filename, extracted_text, bounding_boxes, config, processing_time, num_detections):
    """Save OCR analysis to JSON file with base64 encoded image"""
    try:
        ocr_id = generate_ocr_id(image_data)
        
        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        analysis_data = {
            "id": ocr_id,
            "filename": filename,
            "image_base64": image_base64,
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "config": config,
            "processing_time": processing_time,
            "num_detections": num_detections,
            "created_at": datetime.utcnow().isoformat()
        }
        
        file_path = get_ocr_file(ocr_id)
        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        return ocr_id
    except Exception as e:
        print(f"Error saving OCR analysis: {e}")
        return None

def load_ocr_analysis(ocr_id):
    """Load OCR analysis from JSON file"""
    try:
        file_path = get_ocr_file(ocr_id)
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading OCR analysis: {e}")
        return None

def get_all_ocr_analyses():
    """Get all OCR analyses (with thumbnail data for display)"""
    try:
        analyses = []
        
        for file_path in STORAGE_DIR.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                analyses.append({
                    "id": data["id"],
                    "filename": data.get("filename", "unknown"),
                    "text_preview": data["text"][:50] + "..." if len(data["text"]) > 50 else data["text"],
                    "image_base64": data.get("image_base64", ""),
                    "config": data.get("config", "Unknown"),
                    "num_detections": data.get("num_detections", 0),
                    "created_at": datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
                })
            except Exception:
                continue
        
        analyses.sort(key=lambda x: x["created_at"], reverse=True)
        return analyses[:50]  # Limit to 50 most recent
    except Exception as e:
        print(f"Error loading OCR analyses: {e}")
        return []

def delete_ocr_analysis(ocr_id):
    """Delete an OCR analysis JSON file"""
    try:
        file_path = get_ocr_file(ocr_id)
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting OCR analysis: {e}")
        return False
