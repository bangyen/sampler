import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


ZERO_SHOT_STORAGE_DIR = Path("conversations") / "zero_shot"
ZERO_SHOT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_zero_shot_analysis(
    text: str,
    labels: List[str],
    results: Dict[str, Any],
    model: str,
    processing_time: float,
    use_logprobs: bool = True,
    abstain_threshold: Optional[float] = None
) -> Optional[str]:
    timestamp = datetime.now().isoformat()
    analysis_id = f"zs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    analysis_data = {
        "id": analysis_id,
        "timestamp": timestamp,
        "text": text,
        "candidate_labels": labels,
        "results": results,
        "model": model,
        "processing_time": processing_time,
        "use_logprobs": use_logprobs,
        "abstain_threshold": abstain_threshold,
        "text_length": len(text),
    }
    
    file_path = ZERO_SHOT_STORAGE_DIR / f"{analysis_id}.json"
    
    try:
        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return analysis_id
    except Exception as e:
        print(f"Error saving zero-shot analysis: {e}")
        return None


def load_zero_shot_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    file_path = ZERO_SHOT_STORAGE_DIR / f"{analysis_id}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading zero-shot analysis {analysis_id}: {e}")
        return None


def get_all_zero_shot_analyses(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    analyses = []
    
    try:
        all_files = sorted(ZERO_SHOT_STORAGE_DIR.glob("zs_*.json"), reverse=True)
        total_count = len(all_files)
        
        files_to_process = all_files[offset:offset + limit] if limit else all_files[offset:]
        
        for file_path in files_to_process:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    analyses.append({
                        "id": data["id"],
                        "timestamp": data["timestamp"],
                        "text_preview": data["text"][:100] + "..." if len(data["text"]) > 100 else data["text"],
                        "top_label": data["results"].get("top_label"),
                        "top_score": data["results"].get("top_score"),
                        "model": data["model"],
                        "candidate_labels": data["candidate_labels"],
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
    except Exception as e:
        print(f"Error listing zero-shot analyses: {e}")
        total_count = 0
    
    return {
        "analyses": analyses,
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(analyses)) < total_count
    }


def delete_zero_shot_analysis(analysis_id: str) -> bool:
    file_path = ZERO_SHOT_STORAGE_DIR / f"{analysis_id}.json"
    
    if not file_path.exists():
        return False
    
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting zero-shot analysis {analysis_id}: {e}")
        return False


def clear_all_zero_shot_analyses() -> bool:
    try:
        for file_path in ZERO_SHOT_STORAGE_DIR.glob("zs_*.json"):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        return True
    except Exception as e:
        print(f"Error clearing zero-shot analyses: {e}")
        return False
