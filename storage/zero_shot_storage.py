import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from storage.database import get_session, ZeroShotAnalysis


def save_zero_shot_analysis(
    text: str,
    labels: List[str],
    results: Dict[str, Any],
    model: str,
    processing_time: float,
    use_logprobs: bool = True,
    abstain_threshold: Optional[float] = None
) -> Optional[str]:
    """Save zero-shot analysis to database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        timestamp = datetime.now()
        analysis_id = f"zs_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        analysis = ZeroShotAnalysis(
            analysis_id=analysis_id,
            text=text,
            candidate_labels=labels,
            results=results,
            model=model,
            processing_time=processing_time,
            use_logprobs=use_logprobs,
            abstain_threshold=abstain_threshold,
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.close()
        return analysis_id
    except Exception as e:
        print(f"Error saving zero-shot analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


def load_zero_shot_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Load zero-shot analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        analysis = db_session.query(ZeroShotAnalysis).filter_by(analysis_id=analysis_id).first()

        if not analysis:
            db_session.close()
            return None

        text_value = str(analysis.text) if analysis.text is not None else ""
        result = {
            "id": analysis.analysis_id,
            "timestamp": analysis.created_at.isoformat(),
            "text": text_value,
            "candidate_labels": analysis.candidate_labels,
            "results": analysis.results,
            "model": analysis.model,
            "processing_time": analysis.processing_time,
            "use_logprobs": analysis.use_logprobs,
            "abstain_threshold": analysis.abstain_threshold,
            "text_length": len(text_value),
        }

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading zero-shot analysis {analysis_id}: {e}")
        if db_session:
            db_session.close()
        return None


def get_all_zero_shot_analyses(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """Get all zero-shot analyses with optional pagination"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return {
                "analyses": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "has_more": False
            }

        total_count = db_session.query(ZeroShotAnalysis).count()

        query = db_session.query(ZeroShotAnalysis).order_by(ZeroShotAnalysis.created_at.desc())
        
        if offset > 0:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)

        analyses = query.all()

        result = []
        for analysis in analyses:
            text_content = str(analysis.text) if analysis.text is not None else ""
            text_preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
            result.append({
                "id": analysis.analysis_id,
                "timestamp": analysis.created_at.isoformat(),
                "text_preview": text_preview,
                "top_label": analysis.results.get("top_label"),
                "top_score": analysis.results.get("top_score"),
                "model": analysis.model,
                "candidate_labels": analysis.candidate_labels,
            })

        db_session.close()
        
        return {
            "analyses": result,
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + len(result)) < total_count
        }
    except Exception as e:
        print(f"Error listing zero-shot analyses: {e}")
        if db_session:
            db_session.close()
        return {
            "analyses": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "has_more": False
        }


def delete_zero_shot_analysis(analysis_id: str) -> bool:
    """Delete a zero-shot analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(ZeroShotAnalysis).filter_by(analysis_id=analysis_id).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error deleting zero-shot analysis {analysis_id}: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False


def clear_all_zero_shot_analyses() -> bool:
    """Delete all zero-shot analyses from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(ZeroShotAnalysis).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error clearing zero-shot analyses: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False
