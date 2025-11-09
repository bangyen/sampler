import os
from datetime import datetime
import hashlib
from sqlalchemy.orm import Session
from storage.database import get_session, NERAnalysis


def generate_ner_id(text):
    """Generate a unique ID for NER analysis based on text hash"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def save_ner_analysis(text, entities, model, processing_time):
    """Save NER analysis to database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        ner_id = generate_ner_id(text)

        existing = db_session.query(NERAnalysis).filter_by(analysis_id=ner_id).first()
        if existing:
            db_session.query(NERAnalysis).filter_by(analysis_id=ner_id).update({
                "text": text,
                "entities": entities,
                "model": model,
                "processing_time": processing_time,
            })
        else:
            analysis = NERAnalysis(
                analysis_id=ner_id,
                text=text,
                entities=entities,
                model=model,
                processing_time=processing_time,
            )
            db_session.add(analysis)

        db_session.commit()
        db_session.close()
        return ner_id
    except Exception as e:
        print(f"Error saving NER analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


def load_ner_analysis(ner_id):
    """Load NER analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        analysis = db_session.query(NERAnalysis).filter_by(analysis_id=ner_id).first()

        if not analysis:
            db_session.close()
            return None

        text_content = str(analysis.text) if analysis.text is not None else ""
        result = {
            "id": analysis.analysis_id,
            "text": text_content,
            "entities": analysis.entities,
            "model": analysis.model,
            "processing_time": analysis.processing_time,
            "text_length": len(text_content),
            "created_at": analysis.created_at.isoformat(),
        }

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading NER analysis: {e}")
        if db_session:
            db_session.close()
        return None


def get_all_ner_analyses():
    """Get all NER analyses"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return []

        analyses = (
            db_session.query(NERAnalysis)
            .order_by(NERAnalysis.created_at.desc())
            .limit(50)
            .all()
        )

        result = []
        for analysis in analyses:
            text_content = str(analysis.text) if analysis.text is not None else ""
            entities_data = analysis.entities if analysis.entities is not None else []
            result.append(
                {
                    "id": analysis.analysis_id,
                    "text_preview": (
                        text_content[:100] + "..."
                        if len(text_content) > 100
                        else text_content
                    ),
                    "entity_count": len(entities_data) if isinstance(entities_data, list) else 0,
                    "model": analysis.model,
                    "created_at": analysis.created_at,
                }
            )

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading NER analyses: {e}")
        if db_session:
            db_session.close()
        return []


def delete_ner_analysis(ner_id):
    """Delete a NER analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(NERAnalysis).filter_by(analysis_id=ner_id).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error deleting NER analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False


def clear_all_ner_analyses():
    """Delete all NER analyses from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(NERAnalysis).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error clearing all NER analyses: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False
