import os
from datetime import datetime
import hashlib
import base64
from storage.database import get_session, LayoutAnalysis


def generate_layout_id(image_data):
    """Generate a unique ID for layout analysis based on image hash"""
    return hashlib.md5(image_data).hexdigest()[:16]


def save_layout_analysis(
    image_data,
    filename,
    extracted_text,
    bounding_boxes,
    processing_time,
    num_detections,
):
    """Save layout analysis to database with base64 encoded image"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        layout_id = generate_layout_id(image_data)
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        results = {
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "num_detections": num_detections,
        }

        existing = db_session.query(LayoutAnalysis).filter_by(analysis_id=layout_id).first()
        if existing:
            db_session.query(LayoutAnalysis).filter_by(analysis_id=layout_id).update({
                "filename": filename,
                "image_base64": image_base64,
                "results": results,
                "processing_time": processing_time,
            })
        else:
            analysis = LayoutAnalysis(
                analysis_id=layout_id,
                filename=filename,
                image_base64=image_base64,
                results=results,
                processing_time=processing_time,
            )
            db_session.add(analysis)

        db_session.commit()
        db_session.close()
        return layout_id
    except Exception as e:
        print(f"Error saving layout analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


def load_layout_analysis(layout_id):
    """Load layout analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        analysis = db_session.query(LayoutAnalysis).filter_by(analysis_id=layout_id).first()

        if not analysis:
            db_session.close()
            return None

        result = {
            "id": analysis.analysis_id,
            "filename": analysis.filename,
            "image_base64": analysis.image_base64,
            "text": analysis.results.get("text", ""),
            "bounding_boxes": analysis.results.get("bounding_boxes", []),
            "processing_time": analysis.processing_time,
            "num_detections": analysis.results.get("num_detections", 0),
            "created_at": analysis.created_at.isoformat(),
        }

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading layout analysis: {e}")
        if db_session:
            db_session.close()
        return None


def get_all_layout_analyses():
    """Get all layout analyses"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return []

        analyses = (
            db_session.query(LayoutAnalysis)
            .order_by(LayoutAnalysis.created_at.desc())
            .limit(50)
            .all()
        )

        result = []
        for analysis in analyses:
            text_content = analysis.results.get("text", "")
            result.append(
                {
                    "id": analysis.analysis_id,
                    "filename": analysis.filename,
                    "text_preview": (
                        text_content[:50] + "..." if len(text_content) > 50 else text_content
                    ),
                    "image_base64": analysis.image_base64,
                    "num_detections": analysis.results.get("num_detections", 0),
                    "created_at": analysis.created_at,
                }
            )

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading layout analyses: {e}")
        if db_session:
            db_session.close()
        return []


def delete_layout_analysis(layout_id):
    """Delete a layout analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(LayoutAnalysis).filter_by(analysis_id=layout_id).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error deleting layout analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False


def clear_all_layout_analyses():
    """Delete all layout analyses from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(LayoutAnalysis).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error clearing all layout analyses: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False
