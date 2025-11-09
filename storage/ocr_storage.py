import os
from datetime import datetime
import hashlib
import base64
from storage.database import get_session, OCRAnalysis


def generate_ocr_id(image_data):
    """Generate a unique ID for OCR analysis based on image hash"""
    return hashlib.md5(image_data).hexdigest()[:16]


def save_ocr_analysis(
    image_data,
    filename,
    extracted_text,
    bounding_boxes,
    config,
    processing_time,
    num_detections,
):
    """Save OCR analysis to database with base64 encoded image"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        ocr_id = generate_ocr_id(image_data)
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        results = {
            "text": extracted_text,
            "bounding_boxes": bounding_boxes,
            "num_detections": num_detections,
        }

        existing = db_session.query(OCRAnalysis).filter_by(analysis_id=ocr_id).first()
        if existing:
            db_session.query(OCRAnalysis).filter_by(analysis_id=ocr_id).update({
                "filename": filename,
                "image_base64": image_base64,
                "results": results,
                "config": config,
                "processing_time": processing_time,
            })
        else:
            analysis = OCRAnalysis(
                analysis_id=ocr_id,
                filename=filename,
                image_base64=image_base64,
                results=results,
                config=config,
                processing_time=processing_time,
            )
            db_session.add(analysis)

        db_session.commit()
        db_session.close()
        return ocr_id
    except Exception as e:
        print(f"Error saving OCR analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


def load_ocr_analysis(ocr_id):
    """Load OCR analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return None

        analysis = db_session.query(OCRAnalysis).filter_by(analysis_id=ocr_id).first()

        if not analysis:
            db_session.close()
            return None

        result = {
            "id": analysis.analysis_id,
            "filename": analysis.filename,
            "image_base64": analysis.image_base64,
            "text": analysis.results.get("text", ""),
            "bounding_boxes": analysis.results.get("bounding_boxes", []),
            "config": analysis.config,
            "processing_time": analysis.processing_time,
            "num_detections": analysis.results.get("num_detections", 0),
            "created_at": analysis.created_at.isoformat(),
        }

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading OCR analysis: {e}")
        if db_session:
            db_session.close()
        return None


def get_all_ocr_analyses():
    """Get all OCR analyses (with thumbnail data for display)"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return []

        analyses = (
            db_session.query(OCRAnalysis)
            .order_by(OCRAnalysis.created_at.desc())
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
                    "config": analysis.config,
                    "num_detections": analysis.results.get("num_detections", 0),
                    "created_at": analysis.created_at,
                }
            )

        db_session.close()
        return result
    except Exception as e:
        print(f"Error loading OCR analyses: {e}")
        if db_session:
            db_session.close()
        return []


def delete_ocr_analysis(ocr_id):
    """Delete an OCR analysis from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(OCRAnalysis).filter_by(analysis_id=ocr_id).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error deleting OCR analysis: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False


def clear_all_ocr_analyses():
    """Delete all OCR analyses from database"""
    db_session = None
    try:
        db_session = get_session()
        if not db_session:
            return False

        db_session.query(OCRAnalysis).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        print(f"Error clearing all OCR analyses: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return False
