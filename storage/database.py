import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print(
        "DATABASE_URL environment variable is not set. Conversation persistence is disabled."
    )
    DATABASE_URL = None

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class NERAnalysis(Base):
    __tablename__ = "ner_analyses"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(String(255), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    entities = Column(JSON, nullable=False)
    model = Column(String(255), nullable=False)
    processing_time = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class OCRAnalysis(Base):
    __tablename__ = "ocr_analyses"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    image_base64 = Column(Text, nullable=False)
    results = Column(JSON, nullable=False)
    config = Column(JSON, nullable=False)
    processing_time = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ZeroShotAnalysis(Base):
    __tablename__ = "zero_shot_analyses"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(String(255), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    candidate_labels = Column(JSON, nullable=False)
    results = Column(JSON, nullable=False)
    model = Column(String(255), nullable=False)
    processing_time = Column(JSON, nullable=True)
    use_logprobs = Column(JSON, nullable=True)
    abstain_threshold = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine_cache = None


def get_engine():
    """Create and cache database engine"""
    global _engine_cache
    if _engine_cache is not None:
        return _engine_cache
    
    if not DATABASE_URL:
        return None
    
    _engine_cache = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(_engine_cache)
    return _engine_cache


def get_session():
    """Create a new database session"""
    engine = get_engine()
    if not engine:
        return None
    Session = sessionmaker(bind=engine)
    return Session()


def save_conversation(session_id, messages):
    """Save conversation to database"""
    if not DATABASE_URL:
        return False
    try:
        db_session = get_session()
        if not db_session:
            return False

        conversation = (
            db_session.query(Conversation).filter_by(session_id=session_id).first()
        )
        if not conversation:
            conversation = Conversation(session_id=session_id)
            db_session.add(conversation)
            db_session.commit()

        db_session.query(Message).filter_by(session_id=session_id).delete()

        for msg in messages:
            message = Message(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
                metrics=msg.get("metrics"),
            )
            db_session.add(message)

        db_session.query(Conversation).filter_by(session_id=session_id).update(
            {"updated_at": datetime.utcnow()}
        )
        db_session.commit()
        db_session.close()
        return True
    except Exception:
        return False


def load_conversation(session_id):
    """Load conversation from database"""
    if not DATABASE_URL:
        return []
    try:
        db_session = get_session()
        if not db_session:
            return []

        messages = (
            db_session.query(Message)
            .filter_by(session_id=session_id)
            .order_by(Message.created_at)
            .all()
        )

        result = []
        for msg in messages:
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.metrics is not None:
                message_dict["metrics"] = msg.metrics
            result.append(message_dict)

        db_session.close()
        return result
    except Exception:
        return []


def get_all_conversations():
    """Get all conversation sessions"""
    if not DATABASE_URL:
        return []
    try:
        db_session = get_session()
        if not db_session:
            return []
        conversations = (
            db_session.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

        result = []
        for conv in conversations:
            message_count = (
                db_session.query(Message).filter_by(session_id=conv.session_id).count()
            )

            first_user_message = (
                db_session.query(Message)
                .filter_by(session_id=conv.session_id, role="user")
                .order_by(Message.created_at)
                .first()
            )

            first_message_preview = (
                first_user_message.content[:60] if first_user_message else "No messages"
            )

            result.append(
                {
                    "session_id": conv.session_id,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "message_count": message_count,
                    "first_message": first_message_preview,
                }
            )

        db_session.close()
        return result
    except Exception:
        return []


def delete_conversation(session_id):
    """Delete a conversation from database"""
    if not DATABASE_URL:
        return False
    try:
        db_session = get_session()
        if not db_session:
            return False
        db_session.query(Message).filter_by(session_id=session_id).delete()
        db_session.query(Conversation).filter_by(session_id=session_id).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception:
        return False


def clear_all_conversations():
    """Delete all conversations from database"""
    if not DATABASE_URL:
        return False
    try:
        db_session = get_session()
        if not db_session:
            return False
        db_session.query(Message).delete()
        db_session.query(Conversation).delete()
        db_session.commit()
        db_session.close()
        return True
    except Exception:
        return False
