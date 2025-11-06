import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import streamlit as st

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL environment variable is not set. Conversation persistence is disabled.")
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

@st.cache_resource
def get_engine():
    """Create and cache database engine"""
    if not DATABASE_URL:
        return None
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine

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
        
        conversation = db_session.query(Conversation).filter_by(session_id=session_id).first()
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
                metrics=msg.get("metrics")
            )
            db_session.add(message)
        
        conversation.updated_at = datetime.utcnow()
        db_session.commit()
        db_session.close()
        return True
    except Exception as e:
        st.error(f"Error saving conversation: {str(e)}")
        return False

def load_conversation(session_id):
    """Load conversation from database"""
    if not DATABASE_URL:
        return []
    try:
        db_session = get_session()
        if not db_session:
            return []
        
        messages = db_session.query(Message).filter_by(session_id=session_id).order_by(Message.created_at).all()
        
        result = []
        for msg in messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.metrics:
                message_dict["metrics"] = msg.metrics
            result.append(message_dict)
        
        db_session.close()
        return result
    except Exception as e:
        st.error(f"Error loading conversation: {str(e)}")
        return []

def get_all_conversations():
    """Get all conversation sessions"""
    if not DATABASE_URL:
        return []
    try:
        db_session = get_session()
        if not db_session:
            return []
        conversations = db_session.query(Conversation).order_by(Conversation.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            message_count = db_session.query(Message).filter_by(session_id=conv.session_id).count()
            result.append({
                "session_id": conv.session_id,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": message_count
            })
        
        db_session.close()
        return result
    except Exception as e:
        st.error(f"Error loading conversations: {str(e)}")
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
    except Exception as e:
        st.error(f"Error deleting conversation: {str(e)}")
        return False
