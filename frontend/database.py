import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "chatbot.db")
engine = create_engine(f"sqlite:///{DB_FILE}")
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'sessions'
    
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    prompt = Column(Text)
    response = Column(Text)
    rating = Column(Integer) # 1 to 5
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_message(session_id, role, content):
    db = SessionLocal()
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.close()

def get_messages(session_id):
    db = SessionLocal()
    msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
    db.close()
    return msgs

def save_feedback(session_id, prompt, response, rating):
    db = SessionLocal()
    fb = Feedback(session_id=session_id, prompt=prompt, response=response, rating=rating)
    db.add(fb)
    db.commit()
    db.close()


def delete_session(session_id: str):
    """Delete a chat session and all its associated messages/feedback."""
    db = SessionLocal()
    try:
        db.query(Feedback).filter(Feedback.session_id == session_id).delete()
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        db.query(ChatSession).filter(ChatSession.id == session_id).delete()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_all_sessions():
    from sqlalchemy import func
    db = SessionLocal()
    
    # Group by session_id, order by the most recent message
    session_data = db.query(
        ChatMessage.session_id, 
        func.max(ChatMessage.timestamp).label('last_active'),
        func.min(ChatMessage.id).label('first_msg_id')
    ).group_by(ChatMessage.session_id).order_by(func.max(ChatMessage.timestamp).desc()).all()
    
    sessions = []
    for s_id, last_active, first_msg_id in session_data:
        # Get the first message for a title preview
        first_msg = db.query(ChatMessage).filter(ChatMessage.id == first_msg_id).first()
        title = "New Chat"
        if first_msg:
            title = first_msg.content[:30] + "..." if len(first_msg.content) > 30 else first_msg.content
        sessions.append({"session_id": s_id, "title": title})
        
    db.close()
    return sessions
