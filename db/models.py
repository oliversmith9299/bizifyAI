from sqlalchemy import Column, String, JSON, Text, DateTime
from datetime import datetime

# Shared base from connection
from db.connection import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    # FIX 4: Removed redundant 'id', making user_id the strict Primary Key
    user_id      = Column(String, primary_key=True) 
    status       = Column(String, default="pending")  
    current_step = Column(String, nullable=True)      
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProfileResult(Base):
    __tablename__ = "profile_results"
    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuestionnaireOutput(Base):
    __tablename__ = "questionnaire_outputs"
    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProblemsResult(Base):
    __tablename__ = "problems_results"
    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class IdeaResult(Base):
    __tablename__ = "idea_results"
    user_id      = Column(String, primary_key=True)
    current_idea = Column(Text, nullable=True)
    chat_history = Column(JSON, default=list)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)