# db/models.py
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Always import Base from db.connection — never create a new one.
# If each file creates its own Base(), SQLAlchemy won't know about all tables
# and create_all() will miss them.
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy import Column, String, JSON, Text, DateTime
from datetime import datetime

# Import the SHARED Base from connection — do NOT use declarative_base() here
from db.connection import Base


# ── Keep any existing models your team already defined above this line ────────
# (don't delete them — just add the 4 new ones below)


# ── AI Pipeline Tables ────────────────────────────────────────────────────────

class PipelineRun(Base):
    """Tracks pipeline execution status per user."""
    __tablename__ = "pipeline_runs"

    id           = Column(String, primary_key=True)   # same as user_id
    user_id      = Column(String, nullable=False, index=True)
    status       = Column(String, default="pending")  # pending|running|done|error
    current_step = Column(String, nullable=True)      # profile_analysis|problem_discovery|idea_generation
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProfileResult(Base):
    """Stores the profileAnalysis JSON output per user."""
    __tablename__ = "profile_results"

    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QuestionnaireOutput(Base):
    """Stores the original questionnaireOutput JSON per user."""
    __tablename__ = "questionnaire_outputs"

    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProblemsResult(Base):
    """Stores the problems.json output per user."""
    __tablename__ = "problems_results"

    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class IdeaResult(Base):
    """Stores the current idea and full chat history per user."""
    __tablename__ = "idea_results"

    user_id      = Column(String, primary_key=True)
    current_idea = Column(Text, nullable=True)
    chat_history = Column(JSON, default=list)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
