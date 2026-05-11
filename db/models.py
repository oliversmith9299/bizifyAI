import uuid
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, JSON, Text, DateTime
from datetime import datetime

# Shared base from connection
from db.connection import Base

# ─────────────────────────────────────────────────────────────────────────────
# Real platform tables (matching the DB schema PDF — Section 2, 3, 4, 9)
# These tables are owned by the backend; the AI service reads/writes them.
# Do NOT rename columns — they must stay in sync with the backend migrations.
# ─────────────────────────────────────────────────────────────────────────────

class Business(Base):
    """Section 3 — Businesses & Industries."""
    __tablename__ = "businesses"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    idea_id       = Column(String, nullable=True)   # FK → ideas.id
    owner_id      = Column(String, nullable=False)  # FK → users.id (managed by backend)
    industry_id   = Column(String, nullable=True)   # FK → industries.id
    stage         = Column(String, nullable=True)
    context_json  = Column(JSON,   nullable=True)
    is_archived   = Column(Boolean, default=False)
    archived_at   = Column(DateTime, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Idea(Base):
    """Section 2 — Ideas & Projects. All columns match the platform DB schema exactly."""
    __tablename__ = "ideas"
    id                = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id          = Column(String,  nullable=False)   # FK → users.id
    business_id       = Column(String,  nullable=True)    # FK → businesses.id
    title             = Column(String,  nullable=True)
    description       = Column(Text,    nullable=True)
    status            = Column(String,  default="draft")  # draft | active | archived
    ai_score          = Column(Float,   nullable=True)
    budget            = Column(Float,   nullable=True)
    skills            = Column(JSON,    nullable=True)
    feasibility       = Column(Float,   nullable=True)
    is_score_outdated = Column(Boolean, default=False)    # schema field — tracks stale AI scores
    is_archived       = Column(Boolean, default=False)
    archived_at       = Column(DateTime, nullable=True)
    converted_at      = Column(DateTime, nullable=True)   # schema field — when idea → business
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BusinessRoadmap(Base):
    """Section 4 — AI & Roadmaps."""
    __tablename__          = "business_roadmaps"
    id                     = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    business_id            = Column(String, nullable=False)  # FK → businesses.id
    completion_percentage  = Column(Float, default=0.0)
    created_at             = Column(DateTime, default=datetime.utcnow)


class RoadmapStage(Base):
    """Section 4 — one row per pipeline step per business."""
    __tablename__ = "roadmap_stages"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roadmap_id    = Column(String, ForeignKey("business_roadmaps.id"), nullable=False)
    order_index   = Column(Integer, nullable=True)
    stage_type    = Column(String, nullable=True)   # matches Step enum values
    status        = Column(String, default="pending")  # pending | running | done | error
    output_json   = Column(JSON, nullable=True)
    completed_at  = Column(DateTime, nullable=True)


class ChatSession(Base):
    """Section 9 — Chat, Files & Notifications."""
    __tablename__              = "chat_sessions"
    id                         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id                    = Column(String, nullable=False)   # FK → users.id
    business_id                = Column(String, nullable=True)    # FK → businesses.id
    idea_id                    = Column(String, nullable=True)    # FK → ideas.id
    session_type               = Column(String, nullable=True)    # idea_chat | section_chat
    conversation_summary_json  = Column(JSON, nullable=True)
    created_at                 = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    """Section 9 — individual messages inside a chat session."""
    __tablename__ = "chat_messages"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role       = Column(String, nullable=False)   # user | assistant | system
    content    = Column(Text,   nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    chat_history = Column(JSON, default=lambda: [])
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class IdeaIntakeResult(Base):
    __tablename__ = "idea_intake_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CustomersResult(Base):
    __tablename__ = "customers_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CompetitionResult(Base):
    __tablename__ = "competition_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MarketPotentialResult(Base):
    __tablename__ = "market_potential_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class IdeaStrategyResult(Base):
    __tablename__ = "idea_strategy_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BusinessModelResult(Base):
    __tablename__ = "business_model_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FunctionsListResult(Base):
    __tablename__ = "functions_list_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MVPPlanningResult(Base):
    __tablename__ = "mvp_planning_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UnitEconomicsResult(Base):
    __tablename__ = "unit_economics_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GoToMarketResult(Base):
    __tablename__ = "go_to_market_results"
    user_id      = Column(String, primary_key=True)
    data         = Column(JSON, nullable=False)
    chat_history = Column(JSON, default=lambda: [])
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Agent(Base):
    __tablename__ = "agents"
    id    = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name  = Column(String, nullable=False, unique=True)
    phase = Column(String, nullable=True)

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id                = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stage_id          = Column(String, nullable=True)
    agent_id          = Column(String, ForeignKey("agents.id"), nullable=False)
    status            = Column(String, default="done")
    input_data        = Column(JSON, nullable=True)
    output_data       = Column(JSON, nullable=True)
    confidence_score  = Column(Float, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
