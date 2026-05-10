from datetime import datetime
from db.models import Agent, AgentRun, CompetitionResult, CustomersResult, IdeaIntakeResult, IdeaResult, IdeaStrategyResult, MarketPotentialResult, PipelineRun, ProfileResult, ProblemsResult, QuestionnaireOutput

AGENT_DEFINITIONS = {
    "OneProfileAnalysis": "discovery",
    "TwoProblemDiscovery": "discovery",
    "ThreeIdeaIntakeAgent": "discovery",
    "ThreePersonalizeIdeaChat": "ideation",
    "FourCustomersAgent":       "planning",
    "FiveCompetitionAgent":     "planning",
    "SixMaketPotential":        "planning",
    "SevenIdeaStrategy":        "strategy",
    "FiveCompetitionAgent": "planning",
    "SixMaketPotential": "planning",
    "SevenIdeaStrategy": "strategy",
    "EightBusinessModel": "business",
    "NineFunctionsList": "product",
    "TenMVPPlanning": "product",
    "ElevenUnitEconomicsAgent": "finance",
    "TwelveGoToMarket": "launch",
}

# ── Helper for safe commits (FIX 2 & 3) ───────────────────────────────────────
def safe_commit(db, row):
    """Safely commits to DB, refreshes data, and rolls back on failure."""
    try:
        db.commit()
        db.refresh(row)
        return row
    except Exception as e:
        db.rollback()
        raise e

# ── Pipeline Status ───────────────────────────────────────────────────────────
def upsert_pipeline_status(db, user_id: str, status: str, step: str = None, error: str = None):
    run = db.query(PipelineRun).filter_by(user_id=user_id).first()
    if not run:
        run = PipelineRun(user_id=user_id)
        db.add(run)
    
    run.status = status
    run.current_step = step
    run.error = error
    run.updated_at = datetime.utcnow() # FIX 1: Manual update trigger
    
    return safe_commit(db, run)

def get_pipeline_status(db, user_id: str):
    return db.query(PipelineRun).filter_by(user_id=user_id).first()

# ── Shared Agent Runs ─────────────────────────────────────────────────────────
def get_or_create_agent(db, name: str, phase: str = None):
    row = db.query(Agent).filter_by(name=name).first()
    if not row:
        row = Agent(name=name, phase=phase or AGENT_DEFINITIONS.get(name))
        db.add(row)
        db.flush()
    elif phase and row.phase != phase:
        row.phase = phase
    return row

def seed_agents(db):
    for name, phase in AGENT_DEFINITIONS.items():
        get_or_create_agent(db, name, phase)
    db.commit()

def save_agent_run(
    db,
    user_id: str,
    agent_name: str,
    input_data: dict = None,
    output_data: dict = None,
    status: str = "done",
    stage_id: str = None,
    confidence_score: float = None,
    execution_time_ms: int = None,
):
    agent = get_or_create_agent(db, agent_name, AGENT_DEFINITIONS.get(agent_name))
    input_payload = {"user_id": user_id, "payload": input_data or {}}
    output_payload = {"user_id": user_id, "payload": output_data or {}}

    row = AgentRun(
        stage_id=stage_id,
        agent_id=agent.id,
        status=status,
        input_data=input_payload,
        output_data=output_payload,
        confidence_score=confidence_score,
        execution_time_ms=execution_time_ms,
    )
    db.add(row)
    return safe_commit(db, row)

def get_latest_agent_run(db, user_id: str, agent_name: str):
    agent = db.query(Agent).filter_by(name=agent_name).first()
    if not agent:
        return None

    rows = (
        db.query(AgentRun)
        .filter_by(agent_id=agent.id)
        .order_by(AgentRun.created_at.desc())
        .limit(100)
        .all()
    )
    for row in rows:
        input_user = (row.input_data or {}).get("user_id")
        output_user = (row.output_data or {}).get("user_id")
        if input_user == user_id or output_user == user_id:
            return row
    return None

def get_latest_agent_output(db, user_id: str, agent_name: str):
    row = get_latest_agent_run(db, user_id, agent_name)
    if not row:
        return None
    return (row.output_data or {}).get("payload")

# ── Profile ───────────────────────────────────────────────────────────────────
def save_profile(db, user_id: str, data: dict):
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="OneProfileAnalysis",
        output_data=data,
        status="done",
    )
    row = db.query(ProfileResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProfileResult(user_id=user_id)
        db.add(row)
    
    row.data = data
    return safe_commit(db, row)

def get_profile(db, user_id: str):
    return db.query(ProfileResult).filter_by(user_id=user_id).first()

# ── Questionnaire ─────────────────────────────────────────────────────────────
def save_questionnaire_output(db, user_id: str, data: dict):
    row = db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()
    if not row:
        row = QuestionnaireOutput(user_id=user_id)
        db.add(row)
    
    row.data = data
    row.updated_at = datetime.utcnow() # FIX 1: Manual update trigger
    return safe_commit(db, row)

def get_questionnaire_output(db, user_id: str):
    return db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()

def get_questionnaire_output_json(db, user_id: str):
    row = get_questionnaire_output(db, user_id)
    return row.data if row else None

# ── Problems ──────────────────────────────────────────────────────────────────
def save_problems(db, user_id: str, data: dict):
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="TwoProblemDiscovery",
        output_data=data,
        status="done",
    )
    row = db.query(ProblemsResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProblemsResult(user_id=user_id)
        db.add(row)
    
    row.data = data
    return safe_commit(db, row)

def get_problems(db, user_id: str):
    return db.query(ProblemsResult).filter_by(user_id=user_id).first()

# ── Idea Intake ───────────────────────────────────────────────────────────────
def save_idea_intake(db, user_id: str, data: dict):
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="ThreeIdeaIntakeAgent",
        output_data=data,
        status="pending_clarification" if data.get("_status") == "pending_clarification" else "done",
    )
    row = db.query(IdeaIntakeResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaIntakeResult(user_id=user_id)
        db.add(row)

    row.data = data
    row.updated_at = datetime.utcnow()
    return safe_commit(db, row)

def get_idea_intake(db, user_id: str):
    return db.query(IdeaIntakeResult).filter_by(user_id=user_id).first()

def get_idea_intake_json(db, user_id: str):
    agent_output = get_latest_agent_output(db, user_id, "ThreeIdeaIntakeAgent")
    if agent_output:
        return agent_output
    row = get_idea_intake(db, user_id)
    return row.data if row else None

# ── Idea ──────────────────────────────────────────────────────────────────────
def save_idea(db, user_id: str, idea: str, history: list):
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="ThreePersonalizeIdeaChat",
        output_data={
            "current_idea": idea,
            "chat_history": history,
        },
        status="done",
    )
    row = db.query(IdeaResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaResult(user_id=user_id)
        db.add(row)
    
    row.current_idea = idea
    row.chat_history = history
    row.updated_at = datetime.utcnow() # FIX 1: Manual update trigger
    return safe_commit(db, row)

def get_idea(db, user_id: str):
    return db.query(IdeaResult).filter_by(user_id=user_id).first()

# ── Customers ─────────────────────────────────────────────────────────────────
def save_customers(db, user_id: str, data: dict, chat_history: list = None):
    row = db.query(CustomersResult).filter_by(user_id=user_id).first()
    if not row:
        row = CustomersResult(user_id=user_id)
        db.add(row)

    row.data         = data
    row.chat_history = chat_history if chat_history is not None else (row.chat_history or [])
    row.updated_at   = datetime.utcnow()
    return safe_commit(db, row)

def get_customers(db, user_id: str):
    return db.query(CustomersResult).filter_by(user_id=user_id).first()

def get_customers_json(db, user_id: str):
    row = get_customers(db, user_id)
    return row.data if row else None

# ── Competition ───────────────────────────────────────────────────────────────
def save_competition(db, user_id: str, data: dict, chat_history: list = None):
    row = db.query(CompetitionResult).filter_by(user_id=user_id).first()
    if not row:
        row = CompetitionResult(user_id=user_id)
        db.add(row)

    row.data         = data
    row.chat_history = chat_history if chat_history is not None else (row.chat_history or [])
    row.updated_at   = datetime.utcnow()
    return safe_commit(db, row)

def get_competition(db, user_id: str):
    return db.query(CompetitionResult).filter_by(user_id=user_id).first()

def get_competition_json(db, user_id: str):
    row = get_competition(db, user_id)
    return row.data if row else None

# ── Market Potential ──────────────────────────────────────────────────────────
def save_market_potential(db, user_id: str, data: dict, chat_history: list = None):
    row = db.query(MarketPotentialResult).filter_by(user_id=user_id).first()
    if not row:
        row = MarketPotentialResult(user_id=user_id)
        db.add(row)

    row.data         = data
    row.chat_history = chat_history if chat_history is not None else (row.chat_history or [])
    row.updated_at   = datetime.utcnow()
    return safe_commit(db, row)

def get_market_potential(db, user_id: str):
    return db.query(MarketPotentialResult).filter_by(user_id=user_id).first()

def get_market_potential_json(db, user_id: str):
    row = get_market_potential(db, user_id)
    return row.data if row else None

# ── Idea Strategy ─────────────────────────────────────────────────────────────
def save_idea_strategy(db, user_id: str, data: dict, chat_history: list = None):
    row = db.query(IdeaStrategyResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaStrategyResult(user_id=user_id)
        db.add(row)

    row.data         = data
    row.chat_history = chat_history if chat_history is not None else (row.chat_history or [])
    row.updated_at   = datetime.utcnow()
    return safe_commit(db, row)

def get_idea_strategy(db, user_id: str):
    return db.query(IdeaStrategyResult).filter_by(user_id=user_id).first()

def get_idea_strategy_json(db, user_id: str):
    row = get_idea_strategy(db, user_id)
    return row.data if row else None
