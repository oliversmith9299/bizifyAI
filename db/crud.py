from datetime import datetime
from db.models import IdeaResult, PipelineRun, ProfileResult, ProblemsResult, QuestionnaireOutput


# ── Pipeline Status ───────────────────────────────────────────────────────────

def upsert_pipeline_status(
    db,
    user_id: str,
    status: str,
    step: str = None,
    error: str = None
):
    """Create or update the pipeline run record for a user."""
    run = db.query(PipelineRun).filter_by(user_id=user_id).first()
    if not run:
        run = PipelineRun(id=user_id, user_id=user_id)
        db.add(run)
    run.status = status
    run.current_step = step
    run.error = error
    run.updated_at = datetime.utcnow()
    db.commit()
    return run


def get_pipeline_status(db, user_id: str):
    return db.query(PipelineRun).filter_by(user_id=user_id).first()


# ── Profile ───────────────────────────────────────────────────────────────────

def save_profile(db, user_id: str, data: dict):
    row = db.query(ProfileResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProfileResult(user_id=user_id)
        db.add(row)
    row.data = data
    db.commit()
    return row


def get_profile(db, user_id: str):
    return db.query(ProfileResult).filter_by(user_id=user_id).first()


def save_questionnaire_output(db, user_id: str, data: dict):
    """Create or update the original questionnaireOutput JSON for a user."""
    row = db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()
    if not row:
        row = QuestionnaireOutput(user_id=user_id)
        db.add(row)
    row.data = data
    row.updated_at = datetime.utcnow()
    db.commit()
    return row


def get_questionnaire_output(db, user_id: str):
    """Return the questionnaire output DB row for a user."""
    return db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()


def get_questionnaire_output_json(db, user_id: str):
    """Return only the questionnaireOutput JSON dict for a user."""
    row = get_questionnaire_output(db, user_id)
    return row.data if row else None


# ── Problems ──────────────────────────────────────────────────────────────────

def save_problems(db, user_id: str, data: dict):
    row = db.query(ProblemsResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProblemsResult(user_id=user_id)
        db.add(row)
    row.data = data
    db.commit()
    return row


def get_problems(db, user_id: str):
    return db.query(ProblemsResult).filter_by(user_id=user_id).first()


# ── Idea ──────────────────────────────────────────────────────────────────────

def save_idea(db, user_id: str, idea: str, history: list):
    row = db.query(IdeaResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaResult(user_id=user_id)
        db.add(row)
    row.current_idea = idea
    row.chat_history = history
    row.updated_at = datetime.utcnow()
    db.commit()
    return row


def get_idea(db, user_id: str):
    return db.query(IdeaResult).filter_by(user_id=user_id).first()
