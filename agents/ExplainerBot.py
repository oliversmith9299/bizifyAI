"""
agents/ExplainerBot.py
=======================
Read-only explainer bot.

Purpose: help the founder understand what the pipeline generated WITHOUT
changing any data. Suitable for a "explain this to me" sidebar in the UI.

Allowed:
  - Read and summarise any agent output from the DB
  - Explain what a section means and why it matters
  - Compare two sections side by side
  - Answer "why / how / what does this mean" questions about the pipeline data
  - Explain startup concepts grounded in the user's actual data

Forbidden:
  - Making new recommendations outside the existing pipeline data
  - Changing any stored output
  - Running other agents
  - Advancing the business flow
  - Answering questions outside startup planning and the user's data

Intentionally simpler than generalBot.py — no routing, no triggers,
no intent classification. Just load the data, answer the question, stay in scope.
"""

import json
import logging
from typing import Optional

from db.connection import SessionLocal
from db import crud
from agents.config import client, GROQ_MODEL
from System_Messages.explainer_bot_prompt import EXPLAINER_BOT_SYSTEM_PROMPT

log = logging.getLogger(__name__)

_EXPLAINABLE_SECTIONS: dict[str, tuple[str, str]] = {
    "profile":          ("get_profile",          "Founder Profile"),
    "problems":         ("get_problems",          "Problem Discovery"),
    "idea":             ("get_idea",              "Startup Idea"),
    "customers":        ("get_customers",         "Customer Analysis"),
    "competition":      ("get_competition",       "Competition Analysis"),
    "market_potential": ("get_market_potential",  "Market Potential"),
    "idea_strategy":    ("get_idea_strategy",     "Idea Strategy"),
    "business_model":   ("get_business_model",    "Business Model"),
    "functions_list":   ("get_functions_list",    "Product Functions List"),
    "mvp_planning":     ("get_mvp_planning",      "MVP Plan"),
    "unit_economics":   ("get_unit_economics",    "Unit Economics"),
    "go_to_market":     ("get_go_to_market",      "Go-To-Market Plan"),
}


def _load_section(user_id: str, section: str, db) -> Optional[dict]:
    """Load data for a specific section. Returns None if not yet generated."""
    info = _EXPLAINABLE_SECTIONS.get(section)
    if not info:
        return None
    getter = getattr(crud, info[0], None)
    if not getter:
        return None
    row = getter(db, user_id)
    if not row:
        return None
    if hasattr(row, "data"):
        return row.data
    if hasattr(row, "current_idea"):
        return {"current_idea": row.current_idea}
    return None


def _load_all_available(user_id: str, db) -> dict:
    """Load compact summaries of all available sections."""
    available: dict = {}
    for key, (getter_name, label) in _EXPLAINABLE_SECTIONS.items():
        getter = getattr(crud, getter_name, None)
        if not getter:
            continue
        if getter(db, user_id):
            available[key] = label
    return available


def run_explainer_bot(
    user_id: str,
    message: str,
    history: list,
    section: Optional[str] = None,
) -> dict:
    db = SessionLocal()
    try:
        available = _load_all_available(user_id, db)

        context_parts = [
            "=== SECTIONS AVAILABLE FOR THIS FOUNDER ===",
            ", ".join(available.values()) if available else "No sections generated yet.",
        ]

        section_used: Optional[str] = None

        if section and section in available:
            data = _load_section(user_id, section, db)
            if data:
                label = _EXPLAINABLE_SECTIONS[section][1]
                context_parts += [
                    "",
                    f"=== FULL DATA FOR: {label} ===",
                    json.dumps(data, indent=2)[:4500],
                ]
                section_used = section
        elif not section:
            idea_row = crud.get_idea(db, user_id)
            if idea_row and idea_row.current_idea:
                context_parts += ["", "=== STARTUP IDEA ===", idea_row.current_idea[:500]]

        messages = [
            {"role": "system", "content": EXPLAINER_BOT_SYSTEM_PROMPT},
            {"role": "system", "content": "\n".join(context_parts)},
            *history[-16:],
            {"role": "user",   "content": message},
        ]

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=500,
        )

        return {
            "reply":          response.choices[0].message.content.strip(),
            "section_used":   section_used,
            "data_available": list(available.keys()),
        }

    finally:
        db.close()
