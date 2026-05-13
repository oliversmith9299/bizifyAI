"""
agents/generalBot.py
=====================
General-purpose startup planning chatbot that routes between all 12 pipeline agents.

Architecture — pure Python, no framework:

  User message
       ↓
  Step 1: Intent classification (one fast LLM call → structured JSON)
       ↓
       ├─ chat_about_data     → load section data from DB → answer from it
       ├─ run_section         → trigger the right agent function
       ├─ refine_section      → load section chat context → handle conversation
       ├─ pipeline_status     → summarise completed vs pending steps
       ├─ start_pipeline      → explain how to begin, return pipeline_trigger flag
       ├─ general_startup_chat → answer from available data
       └─ out_of_scope        → decline and redirect

Why no LangGraph / LangChain / AutoGen:
  The routing logic is a single conditional branch — not a graph, not parallel,
  not stateful multi-agent. Adding a framework would add complexity without benefit
  and break consistency with the other 12 agents that all use raw Python + Groq.
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agents.utils import parse_llm_json
from db.connection import SessionLocal
from db import crud
from System_Messages.general_bot_prompt import (
    GENERAL_BOT_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
)

load_dotenv()

log = logging.getLogger(__name__)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# Map section names (used in intents) to crud getter functions
_SECTION_LOADERS: dict[str, str] = {
    "profile":          "get_profile",
    "problems":         "get_problems",
    "idea_intake":      "get_idea_intake",
    "idea":             "get_idea",
    "customers":        "get_customers",
    "competition":      "get_competition",
    "market_potential": "get_market_potential",
    "idea_strategy":    "get_idea_strategy",
    "business_model":   "get_business_model",
    "functions_list":   "get_functions_list",
    "mvp_planning":     "get_mvp_planning",
    "unit_economics":   "get_unit_economics",
    "go_to_market":     "get_go_to_market",
}

# Map section names to their human-readable labels
_SECTION_LABELS: dict[str, str] = {
    "profile":          "Founder Profile Analysis",
    "problems":         "Problem Discovery",
    "idea_intake":      "Idea Intake (Returning User)",
    "idea":             "Idea Generation & Chat",
    "customers":        "Customer Analysis",
    "competition":      "Competition Analysis",
    "market_potential": "Market Potential",
    "idea_strategy":    "Idea Strategy",
    "business_model":   "Business Model",
    "functions_list":   "Product Functions List",
    "mvp_planning":     "MVP Planning",
    "unit_economics":   "Unit Economics",
    "go_to_market":     "Go-To-Market Plan",
}

# Map section names to the API endpoint route to trigger them
_SECTION_ROUTES: dict[str, str] = {
    "customers":        "/pipeline/customers/{user_id}",
    "competition":      "/pipeline/competition/{user_id}",
    "market_potential": "/pipeline/market-potential/{user_id}",
    "idea_strategy":    "/pipeline/idea-strategy/{user_id}",
    "business_model":   "/pipeline/business-model/{user_id}",
    "functions_list":   "/pipeline/functions-list/{user_id}",
    "mvp_planning":     "/pipeline/mvp-planning/{user_id}",
    "unit_economics":   "/pipeline/unit-economics/{user_id}",
    "go_to_market":     "/pipeline/go-to-market/{user_id}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Context snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline_snapshot(user_id: str, db) -> dict:
    """
    Load a compact snapshot of everything the user has done so far.
    Returns a dict with status flags and key insight summaries per section.
    This is the bot's "world knowledge" about this specific user.
    """
    snapshot: dict = {"user_id": user_id, "sections": {}}

    # Check each section and extract its most important insight for the summary
    getters = {
        "idea": ("get_idea", lambda r: r.current_idea[:300] if r and r.current_idea else None),
        "customers": ("get_customers", lambda r: r.data.get("primary_segment", {}).get("reason", "") if r else None),
        "competition": ("get_competition", lambda r: r.data.get("summary", "") if r else None),
        "market_potential": ("get_market_potential", lambda r: r.data.get("summary", "") if r else None),
        "idea_strategy": ("get_idea_strategy", lambda r: r.data.get("core_promise", "") if r else None),
        "business_model": ("get_business_model", lambda r: r.data.get("business_model_type", "") if r else None),
        "functions_list": ("get_functions_list", lambda r: r.data.get("product_type", "") if r else None),
        "mvp_planning": ("get_mvp_planning", lambda r: r.data.get("mvp_goal", "") if r else None),
        "unit_economics": ("get_unit_economics", lambda r: r.data.get("overall_viability", {}).get("viability_reasoning", "") if r else None),
        "go_to_market": ("get_go_to_market", lambda r: r.data.get("summary", "") if r else None),
        "profile": ("get_profile", lambda r: r.data.get("founder_profile", {}).get("experience_level", "") if r else None),
        "problems": ("get_problems", lambda r: f"{len(r.data.get('problems', []))} problems found" if r else None),
    }

    for section, (getter_name, extractor) in getters.items():
        getter = getattr(crud, getter_name, None)
        if getter:
            row = getter(db, user_id)
            insight = extractor(row) if row else None
            snapshot["sections"][section] = {
                "done":    row is not None,
                "label":   _SECTION_LABELS.get(section, section),
                "insight": (insight or "")[:200] if insight else "",
            }

    # Count completed sections
    done_count = sum(1 for s in snapshot["sections"].values() if s["done"])
    snapshot["completed_count"] = done_count
    snapshot["total_sections"]  = len(snapshot["sections"])

    return snapshot


def _snapshot_to_context_string(snapshot: dict) -> str:
    """Convert the pipeline snapshot into a readable context string for the LLM."""
    lines = [
        f"=== PIPELINE STATUS ({snapshot['completed_count']}/{snapshot['total_sections']} sections complete) ===",
    ]

    done   = [(k, v) for k, v in snapshot["sections"].items() if v["done"]]
    pending = [(k, v) for k, v in snapshot["sections"].items() if not v["done"]]

    if done:
        lines.append("\nCOMPLETED SECTIONS:")
        for _, v in done:
            insight = f" — {v['insight']}" if v["insight"] else ""
            lines.append(f"  ✅ {v['label']}{insight[:120]}")

    if pending:
        lines.append("\nPENDING SECTIONS (not yet generated):")
        for _, v in pending:
            lines.append(f"  ⬜ {v['label']}")

    return "\n".join(lines)


def _load_section_data(user_id: str, section: str, db) -> Optional[dict]:
    """Load the full data for a specific section from DB."""
    getter_info = _SECTION_LOADERS.get(section)
    if not getter_info:
        return None
    getter = getattr(crud, getter_info, None)
    if not getter:
        return None
    row = getter(db, user_id)
    if not row:
        return None
    if hasattr(row, "data"):
        return row.data
    if hasattr(row, "current_idea"):
        return {"current_idea": row.current_idea, "chat_history_length": len(row.chat_history or [])}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_intent(
    message: str,
    history: list,
    snapshot: dict,
) -> dict:
    """
    One fast LLM call to classify what the user wants.
    Returns dict with intent, section, confidence, reasoning.
    """
    # Build a minimal snapshot summary for the classifier
    done_sections   = [k for k, v in snapshot["sections"].items() if v["done"]]
    pending_sections = [k for k, v in snapshot["sections"].items() if not v["done"]]

    context = (
        f"Completed sections: {done_sections}\n"
        f"Pending sections: {pending_sections}"
    )

    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
        {"role": "system", "content": context},
        *history[-6:],   # only recent turns — classifier is fast
        {"role": "user",  "content": message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,   # very deterministic for classification
        max_tokens=200,
    )

    try:
        return parse_llm_json(response.choices[0].message.content)
    except Exception:
        # Fallback: if classification fails, treat as general chat
        return {"intent": "general_startup_chat", "section": None, "confidence": 0.5, "reasoning": "parse error"}


# ─────────────────────────────────────────────────────────────────────────────
# Response generators per intent
# ─────────────────────────────────────────────────────────────────────────────

def _respond_from_data(
    message: str,
    history: list,
    section: Optional[str],
    snapshot: dict,
    section_data: Optional[dict],
) -> str:
    """Answer a question grounded in the user's actual pipeline data."""
    context_parts = [_snapshot_to_context_string(snapshot)]

    if section_data:
        context_parts.append(
            f"\n=== DETAILED DATA FOR [{_SECTION_LABELS.get(section, section)}] ===\n"
            + json.dumps(section_data, indent=2)[:4000]
        )

    messages = [
        {"role": "system",  "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system",  "content": "\n\n".join(context_parts)},
        *history[-20:],
        {"role": "user",    "content": message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()


def _respond_pipeline_status(snapshot: dict, history: list, message: str) -> str:
    """Summarise pipeline progress and recommend the next step."""
    context = _snapshot_to_context_string(snapshot)

    # Determine next recommended step
    pending = [k for k, v in snapshot["sections"].items() if not v["done"]]
    next_step = pending[0] if pending else None
    next_label = _SECTION_LABELS.get(next_step, next_step) if next_step else None

    status_note = (
        f"\nNEXT RECOMMENDED STEP: Generate '{next_label}' "
        f"(tell the user to ask: 'generate my {next_step}')"
        if next_label
        else "\nAll sections complete. Encourage the founder to review and refine."
    )

    messages = [
        {"role": "system", "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system", "content": context + status_note},
        *history[-10:],
        {"role": "user",   "content": message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def _respond_run_section(section: Optional[str], snapshot: dict) -> tuple[str, Optional[str]]:
    """
    Tell the user the bot will run the section, and return the route to trigger.
    Returns (reply_text, route_to_trigger).
    """
    if not section:
        return (
            "I can run any of these sections for you: "
            + ", ".join(_SECTION_LABELS[s] for s in _SECTION_ROUTES)
            + ". Which one would you like?",
            None,
        )

    label = _SECTION_LABELS.get(section, section)
    is_done = snapshot["sections"].get(section, {}).get("done", False)

    if section not in _SECTION_ROUTES:
        return (
            f"The '{label}' section is generated automatically during the main pipeline. "
            "Make sure the pipeline has run first by asking your backend to call POST /pipeline/run.",
            None,
        )

    route = _SECTION_ROUTES[section]
    action = "Regenerating" if is_done else "Generating"
    return (
        f"{action} your **{label}**... This will take a moment. "
        f"{'The previous result will be replaced.' if is_done else ''}",
        route,
    )


def _respond_refine_section(section: Optional[str], snapshot: dict) -> tuple[str, Optional[str]]:
    """Tell the user how to refine a section via its section-specific chat."""
    if not section:
        return (
            "Which section would you like to refine? I can help with: "
            + ", ".join(_SECTION_LABELS[s] for s in _SECTION_ROUTES)
            + ".",
            None,
        )

    label    = _SECTION_LABELS.get(section, section)
    is_done  = snapshot["sections"].get(section, {}).get("done", False)

    if not is_done:
        return (
            f"You haven't generated the **{label}** yet. "
            f"Ask me to 'generate my {section}' first, then we can refine it.",
            None,
        )

    chat_route = _SECTION_ROUTES.get(section, "").replace("}", "/chat}")
    return (
        f"To refine your **{label}**, use the section chat endpoint at "
        f"`{chat_route}`. Send your refinement request there and I'll apply it. "
        f"What change do you want to make?",
        chat_route if chat_route else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_general_bot(
    user_id: str,
    message: str,
    history: list,
) -> dict:
    """
    Process a user message and return a structured response.

    Parameters
    ----------
    user_id : str   — the founder's user ID
    message : str   — the user's current message
    history : list  — prior conversation turns [{"role": ..., "content": ...}]

    Returns
    -------
    {
      "reply":            str   — the assistant's response
      "intent":           str   — what was classified
      "section":          str | None
      "action":           str   — "answered" | "route_to_section" | "declined" | "status"
      "route_to_trigger": str | None  — if action is route_to_section, the API path
      "trigger_needed":   bool  — True if the backend should call an agent endpoint
    }
    """
    db = SessionLocal()
    try:
        # ── Load pipeline state ───────────────────────────────────────────────
        snapshot = _load_pipeline_snapshot(user_id, db)

        # ── Classify intent ───────────────────────────────────────────────────
        classification = _classify_intent(message, history, snapshot)
        intent  = classification.get("intent", "general_startup_chat")
        section = classification.get("section")

        log.info(f"[GeneralBot] user={user_id} intent={intent} section={section}")

        # ── Route by intent ───────────────────────────────────────────────────

        if intent == "out_of_scope":
            return {
                "reply":            OUT_OF_SCOPE_RESPONSE,
                "intent":           intent,
                "section":          section,
                "action":           "declined",
                "route_to_trigger": None,
                "trigger_needed":   False,
            }

        if intent == "pipeline_status":
            reply = _respond_pipeline_status(snapshot, history, message)
            return {
                "reply":            reply,
                "intent":           intent,
                "section":          section,
                "action":           "status",
                "route_to_trigger": None,
                "trigger_needed":   False,
            }

        if intent == "start_pipeline":
            # Pipeline start is handled by the backend (/pipeline/run)
            is_fresh = not any(v["done"] for v in snapshot["sections"].values())
            reply = (
                "To start a new pipeline, your backend needs to call "
                "POST /pipeline/run with your questionnaire data. "
                + (
                    "You don't have any sections generated yet — this is the right first step."
                    if is_fresh
                    else f"You already have {snapshot['completed_count']} sections. "
                         "Starting over will replace them."
                )
            )
            return {
                "reply":            reply,
                "intent":           intent,
                "section":          None,
                "action":           "start_pipeline_instruction",
                "route_to_trigger": "/pipeline/run",
                "trigger_needed":   False,  # backend handles this, not the bot
            }

        if intent == "run_section":
            reply, route = _respond_run_section(section, snapshot)
            return {
                "reply":            reply,
                "intent":           intent,
                "section":          section,
                "action":           "route_to_section",
                "route_to_trigger": route.replace("{user_id}", user_id) if route else None,
                "trigger_needed":   route is not None,
            }

        if intent == "refine_section":
            reply, route = _respond_refine_section(section, snapshot)
            return {
                "reply":            reply,
                "intent":           intent,
                "section":          section,
                "action":           "refine_section_instruction",
                "route_to_trigger": route.replace("{user_id}", user_id) if route else None,
                "trigger_needed":   False,  # user must call the section chat endpoint directly
            }

        # Default: chat_about_data or general_startup_chat
        section_data = _load_section_data(user_id, section, db) if section else None
        reply = _respond_from_data(message, history, section, snapshot, section_data)

        return {
            "reply":            reply,
            "intent":           intent,
            "section":          section,
            "action":           "answered",
            "route_to_trigger": None,
            "trigger_needed":   False,
        }

    except Exception as e:
        log.error(f"[GeneralBot] error for user {user_id}: {e}", exc_info=True)
        raise

    finally:
        db.close()
