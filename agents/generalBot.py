"""
agents/generalBot.py
=====================
Conversational startup planning assistant.

The bot handles everything itself:
  - Runs agents directly when the user asks (no "call this route" responses)
  - Automatically detects which prerequisites are missing and asks the user
    before running a long chain
  - Saves every agent result to the DB so the user finds their work if they
    close the chat
  - Presents all results as natural conversation, never exposing API internals

Confirmation flow
-----------------
When prerequisites are missing the bot asks the user to confirm before running
multiple sections.  The pending list is embedded in the reply as an invisible
HTML comment  <!-- PENDING:section1,section2 -->  which markdown renderers hide
from the user but which is preserved in the conversation history that the
frontend echoes back on the next turn.  On the next message the bot checks for
that marker, detects a confirm / decline intent, and acts accordingly.
"""

import json
import logging
import re
from typing import Optional

from agents.config import client, GROQ_MODEL
from agents.utils import parse_llm_json
from db.connection import SessionLocal
from db import crud
from System_Messages.general_bot_prompt import (
    GENERAL_BOT_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Section registry
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_LABELS: dict[str, str] = {
    "profile":          "Founder Profile",
    "problems":         "Problem Discovery",
    "idea":             "Idea Generation",
    "idea_intake":      "Idea Definition",
    "customers":        "Customer Analysis",
    "competition":      "Competition Analysis",
    "market_potential": "Market Potential",
    "idea_strategy":    "Idea Strategy",
    "business_model":   "Business Model",
    "functions_list":   "Product Functions",
    "mvp_planning":     "MVP Planning",
    "unit_economics":   "Unit Economics",
    "go_to_market":     "Go-To-Market Plan",
}

# Analysis sections the bot can run directly (excludes pipeline-level sections
# that are handled by the onboarding flow: profile, problems, idea)
_RUNNABLE_SECTIONS = {
    "idea_intake", "customers", "competition", "market_potential",
    "idea_strategy", "business_model", "functions_list",
    "mvp_planning", "unit_economics", "go_to_market",
}

# Sections that should exist before running a given section.
# The bot will ask the user to confirm before auto-running the full chain.
_SHOULD_RUN_BEFORE: dict[str, list[str]] = {
    "idea_intake":      [],
    "customers":        [],
    "competition":      [],
    "market_potential": [],
    "idea_strategy":    ["customers", "competition", "market_potential"],
    "business_model":   ["customers", "idea_strategy"],
    "functions_list":   ["business_model"],
    "mvp_planning":     ["functions_list"],
    "unit_economics":   ["business_model"],
    "go_to_market":     ["customers", "unit_economics"],
}

# Canonical run order so chains always execute in the right sequence
_SECTION_ORDER = [
    "idea_intake", "customers", "competition", "market_potential",
    "idea_strategy", "business_model", "functions_list",
    "mvp_planning", "unit_economics", "go_to_market",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pending-confirmation helpers
# ─────────────────────────────────────────────────────────────────────────────

_PENDING_RE = re.compile(r"<!--PENDING:([^-]+?)-->")


def _embed_pending(reply: str, sections: list[str]) -> str:
    """Append an invisible HTML comment carrying the pending section list."""
    return reply + f"\n<!--PENDING:{','.join(sections)}-->"


def _extract_pending(history: list) -> list[str]:
    """
    Scan the last assistant message in history for a PENDING marker.
    Returns the section list, or [] if not found.
    """
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            m = _PENDING_RE.search(msg.get("content", ""))
            if m:
                return [s.strip() for s in m.group(1).split(",") if s.strip()]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline_snapshot(user_id: str, db) -> dict:
    """
    Compact view of everything the user has done so far.
    Used by the LLM as "world knowledge" about this founder.
    """
    getters = {
        "idea":             ("get_idea",             lambda r: r.current_idea[:300] if r and r.current_idea else None),
        "idea_intake":      ("get_idea_intake",       lambda r: r.idea_summary[:200] if r and r.idea_summary else None),
        "customers":        ("get_customers",         lambda r: r.data.get("summary", "")[:200] if r else None),
        "competition":      ("get_competition",       lambda r: r.data.get("summary", "")[:200] if r else None),
        "market_potential": ("get_market_potential",  lambda r: r.data.get("summary", "")[:200] if r else None),
        "idea_strategy":    ("get_idea_strategy",     lambda r: r.data.get("core_promise", "")[:200] if r else None),
        "business_model":   ("get_business_model",    lambda r: r.data.get("business_model_type", "") if r else None),
        "functions_list":   ("get_functions_list",    lambda r: r.data.get("product_type", "") if r else None),
        "mvp_planning":     ("get_mvp_planning",      lambda r: r.data.get("mvp_goal", "")[:200] if r else None),
        "unit_economics":   ("get_unit_economics",    lambda r: r.data.get("overall_viability", {}).get("verdict", "") if r else None),
        "go_to_market":     ("get_go_to_market",      lambda r: r.data.get("summary", "")[:200] if r else None),
        "profile":          ("get_profile",           lambda r: r.data.get("founder_profile", {}).get("experience_level", "") if r else None),
        "problems":         ("get_problems",          lambda r: f"{len(r.data.get('problems', []))} problems found" if r else None),
    }

    sections: dict = {}
    for section, (getter_name, extractor) in getters.items():
        getter = getattr(crud, getter_name, None)
        row    = getter(db, user_id) if getter else None
        insight = extractor(row) if row else None
        sections[section] = {
            "done":    row is not None,
            "label":   _SECTION_LABELS.get(section, section),
            "insight": (insight or "")[:200],
        }

    done_count = sum(1 for s in sections.values() if s["done"])
    return {
        "user_id":         user_id,
        "sections":        sections,
        "completed_count": done_count,
        "total_sections":  len(sections),
    }


def _snapshot_to_context(snapshot: dict) -> str:
    lines = [
        f"=== PIPELINE STATUS ({snapshot['completed_count']}/{snapshot['total_sections']} sections complete) ===",
    ]
    done    = [(k, v) for k, v in snapshot["sections"].items() if v["done"]]
    pending = [(k, v) for k, v in snapshot["sections"].items() if not v["done"]]

    if done:
        lines.append("\nCOMPLETED:")
        for _, v in done:
            suffix = f" — {v['insight'][:120]}" if v["insight"] else ""
            lines.append(f"  ✅ {v['label']}{suffix}")
    if pending:
        lines.append("\nNOT YET GENERATED:")
        for _, v in pending:
            lines.append(f"  ⬜ {v['label']}")
    return "\n".join(lines)


def _load_section_data(user_id: str, section: Optional[str], db) -> Optional[dict]:
    """Load full data for a specific section."""
    loaders = {
        "profile":          "get_profile",
        "problems":         "get_problems",
        "idea":             "get_idea",
        "idea_intake":      "get_idea_intake",
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
    getter_name = loaders.get(section or "")
    if not getter_name:
        return None
    getter = getattr(crud, getter_name, None)
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


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_intent(message: str, history: list, snapshot: dict) -> dict:
    done_sections    = [k for k, v in snapshot["sections"].items() if v["done"]]
    pending_sections = [k for k, v in snapshot["sections"].items() if not v["done"]]

    context = (
        f"Completed sections: {done_sections}\n"
        f"Pending sections: {pending_sections}\n"
        f"Last assistant message (for confirm/decline detection): "
        + (history[-1]["content"][:300] if history and history[-1].get("role") == "assistant" else "none")
    )

    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
        {"role": "system", "content": context},
        *history[-6:],
        {"role": "user",   "content": message},
    ]

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=200,
        )
        return parse_llm_json(resp.choices[0].message.content)
    except Exception:
        return {"intent": "general_startup_chat", "section": None, "confidence": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Agent runners  (run agent + save to DB)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_section(section: str, user_id: str, db) -> dict:
    """
    Import the right agent, call it with all available context from DB,
    save the result to DB, and return the result dict.
    """
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    idea_text    = idea_row.current_idea if idea_row else ""
    problems     = problems_row.data if problems_row else {}

    if section == "customers":
        from agents.FourCustomersAgent import run_customers_analysis
        profile_row = crud.get_profile(db, user_id)
        result = run_customers_analysis(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            profile=profile_row.data if profile_row else None,
        )
        crud.save_customers(db, user_id, result)

    elif section == "competition":
        from agents.FiveCompetitionAgent import run_competition_analysis
        customers_row = crud.get_customers(db, user_id)
        result = run_competition_analysis(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data if customers_row else None,
        )
        crud.save_competition(db, user_id, result)

    elif section == "market_potential":
        from agents.SixMaketPotential import run_market_potential
        customers_row   = crud.get_customers(db, user_id)
        competition_row = crud.get_competition(db, user_id)
        result = run_market_potential(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data   if customers_row   else None,
            competition=competition_row.data if competition_row else None,
        )
        crud.save_market_potential(db, user_id, result)

    elif section == "idea_strategy":
        from agents.SevenIdeaStrategy import run_idea_strategy
        customers_row   = crud.get_customers(db, user_id)
        competition_row = crud.get_competition(db, user_id)
        mp_row          = crud.get_market_potential(db, user_id)
        result = run_idea_strategy(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data     if customers_row   else None,
            competition=competition_row.data if competition_row else None,
            market_potential=mp_row.data     if mp_row          else None,
        )
        crud.save_idea_strategy(db, user_id, result)

    elif section == "business_model":
        from agents.EightBusinessModel import run_business_model
        customers_row   = crud.get_customers(db, user_id)
        competition_row = crud.get_competition(db, user_id)
        mp_row          = crud.get_market_potential(db, user_id)
        strategy_row    = crud.get_idea_strategy(db, user_id)
        result = run_business_model(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data     if customers_row   else None,
            competition=competition_row.data if competition_row else None,
            market_potential=mp_row.data     if mp_row          else None,
            strategy=strategy_row.data       if strategy_row    else None,
        )
        crud.save_business_model(db, user_id, result)

    elif section == "functions_list":
        from agents.NineFunctionsList import run_functions_list
        customers_row   = crud.get_customers(db, user_id)
        competition_row = crud.get_competition(db, user_id)
        mp_row          = crud.get_market_potential(db, user_id)
        strategy_row    = crud.get_idea_strategy(db, user_id)
        bm_row          = crud.get_business_model(db, user_id)
        result = run_functions_list(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data     if customers_row   else None,
            competition=competition_row.data if competition_row else None,
            market_potential=mp_row.data     if mp_row          else None,
            strategy=strategy_row.data       if strategy_row    else None,
            business_model=bm_row.data       if bm_row          else None,
        )
        crud.save_functions_list(db, user_id, result)

    elif section == "mvp_planning":
        from agents.TenMVPPlanning import run_mvp_planning
        customers_row = crud.get_customers(db, user_id)
        mp_row        = crud.get_market_potential(db, user_id)
        strategy_row  = crud.get_idea_strategy(db, user_id)
        bm_row        = crud.get_business_model(db, user_id)
        fl_row        = crud.get_functions_list(db, user_id)
        result = run_mvp_planning(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data if customers_row else None,
            market_potential=mp_row.data if mp_row        else None,
            strategy=strategy_row.data   if strategy_row  else None,
            business_model=bm_row.data   if bm_row        else None,
            functions_list=fl_row.data   if fl_row        else None,
        )
        crud.save_mvp_planning(db, user_id, result)

    elif section == "unit_economics":
        from agents.ElevenUnitEconomicsAgent import run_unit_economics
        customers_row = crud.get_customers(db, user_id)
        mp_row        = crud.get_market_potential(db, user_id)
        strategy_row  = crud.get_idea_strategy(db, user_id)
        bm_row        = crud.get_business_model(db, user_id)
        mvp_row       = crud.get_mvp_planning(db, user_id)
        result = run_unit_economics(
            user_id=user_id,
            idea=idea_text,
            customers=customers_row.data if customers_row else None,
            market_potential=mp_row.data if mp_row        else None,
            strategy=strategy_row.data   if strategy_row  else None,
            business_model=bm_row.data   if bm_row        else None,
            mvp_planning=mvp_row.data    if mvp_row       else None,
        )
        crud.save_unit_economics(db, user_id, result)

    elif section == "go_to_market":
        from agents.TwelveGoToMarket import run_go_to_market
        customers_row   = crud.get_customers(db, user_id)
        competition_row = crud.get_competition(db, user_id)
        mp_row          = crud.get_market_potential(db, user_id)
        strategy_row    = crud.get_idea_strategy(db, user_id)
        bm_row          = crud.get_business_model(db, user_id)
        mvp_row         = crud.get_mvp_planning(db, user_id)
        ue_row          = crud.get_unit_economics(db, user_id)
        result = run_go_to_market(
            user_id=user_id,
            idea=idea_text,
            problems=problems,
            customers=customers_row.data     if customers_row   else None,
            competition=competition_row.data if competition_row else None,
            market_potential=mp_row.data     if mp_row          else None,
            strategy=strategy_row.data       if strategy_row    else None,
            business_model=bm_row.data       if bm_row          else None,
            mvp_planning=mvp_row.data        if mvp_row         else None,
            unit_economics=ue_row.data       if ue_row          else None,
        )
        crud.save_go_to_market(db, user_id, result)

    else:
        raise ValueError(f"Unknown runnable section: {section!r}")

    return result


def _run_sections_in_order(sections: list[str], user_id: str, db) -> dict[str, dict]:
    """Run multiple sections sequentially and return {section: result}."""
    results: dict[str, dict] = {}
    for section in sections:
        log.info(f"[GeneralBot] running section={section} for user={user_id}")
        results[section] = _run_one_section(section, user_id, db)
    return results


def _build_run_plan(requested: str, snapshot: dict) -> list[str]:
    """
    Return the ordered list of sections to run, including any prerequisites
    that are not yet done.  The requested section is always last.
    """
    prereqs  = _SHOULD_RUN_BEFORE.get(requested, [])
    missing  = [p for p in prereqs if not snapshot["sections"].get(p, {}).get("done", False)]

    plan: list[str] = []
    for section in _SECTION_ORDER:
        if section in missing:
            plan.append(section)
    plan.append(requested)
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Natural language response generators
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_results(sections_run: list[str], results: dict[str, dict], snapshot: dict) -> str:
    """
    Ask the LLM to present the agent results as a natural conversational summary.
    """
    context_parts = [_snapshot_to_context(snapshot)]
    for section in sections_run:
        label  = _SECTION_LABELS.get(section, section)
        data   = json.dumps(results.get(section, {}), indent=2)[:3000]
        context_parts.append(f"=== JUST GENERATED: {label} ===\n{data}")

    labels = [_SECTION_LABELS.get(s, s) for s in sections_run]
    prompt = (
        f"I just generated the following for the founder: {', '.join(labels)}. "
        "Summarise the most important insights from this analysis in a natural, "
        "conversational way — 3 to 5 key points. End by suggesting what to explore next."
    )

    messages = [
        {"role": "system", "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(context_parts)},
        {"role": "user",   "content": prompt},
    ]

    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.5, max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def _respond_from_data(message: str, history: list, section: Optional[str],
                       snapshot: dict, section_data: Optional[dict]) -> str:
    """Answer a question grounded in the user's pipeline data."""
    context_parts = [_snapshot_to_context(snapshot)]
    if section_data and section:
        label = _SECTION_LABELS.get(section, section)
        context_parts.append(
            f"=== {label} DATA ===\n"
            + json.dumps(section_data, indent=2)[:4000]
        )

    messages = [
        {"role": "system", "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(context_parts)},
        *history[-20:],
        {"role": "user",   "content": message},
    ]

    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_general_bot(user_id: str, message: str, history: list) -> dict:
    """
    Process one conversation turn.

    Returns
    -------
    {
      "reply":   str   — the assistant's response (may contain invisible <!--PENDING:...-->)
      "intent":  str   — classified intent
      "section": str | None
      "action":  str   — "answered" | "ran_sections" | "needs_confirmation" |
                         "declined" | "status"
      # kept for backwards compatibility — always None / False now
      "route_to_trigger": None
      "trigger_needed":   False
    }
    """
    db = SessionLocal()
    try:
        snapshot = _load_pipeline_snapshot(user_id, db)

        # ── Step 1: check for a pending confirmation from the previous turn ──────
        pending = _extract_pending(history)
        if pending:
            classification = _classify_intent(message, history, snapshot)
            intent = classification.get("intent", "general_startup_chat")

            if intent == "confirm_action":
                log.info(f"[GeneralBot] confirmed run of {pending} for user={user_id}")
                results  = _run_sections_in_order(pending, user_id, db)
                snapshot = _load_pipeline_snapshot(user_id, db)   # refresh
                reply    = _summarise_results(pending, results, snapshot)
                return _resp(reply, "confirm_action", pending[-1], "ran_sections")

            if intent == "decline_action":
                reply = "No problem! Let me know if you want to run something else or have any questions about your startup."
                return _resp(reply, "decline_action", None, "declined")
            # else: treat as a fresh message (pending was stale)

        # ── Step 2: classify the fresh message ───────────────────────────────────
        classification = _classify_intent(message, history, snapshot)
        intent  = classification.get("intent", "general_startup_chat")
        section = classification.get("section")
        log.info(f"[GeneralBot] user={user_id} intent={intent} section={section}")

        # ── Out of scope ──────────────────────────────────────────────────────────
        if intent == "out_of_scope":
            return _resp(OUT_OF_SCOPE_RESPONSE, intent, section, "declined")

        # ── Pipeline status ───────────────────────────────────────────────────────
        if intent == "pipeline_status":
            reply = _respond_from_data(message, history, None, snapshot, None)
            return _resp(reply, intent, None, "status")

        # ── Idea intake (conversational agent — proxied turn by turn) ─────────────
        if intent == "run_section" and section == "idea_intake":
            has_profile = snapshot["sections"].get("profile", {}).get("done", False)
            if not has_profile:
                reply = (
                    "It looks like you haven't completed your startup profile yet. "
                    "Once that's set up I can help you define and refine your business idea!"
                )
                return _resp(reply, intent, section, "answered")

            from agents.ThreeIdeaIntakeAgent import run_idea_intake_chat
            result = run_idea_intake_chat(user_id=user_id, message=message, history=history)
            crud.save_idea_intake(db, user_id, result)
            reply = result.get("reply", "Tell me more about your idea.")
            return _resp(reply, intent, section, "answered")

        # ── Run an analysis section ───────────────────────────────────────────────
        if intent == "run_section" and section in _RUNNABLE_SECTIONS:
            # Hard requirement: idea + problems must exist (set by the pipeline)
            if not crud.get_idea(db, user_id) or not crud.get_problems(db, user_id):
                reply = (
                    "I can't run that analysis yet — your startup idea and problem "
                    "discovery haven't been set up. Have you completed the onboarding?"
                )
                return _resp(reply, intent, section, "answered")

            plan         = _build_run_plan(section, snapshot)
            missing_plan = plan[:-1]   # everything except the requested section

            if not missing_plan:
                # No prerequisites needed — just run it
                label   = _SECTION_LABELS.get(section, section)
                result  = _run_one_section(section, user_id, db)
                snapshot = _load_pipeline_snapshot(user_id, db)
                reply   = _summarise_results([section], {section: result}, snapshot)
                return _resp(reply, intent, section, "ran_sections")

            # Prerequisites are missing — ask for confirmation
            missing_labels = [_SECTION_LABELS.get(s, s) for s in missing_plan]
            target_label   = _SECTION_LABELS.get(section, section)
            confirmation_text = (
                f"To generate your **{target_label}**, I need to run "
                f"{len(missing_plan)} other {'analysis' if len(missing_plan) == 1 else 'analyses'} first:\n\n"
                + "\n".join(f"- {l}" for l in missing_labels)
                + f"\n\nShall I go ahead and run all {len(plan)} of them? Just say **yes** to confirm."
            )
            reply = _embed_pending(confirmation_text, plan)
            return _resp(reply, intent, section, "needs_confirmation")

        # ── Refine an existing section ────────────────────────────────────────────
        if intent == "refine_section" and section:
            is_done = snapshot["sections"].get(section, {}).get("done", False)
            label   = _SECTION_LABELS.get(section, section)
            if not is_done:
                reply = (
                    f"I haven't generated your **{label}** yet. "
                    f"Want me to run it now?"
                )
                return _resp(reply, "run_section", section, "answered")

            # Load section data and refine via chat
            section_data = _load_section_data(user_id, section, db)
            reply = _respond_from_data(message, history, section, snapshot, section_data)
            return _resp(reply, intent, section, "answered")

        # ── Default: chat from data ───────────────────────────────────────────────
        section_data = _load_section_data(user_id, section, db) if section else None
        reply = _respond_from_data(message, history, section, snapshot, section_data)
        return _resp(reply, intent, section, "answered")

    except Exception as e:
        log.error(f"[GeneralBot] error for user={user_id}: {e}", exc_info=True)
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resp(reply: str, intent: str, section: Optional[str], action: str) -> dict:
    return {
        "reply":            reply,
        "intent":           intent,
        "section":          section,
        "action":           action,
        # kept for backwards compatibility — frontend no longer needs to act on these
        "route_to_trigger": None,
        "trigger_needed":   False,
    }
