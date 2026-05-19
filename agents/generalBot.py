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
    """
    Classify the user's intent using the LLM on every message.

    No keyword shortcuts — keywords produce silent wrong results for edge cases
    like "I'm not sure yet" (would wrongly match 'not' → decline) or
    "definitely not" (matches both confirm and decline sets simultaneously).
    The LLM understands full context; keywords don't.

    The context explicitly tells the LLM whether a confirmation is pending
    and what sections are waiting, so confirm/decline detection is reliable.
    """
    pending_sections = _extract_pending(history)
    has_pending      = bool(pending_sections)

    done_sections    = [k for k, v in snapshot["sections"].items() if v["done"]]
    not_done         = [k for k, v in snapshot["sections"].items() if not v["done"]]
    last_bot = next(
        (m["content"][:400] for m in reversed(history) if m.get("role") == "assistant"),
        "none",
    )

    context_lines = [
        f"Completed sections: {done_sections}",
        f"Not yet generated: {not_done}",
        f"Awaiting user confirmation: {has_pending}",
    ]
    if has_pending:
        context_lines.append(
            f"Sections waiting to run (user must confirm): {pending_sections}"
        )
    context_lines.append(f"Last assistant message: {last_bot}")

    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
        {"role": "system", "content": "\n".join(context_lines)},
        *history[-6:],
        {"role": "user",   "content": message},
    ]

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=200,
        )
        return parse_llm_json(resp.choices[0].message.content)
    except Exception as e:
        log.warning("[GeneralBot] intent classification failed (%s) — defaulting to chat", e)
        return {"intent": "general_startup_chat", "section": None, "confidence": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Section dispatch table
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry: section_key → {
#   "module"  : dotted import path of the agent module
#   "fn"      : function name inside that module
#   "needs"   : {db_getter_key: kwarg_name} — extra DB rows to load
#   "save"    : crud function name to persist the result
#   "problems": True if the agent function accepts a `problems` kwarg
# }
#
# Adding a new section = adding one dict entry here. Nothing else changes.

_DB_GETTERS = {
    "profile":          lambda db, uid: crud.get_profile(db, uid),
    "customers":        lambda db, uid: crud.get_customers(db, uid),
    "competition":      lambda db, uid: crud.get_competition(db, uid),
    "market_potential": lambda db, uid: crud.get_market_potential(db, uid),
    "idea_strategy":    lambda db, uid: crud.get_idea_strategy(db, uid),
    "business_model":   lambda db, uid: crud.get_business_model(db, uid),
    "functions_list":   lambda db, uid: crud.get_functions_list(db, uid),
    "mvp_planning":     lambda db, uid: crud.get_mvp_planning(db, uid),
    "unit_economics":   lambda db, uid: crud.get_unit_economics(db, uid),
}

_SECTION_DISPATCH: dict[str, dict] = {
    "customers": {
        "module": "agents.FourCustomersAgent", "fn": "run_customers_analysis",
        "needs": {"profile": "profile"},
        "save": "save_customers", "problems": True,
    },
    "competition": {
        "module": "agents.FiveCompetitionAgent", "fn": "run_competition_analysis",
        "needs": {"customers": "customers"},
        "save": "save_competition", "problems": True,
    },
    "market_potential": {
        "module": "agents.SixMaketPotential", "fn": "run_market_potential",
        "needs": {"customers": "customers", "competition": "competition"},
        "save": "save_market_potential", "problems": True,
    },
    "idea_strategy": {
        "module": "agents.SevenIdeaStrategy", "fn": "run_idea_strategy",
        "needs": {"customers": "customers", "competition": "competition", "market_potential": "market_potential"},
        "save": "save_idea_strategy", "problems": True,
    },
    "business_model": {
        "module": "agents.EightBusinessModel", "fn": "run_business_model",
        "needs": {"customers": "customers", "competition": "competition",
                  "market_potential": "market_potential", "idea_strategy": "strategy"},
        "save": "save_business_model", "problems": True,
    },
    "functions_list": {
        "module": "agents.NineFunctionsList", "fn": "run_functions_list",
        "needs": {"customers": "customers", "competition": "competition",
                  "market_potential": "market_potential", "idea_strategy": "strategy",
                  "business_model": "business_model"},
        "save": "save_functions_list", "problems": True,
    },
    "mvp_planning": {
        "module": "agents.TenMVPPlanning", "fn": "run_mvp_planning",
        "needs": {"customers": "customers", "market_potential": "market_potential",
                  "idea_strategy": "strategy", "business_model": "business_model",
                  "functions_list": "functions_list"},
        "save": "save_mvp_planning", "problems": True,
    },
    "unit_economics": {
        "module": "agents.ElevenUnitEconomicsAgent", "fn": "run_unit_economics",
        "needs": {"customers": "customers", "market_potential": "market_potential",
                  "idea_strategy": "strategy", "business_model": "business_model",
                  "mvp_planning": "mvp_planning"},
        "save": "save_unit_economics", "problems": False,  # unit_economics has no problems param
    },
    "go_to_market": {
        "module": "agents.TwelveGoToMarket", "fn": "run_go_to_market",
        "needs": {"customers": "customers", "competition": "competition",
                  "market_potential": "market_potential", "idea_strategy": "strategy",
                  "business_model": "business_model", "mvp_planning": "mvp_planning",
                  "unit_economics": "unit_economics"},
        "save": "save_go_to_market", "problems": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Agent runners  (run agent + save to DB)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_section(section: str, user_id: str, db) -> dict:
    """
    Look up the section in _SECTION_DISPATCH, load required DB context,
    call the agent, save the result, and return it.

    Adding a new section requires only a new entry in _SECTION_DISPATCH above.
    """
    import importlib

    cfg = _SECTION_DISPATCH.get(section)
    if cfg is None:
        raise ValueError(f"Unknown runnable section: {section!r}")

    # Base data every agent needs
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    kwargs: dict = {
        "user_id": user_id,
        "idea":    idea_row.current_idea if idea_row else "",
    }
    if cfg["problems"]:
        kwargs["problems"] = problems_row.data if problems_row else {}

    # Load additional context rows requested by this section
    for db_key, kwarg_name in cfg["needs"].items():
        row = _DB_GETTERS[db_key](db, user_id)
        kwargs[kwarg_name] = row.data if row else None

    # Lazy-import the agent module and call the function
    module    = importlib.import_module(cfg["module"])
    agent_fn  = getattr(module, cfg["fn"])
    result    = agent_fn(**kwargs)

    # Persist via the matching crud save function
    getattr(crud, cfg["save"])(db, user_id, result)

    return result


def _run_new_user_pipeline_inline(user_id: str, db) -> str:
    """
    Run the new-user pipeline synchronously inside the general bot:
      1. Profile Analysis  (skipped if already done)
      2. Problem Discovery (skipped if already done)
      3. Idea Generation   (always runs — that's what the user asked for)

    Saves every result to DB (same as the background orchestrator does).
    Returns a conversational reply presenting the generated idea.
    """
    from agents.PipelineRunner import (
        run_profile_analysis,
        run_problem_discovery,
        generate_opening_idea,
        build_context,
    )

    questionnaire = crud.get_questionnaire_from_profile(db, user_id)
    skills        = crud.get_skills_from_profile(db, user_id)

    # Persist the questionnaire blob so downstream CRUD helpers can find it
    crud.save_questionnaire_output(db, user_id, questionnaire)
    crud.upsert_pipeline_status(db, user_id, "running", "profile_analysis")

    # ── Step 1: Profile Analysis (skip if already done) ──────────────────────
    existing_profile = crud.get_profile(db, user_id)
    if existing_profile:
        profile = existing_profile.data
        log.info(f"[GeneralBot] profile_analysis already done for user={user_id}, skipping")
    else:
        log.info(f"[GeneralBot] running profile_analysis for user={user_id}")
        profile = run_profile_analysis(questionnaire, skills)
        crud.save_profile(db, user_id, profile)

    # ── Step 2: Problem Discovery (skip if already done) ─────────────────────
    crud.upsert_pipeline_status(db, user_id, "running", "problem_discovery")
    existing_problems = crud.get_problems(db, user_id)
    if existing_problems:
        problems = existing_problems.data
        log.info(f"[GeneralBot] problem_discovery already done for user={user_id}, skipping")
    else:
        log.info(f"[GeneralBot] running problem_discovery for user={user_id}")
        problems = run_problem_discovery(profile, questionnaire)
        crud.save_problems(db, user_id, problems)

    # ── Step 3: Idea Generation (always run — user asked for an idea) ─────────
    crud.upsert_pipeline_status(db, user_id, "running", "idea_chat")
    log.info(f"[GeneralBot] generating idea for user={user_id}")
    context = build_context(problems, questionnaire, skills)
    idea    = generate_opening_idea(context)
    crud.save_idea(db, user_id, idea, [])
    crud.upsert_pipeline_status(db, user_id, "done", None)

    log.info(f"[GeneralBot] pipeline complete — idea saved for user={user_id}")

    return (
        "Here's a startup idea based on your profile and the problems I discovered:\n\n"
        + idea
        + "\n\nWhat do you think? Want to refine it, or shall we start building out the full business plan?"
    )


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
        if pending:  # noqa: SIM102
            classification = _classify_intent(message, history, snapshot)
            intent = classification.get("intent", "general_startup_chat")

            if intent == "confirm_action":
                log.info(f"[GeneralBot] confirmed run of {pending} for user={user_id}")
                results  = _run_sections_in_order(pending, user_id, db)
                snapshot = _load_pipeline_snapshot(user_id, db)
                reply    = _summarise_results(pending, results, snapshot)
                return _resp_and_save(db, user_id, message, reply, "confirm_action", pending[-1], "ran_sections")

            if intent == "decline_action":
                reply = "No problem! Let me know if you want to run something else or have any questions about your startup."
                return _resp_and_save(db, user_id, message, reply, "decline_action", None, "declined")
            # else: treat as a fresh message (pending was stale)

        # ── Step 2: classify the fresh message ───────────────────────────────────
        classification = _classify_intent(message, history, snapshot)
        intent  = classification.get("intent", "general_startup_chat")
        section = classification.get("section")
        log.info(f"[GeneralBot] user={user_id} intent={intent} section={section}")

        # ── Out of scope ──────────────────────────────────────────────────────────
        if intent == "out_of_scope":
            return _resp_and_save(db, user_id, message, OUT_OF_SCOPE_RESPONSE, intent, section, "declined")

        # ── Pipeline status ───────────────────────────────────────────────────────
        if intent == "pipeline_status":
            reply = _respond_from_data(message, history, None, snapshot, None)
            return _resp_and_save(db, user_id, message, reply, intent, None, "status")

        # ── Idea — choose path based on what the user has ────────────────────────
        if intent == "run_section" and section in ("idea_intake", "idea"):
            user_profile_row = crud.get_user_profile(db, user_id)
            has_questionnaire = (
                user_profile_row is not None
                and user_profile_row.questionnaire_json is not None
            )

            if has_questionnaire:
                # ── NEW USER PATH: run full pipeline inline ────────────────────
                # questionnaire_json + skills_json exist → ProfileAnalysis →
                # ProblemDiscovery → IdeaGeneration, all saved to DB
                reply = _run_new_user_pipeline_inline(user_id, db)
                return _resp_and_save(db, user_id, message, reply, intent, "idea", "ran_sections")

            # ── RETURNING USER PATH: proxy through IdeaIntake chat agent ──────
            # No questionnaire — user already has or is describing a specific idea
            has_profile_or_intake = (
                snapshot["sections"].get("profile", {}).get("done", False)
                or user_profile_row is not None
                or crud.get_idea_intake(db, user_id) is not None
            )
            if not has_profile_or_intake:
                reply = (
                    "It looks like you haven't completed your startup profile yet. "
                    "Once that's set up I can help you define and refine your business idea!"
                )
                return _resp_and_save(db, user_id, message, reply, intent, section, "answered")

            from agents.ThreeIdeaIntakeAgent import run_idea_intake
            result = run_idea_intake(
                user_id=user_id,
                user_message=message,
                history=history,
            )
            crud.save_idea_intake(db, user_id, result)
            reply = result.get("reply", "Tell me more about your idea.")
            return _resp_and_save(db, user_id, message, reply, intent, section, "answered")

        # ── Run an analysis section ───────────────────────────────────────────────
        if intent == "run_section" and section in _RUNNABLE_SECTIONS:
            if not crud.get_idea(db, user_id) or not crud.get_problems(db, user_id):
                reply = (
                    "I can't run that analysis yet — your startup idea and problem "
                    "discovery haven't been set up. Have you completed the onboarding?"
                )
                return _resp_and_save(db, user_id, message, reply, intent, section, "answered")

            plan         = _build_run_plan(section, snapshot)
            missing_plan = plan[:-1]

            if not missing_plan:
                result   = _run_one_section(section, user_id, db)
                snapshot = _load_pipeline_snapshot(user_id, db)
                reply    = _summarise_results([section], {section: result}, snapshot)
                return _resp_and_save(db, user_id, message, reply, intent, section, "ran_sections")

            # Prerequisites missing — ask for confirmation
            missing_labels    = [_SECTION_LABELS.get(s, s) for s in missing_plan]
            target_label      = _SECTION_LABELS.get(section, section)
            confirmation_text = (
                f"To generate your **{target_label}**, I need to run "
                f"{len(missing_plan)} other {'analysis' if len(missing_plan) == 1 else 'analyses'} first:\n\n"
                + "\n".join(f"- {l}" for l in missing_labels)
                + f"\n\nShall I go ahead and run all {len(plan)} of them? Just say **yes** to confirm."
            )
            reply = _embed_pending(confirmation_text, plan)
            return _resp_and_save(db, user_id, message, reply, intent, section, "needs_confirmation")

        # ── Refine an existing section ────────────────────────────────────────────
        if intent == "refine_section" and section:
            is_done = snapshot["sections"].get(section, {}).get("done", False)
            label   = _SECTION_LABELS.get(section, section)
            if not is_done:
                reply = f"I haven't generated your **{label}** yet. Want me to run it now?"
                return _resp_and_save(db, user_id, message, reply, "run_section", section, "answered")

            section_data = _load_section_data(user_id, section, db)
            reply = _respond_from_data(message, history, section, snapshot, section_data)
            return _resp_and_save(db, user_id, message, reply, intent, section, "answered")

        # ── Default: chat from data ───────────────────────────────────────────────
        section_data = _load_section_data(user_id, section, db) if section else None
        reply = _respond_from_data(message, history, section, snapshot, section_data)
        return _resp_and_save(db, user_id, message, reply, intent, section, "answered")

    except Exception as e:
        log.error(f"[GeneralBot] error for user={user_id}: {e}", exc_info=True)
        raise
    finally:
        db.close()


def _resp_and_save(
    db,
    user_id: str,
    user_message: str,
    reply: str,
    intent: str,
    section: Optional[str],
    action: str,
) -> dict:
    """Build the response dict and persist the turn to chat_messages."""
    try:
        crud.save_general_bot_messages(db, user_id, user_message, reply)
    except Exception as e:
        # Non-fatal: log but don't break the response
        log.warning(f"[GeneralBot] could not save chat messages for user={user_id}: {e}")
    return _resp(reply, intent, section, action)


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
