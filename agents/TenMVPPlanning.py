"""
agents/TenMVPPlanning.py
=========================
Pipeline step 10 — MVP Planning.

Inputs  : idea + problems + customers + competition + market_potential +
          strategy + business_model + functions_list (all from DB)
Outputs : mvp_goal, riskiest_assumptions, scope (included/excluded),
          core_user_flows, build_plan, validation_experiments,
          launch_criteria, testing_plan, qa_checklist,
          first_100_users_plan, summary

Search strategy:
  MVP scope and timeline are grounded by searching how similar no-code
  products were launched (Reddit startup launches, Indie Hackers, Product Hunt).
  Also searches for validation experiments that worked in the same domain.

Also exposes a section-scoped refinement chat and a streaming chat variant.

DB flow:
  run_mvp_planning() → saves to mvp_planning_results + agent_runs
  chat_mvp_planning() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from agents.config import client, GROQ_MODEL, SERPER_API_KEY
from db.connection import SessionLocal
from db import crud
from System_Messages.mvp_planning_prompt import (
    MVP_PLANNING_PROMPT,
    MVP_PLANNING_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_search_queries(
    idea: str,
    business_model: dict,
    functions_list: dict,
    region: str,
) -> list[str]:
    """
    Build MVP-focused queries targeting:
    - How similar no-code products were launched
    - Validation experiments that worked in this domain
    - Indie Hackers / Product Hunt launches in this space
    - First 100 users acquisition tactics for this model
    """
    region_mod  = region.strip() if region and region.lower() != "global" else ""
    idea_seed   = " ".join(idea.split()[:6]) if idea else ""
    bm_type     = business_model.get("business_model_type", "")
    no_code_stack = [
        t.get("tool", "") for t in
        business_model.get("no_code_stack", []) + functions_list.get("no_code_stack", [])
    ]
    primary_tool = no_code_stack[0] if no_code_stack else ""

    queries: list[str] = [
        f"{idea_seed} MVP no-code launch case study",
        f"{bm_type} MVP launch how to validate {region_mod}".strip(),
        f"how to get first 100 users {idea_seed} {region_mod}".strip(),
        f"{idea_seed} indie hackers product hunt launch story",
        f"no-code {bm_type} launch in 4 weeks tutorial".strip(),
        f"how to validate {idea_seed} startup idea reddit".strip(),
        f"{primary_tool} marketplace launch guide".strip() if primary_tool else "",
        f"MVP testing plan {bm_type} checklist".strip(),
    ]

    return [q for q in queries if q][:12]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_analysis_context(
    idea: str,
    problems: dict,
    customers: dict,
    strategy: dict,
    business_model: dict,
    functions_list: dict,
    region: str,
    sources: list,
    source_mode: str,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Region: {region or 'Global'}",
        "",
    ]

    # Riskiest assumptions from strategy — directly inform MVP scope
    if strategy:
        assumptions = strategy.get("key_assumptions", [])
        if assumptions:
            parts += ["=== KEY ASSUMPTIONS FROM STRATEGY (these drive MVP scope) ==="]
            for a in assumptions[:4]:
                parts.append(
                    f"  [{a.get('id')}] {a.get('assumption','')} "
                    f"— risk: {a.get('risk_level','?')} "
                    f"→ validate via: {a.get('how_to_validate','')}"
                )

        vp = strategy.get("validation_priorities", [])
        if vp:
            parts += ["", "=== VALIDATION PRIORITIES FROM STRATEGY ==="]
            for v in vp[:3]:
                parts.append(
                    f"  [{v.get('id')}] {v.get('what_to_validate','')} "
                    f"— method: {v.get('method','')} "
                    f"— metric: {v.get('success_metric','')}"
                )

    # Core functions — these become MVP scope
    if functions_list:
        parts += ["", "=== CORE FUNCTIONS (MVP must include these) ==="]
        for f in functions_list.get("core_functions", []):
            parts.append(
                f"  [{f.get('id')}] {f.get('name','')} "
                f"[{f.get('priority','')}] — tool: {f.get('no_code_solution','')}"
            )

        parts += ["", "=== NICE-TO-HAVE (MVP must EXCLUDE these) ==="]
        for nf in functions_list.get("nice_to_have_functions", []):
            parts.append(
                f"  [{nf.get('id')}] {nf.get('name','')} "
                f"— add when: {nf.get('when_to_add','')}"
            )

        warnings = functions_list.get("feature_creep_warnings", [])
        if warnings:
            parts += ["", "=== FEATURE CREEP WARNINGS ==="]
            for w in warnings:
                parts.append(f"  ⚠ {w}")

        stack = functions_list.get("no_code_stack", [])
        if stack:
            parts += ["", "=== NO-CODE STACK ==="]
            for t in stack:
                parts.append(
                    f"  {t.get('tool','')} — {t.get('purpose','')} "
                    f"(${t.get('monthly_cost_usd', 0)}/mo)"
                )

    # Business model — shapes revenue validation in MVP
    if business_model:
        parts += ["", "=== BUSINESS MODEL CONSTRAINTS ==="]
        bm_type = business_model.get("business_model_type", "")
        if bm_type:
            parts.append(f"  Model type: {bm_type}")
        for rs in business_model.get("revenue_streams", [])[:2]:
            parts.append(
                f"  Revenue: {rs.get('name','')} at {rs.get('pricing','')} "
                f"— {rs.get('type','')}"
            )
        fit = business_model.get("founder_fit_assessment", {})
        if fit:
            parts.append(f"  Founder can execute: {fit.get('can_execute','?')}")
            parts.append(f"  Biggest risk: {fit.get('biggest_execution_risk','')}")

    # Primary customer — defines user flows
    if customers:
        primary = customers.get("primary_segment", {})
        pid     = primary.get("id", "")
        parts += ["", "=== PRIMARY CUSTOMER (defines core user flows) ==="]
        for seg in customers.get("customer_segments", []):
            if seg.get("id") == pid:
                parts.append(f"  {seg.get('name','')} — {seg.get('description','')[:100]}")
                parts.append(f"  Where to find: {', '.join(seg.get('where_to_find',[])[:3])}")
                break
        ea = customers.get("early_adopter_profile", "")
        if ea:
            parts.append(f"  Early adopter: {ea}")

    # Problems — validation experiments prove these are real
    parts += ["", "=== TOP PROBLEMS (validation experiments must prove these are real) ==="]
    for p in problems.get("problems", [])[:3]:
        parts.append(f"  [{p.get('id','?')}] {p.get('title','')}")

    if sources:
        parts += ["", f"=== WEB RESEARCH ({len(sources)} sources — MVP launches + validation tactics) ==="]
        for s in sources:
            parts.append(f"[{s['url']}]\n{s['content'][:600]}")
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {source_mode} ===",
            "No web sources. Base MVP plan on strategy assumptions and functions list above.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_mvp_planning(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    market_potential: dict = None,
    strategy: dict = None,
    business_model: dict = None,
    functions_list: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Generate the MVP plan for the saved idea.

    Steps:
      1. Extract region, no-code stack, and core functions from prior analysis
      2. Search for similar MVP launches, validation experiments, first-user tactics
      3. Build enriched LLM context from all pipeline data + web sources
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id          : str  — for DB persistence
    idea             : str  — saved idea text
    problems         : dict — ProblemDiscovery output
    customers        : dict — FourCustomersAgent output (optional)
    market_potential : dict — SixMaketPotential output (optional)
    strategy         : dict — SevenIdeaStrategy output (optional)
    business_model   : dict — EightBusinessModel output (optional)
    functions_list   : dict — NineFunctionsList output (optional)
    custom_prompt    : str  — extra instruction for regenerate-custom
    """
    region = "Global"
    if market_potential:
        region = market_potential.get("target_region", region)
    elif customers:
        for seg in customers.get("customer_segments", []):
            desc = (seg.get("description", "") + seg.get("size_estimate", "")).lower()
            for kw in ["mena", "egypt", "saudi", "uae", "gulf"]:
                if kw in desc:
                    region = kw.upper()
                    break
            if region != "Global":
                break

    # ── 1. Search ────────────────────────────────────────────────────────────
    sources: list = []
    source_mode   = "synthesis_only"

    if SERPER_API_KEY:
        queries = _build_search_queries(
            idea, business_model or {}, functions_list or {}, region
        )
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=10)
        source_mode = "web_sourced" if sources else "synthesis_only"
    else:
        log.warning("SERPER_API_KEY not set — building MVP plan from pipeline data only")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(
        idea, problems,
        customers or {}, strategy or {},
        business_model or {}, functions_list or {},
        region, sources, source_mode,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": MVP_PLANNING_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
        max_tokens=4000,
    )

    raw    = response.choices[0].message.content
    result = parse_llm_json(raw)
    result["source_mode"]  = source_mode
    result["sources_used"] = len(sources)
    result["sources_list"] = [{"url": s["url"], "title": s.get("title", s["url"])} for s in sources]

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_mvp_planning(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────

def chat_mvp_planning(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the MVP plan through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT MVP PLAN ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": MVP_PLANNING_CHAT_PROMPT},
        {"role": "system", "content": context},
        *history[-20:],
        {"role": "user",   "content": user_message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()
