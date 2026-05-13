"""
agents/ElevenUnitEconomicsAgent.py
====================================
Pipeline step 11 — Unit Economics.

Inputs  : idea + problems + customers + market_potential + strategy +
          business_model + mvp_planning (all from DB)
Outputs : revenue_model_summary, pricing_assumptions, cost_assumptions,
          gross_margin, cac_analysis, ltv_analysis, ltv_cac_ratio,
          payback_period, break_even, monthly_projections,
          weak_assumptions, pricing_tests, overall_viability, summary

Search strategy:
  Unit economics REQUIRES real benchmarks — LLM-only numbers are unreliable.
  Searches target: industry CAC benchmarks, e-commerce LTV/churn data,
  marketplace take-rate comparisons, MENA e-commerce financial benchmarks.

Also exposes a section-scoped refinement chat and a streaming chat variant.

DB flow:
  run_unit_economics() → saves to unit_economics_results + agent_runs
  chat_unit_economics() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from agents.config import client, GROQ_MODEL, SERPER_API_KEY
from db.connection import SessionLocal
from db import crud
from System_Messages.unit_economics_prompt import (
    UNIT_ECONOMICS_PROMPT,
    UNIT_ECONOMICS_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_search_queries(
    idea: str,
    business_model: dict,
    region: str,
) -> list[str]:
    """
    Build financial-benchmark-focused queries targeting:
    - CAC benchmarks for this business model type in the region
    - LTV and churn benchmarks for comparable products
    - Take-rate / commission-rate industry norms
    - MENA e-commerce financial benchmarks
    """
    region_mod  = region.strip() if region and region.lower() != "global" else ""
    idea_seed   = " ".join(idea.split()[:5]) if idea else ""
    bm_type     = business_model.get("business_model_type", "")
    commission  = str(
        business_model.get("revenue_streams", [{}])[0].get("pricing", "")
        if business_model.get("revenue_streams") else ""
    )

    queries: list[str] = [
        f"{bm_type} CAC customer acquisition cost benchmark {region_mod}".strip(),
        f"{bm_type} LTV lifetime value churn rate benchmark".strip(),
        f"{idea_seed} unit economics metrics startup".strip(),
        f"marketplace commission rate take rate benchmark {region_mod}".strip(),
        f"e-commerce LTV CAC ratio {region_mod}".strip(),
        f"{idea_seed} average order value repeat purchase rate".strip(),
        f"startup {bm_type} break even analysis".strip(),
        f"MENA e-commerce financial metrics CAC LTV {region_mod}".strip(),
        f"{bm_type} gross margin benchmark industry average".strip(),
        f"how much does it cost to acquire customer {idea_seed}".strip(),
    ]

    return [q for q in queries if q][:12]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_analysis_context(
    idea: str,
    customers: dict,
    market_potential: dict,
    strategy: dict,
    business_model: dict,
    mvp_planning: dict,
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

    # Business model — primary input for revenue and cost assumptions
    if business_model:
        bm_type = business_model.get("business_model_type", "")
        parts += ["=== BUSINESS MODEL FINANCIALS ==="]
        if bm_type:
            parts.append(f"  Model type: {bm_type}")

        for rs in business_model.get("revenue_streams", []):
            parts.append(
                f"  Revenue [{rs.get('id')}]: {rs.get('name','')} — "
                f"{rs.get('type','')} at {rs.get('pricing','')} "
                f"({rs.get('estimated_monthly_at_scale','')})"
            )

        cost = business_model.get("business_model_canvas", {}).get("cost_structure", {})
        fixed = cost.get("total_fixed_monthly_usd", 0)
        if fixed:
            parts.append(f"  Fixed monthly costs: ${fixed}")
        for fc in cost.get("fixed_costs", [])[:5]:
            parts.append(f"    - {fc.get('item','')}: ${fc.get('monthly_usd',0)}/mo")

        pricing = business_model.get("pricing_strategy", {})
        if pricing:
            parts.append(f"  Pricing approach: {pricing.get('approach','')}")
            pts = pricing.get("price_points", {})
            for k, v in list(pts.items())[:4]:
                parts.append(f"    {k}: {v}")

    # Customer insights — drives LTV and churn assumptions
    if customers:
        primary = customers.get("primary_segment", {})
        pid     = primary.get("id", "")
        parts += ["", "=== CUSTOMER FINANCIAL SIGNALS ==="]
        for seg in customers.get("customer_segments", []):
            if seg.get("id") == pid:
                parts.append(
                    f"  Primary: {seg.get('name','')} — "
                    f"WTP: {seg.get('willingness_to_pay','?')}, "
                    f"size: {seg.get('size_estimate','?')}"
                )
                ch = seg.get("where_to_find", [])
                if ch:
                    parts.append(f"  Acquisition channels: {', '.join(ch[:3])}")
                break
        acq = customers.get("acquisition_channels", [])
        if acq:
            parts.append(f"  All channels: {', '.join(acq[:4])}")

    # Market potential — scale ceiling for projections
    if market_potential:
        som = market_potential.get("som", {})
        parts += [
            "",
            "=== MARKET SCALE (bounds for projections) ===",
            f"  SOM: {som.get('value','')} {som.get('unit','')} "
            f"over {som.get('timeline','')}",
        ]

    # MVP launch plan — timeline and first-user targets
    if mvp_planning:
        parts += ["", "=== MVP FINANCIAL TARGETS ==="]
        phases = mvp_planning.get("build_plan", {}).get("phases", [])
        for ph in phases[:3]:
            parts.append(f"  Phase {ph.get('phase')}: {ph.get('milestone','')}")
        u100 = mvp_planning.get("first_100_users_plan", "")
        if u100:
            parts.append(f"  First 100 users: {u100[:200]}")

    # Strategy — validation priorities give CAC and conversion rate hints
    if strategy:
        vp = strategy.get("validation_priorities", [])
        if vp:
            parts += ["", "=== VALIDATION PRIORITIES (inform CAC estimates) ==="]
            for v in vp[:3]:
                parts.append(
                    f"  [{v.get('id')}] {v.get('what_to_validate','')} "
                    f"— metric: {v.get('success_metric','')}"
                )

    if sources:
        parts += [
            "",
            f"=== WEB RESEARCH ({len(sources)} sources — USE FOR BENCHMARKS) ===",
            "Priority: use any CAC, LTV, churn, or AOV benchmarks found below.",
        ]
        for s in sources:
            parts.append(f"[{s['url']}]\n{s['content'][:700]}")
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {source_mode} ===",
            "No web sources. Use conservative industry estimates. Label all numbers as estimates.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_economics(
    user_id: str,
    idea: str,
    customers: dict = None,
    market_potential: dict = None,
    strategy: dict = None,
    business_model: dict = None,
    mvp_planning: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Analyse the unit economics for the saved idea.

    Steps:
      1. Extract region, revenue model type, and pricing from business_model
      2. Search for CAC/LTV/churn/margin benchmarks in this domain and region
      3. Build enriched LLM context from all pipeline data + web sources
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id          : str  — for DB persistence
    idea             : str  — saved idea text
    customers        : dict — FourCustomersAgent output (optional)
    market_potential : dict — SixMaketPotential output (optional)
    strategy         : dict — SevenIdeaStrategy output (optional)
    business_model   : dict — EightBusinessModel output (optional; primary input)
    mvp_planning     : dict — TenMVPPlanning output (optional)
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
    source_mode   = "llm_derived"

    if SERPER_API_KEY:
        queries = _build_search_queries(idea, business_model or {}, region)
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=12)
        source_mode = "web_sourced" if sources else "llm_derived"
    else:
        log.warning("SERPER_API_KEY not set — unit economics based on LLM estimates only")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(
        idea,
        customers or {}, market_potential or {},
        strategy or {}, business_model or {},
        mvp_planning or {},
        region, sources, source_mode,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": UNIT_ECONOMICS_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,   # very low — financial numbers must be consistent
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
        crud.save_unit_economics(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────

def chat_unit_economics(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine unit economics through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT UNIT ECONOMICS ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": UNIT_ECONOMICS_CHAT_PROMPT},
        {"role": "system", "content": context},
        *history[-20:],
        {"role": "user",   "content": user_message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1400,
    )

    return response.choices[0].message.content.strip()
