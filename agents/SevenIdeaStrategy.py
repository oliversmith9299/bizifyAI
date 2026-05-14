"""
agents/SevenIdeaStrategy.py
============================
Pipeline step 7 — Idea Strategy.

Inputs  : idea + problems + customers + competition + market_potential (all from DB)
Outputs : value_proposition, positioning, core_promise, differentiation_strategy,
          key_assumptions, validation_priorities, strategic_direction,
          unfair_advantages, strategic_risks, summary

Search strategy:
  Strategy is primarily a synthesis agent — it reasons over all previous outputs.
  Web search adds value by finding:
  - Real competitor positioning statements to benchmark against
  - Examples of successful validation experiments in the same industry
  - Startup strategy case studies in similar verticals/regions
  - Positioning examples from direct competitors found in agent 5

Also exposes a section-scoped refinement chat.

DB flow:
  run_idea_strategy() → saves to idea_strategy_results + agent_runs
  chat_idea_strategy() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY
from db.connection import SessionLocal
from db import crud
from System_Messages.idea_strategy_prompt import (
    IDEA_STRATEGY_PROMPT,
    IDEA_STRATEGY_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_search_queries(
    idea: str,
    problems: dict,
    competition: dict,
    region: str,
) -> list:
    """
    Build strategy-focused search queries.
    Targets competitor positioning, validation methods, and startup strategy examples.
    """
    region_mod = region.strip() if region and region.lower() != "global" else ""
    idea_seed  = " ".join(idea.split()[:6]) if idea else ""

    # Extract industry from problems
    industry = ""
    for p in problems.get("problems", [])[:2]:
        if p.get("industry"):
            industry = p["industry"]
            break

    # Extract top competitor names from competition data
    competitor_names = [
        c.get("name", "")
        for c in competition.get("direct_competitors", [])[:2]
        if c.get("name")
    ]

    queries = []

    # Positioning and value proposition benchmarks
    queries.append(f"{idea_seed} value proposition examples {region_mod}".strip())
    queries.append(f"{industry} startup positioning statement examples".strip())
    if competitor_names:
        for name in competitor_names:
            queries.append(f"{name} value proposition positioning")
            queries.append(f"{name} marketing strategy")

    # Validation experiments in this domain
    queries.append(f"{idea_seed} startup validation experiments {region_mod}".strip())
    queries.append(f"how to validate {industry} startup idea {region_mod}".strip())
    queries.append(f"{industry} startup MVP validation case study".strip())

    # Strategy case studies
    queries.append(f"successful {industry} startup strategy {region_mod}".strip())
    queries.append(f"{industry} go to market strategy {region_mod}".strip())

    return [q for q in queries if q][:12]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_analysis_context(
    idea: str,
    problems: dict,
    customers: dict,
    competition: dict,
    market_potential: dict,
    region: str,
    sources: list,
    source_mode: str,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Region: {region or 'Global'}",
        "",
        "=== DISCOVERED PROBLEMS (top 3) ===",
    ]

    for p in problems.get("problems", [])[:3]:
        parts.append(
            f"[{p.get('id','?')}] {p.get('title','')}\n"
            f"  Gap: {p.get('gap_opportunity','')}"
        )

    if customers:
        parts += ["", "=== CUSTOMER STRATEGY SNAPSHOT ==="]
        primary = customers.get("primary_segment", {})
        pid = primary.get("id", "")
        for seg in customers.get("customer_segments", []):
            marker = "★ PRIMARY" if seg.get("id") == pid else "  "
            parts.append(
                f"{marker} [{seg.get('id')}] {seg.get('name','')} — "
                f"WTP: {seg.get('willingness_to_pay','?')}, "
                f"where: {', '.join(seg.get('where_to_find', [])[:2])}"
            )
        ea = customers.get("early_adopter_profile", "")
        if ea:
            parts.append(f"Early adopter: {ea}")

    if competition:
        parts += ["", "=== COMPETITIVE POSITION ==="]
        for c in competition.get("direct_competitors", [])[:2]:
            parts.append(
                f"  [{c.get('id','?')}] {c.get('name','')} — "
                f"weaknesses: {', '.join(c.get('weaknesses', [])[:2])}"
            )
        gaps = competition.get("positioning_gaps", [])
        for g in gaps[:2]:
            parts.append(f"  Gap: {g.get('gap','')} → {g.get('opportunity','')}")
        diff_opps = competition.get("differentiation_opportunities", [])
        if diff_opps:
            parts.append(f"  Best differentiator: {diff_opps[0]}")

    if market_potential:
        parts += ["", "=== MARKET OPPORTUNITY SNAPSHOT ==="]
        tam = market_potential.get("tam", {})
        som = market_potential.get("som", {})
        score = market_potential.get("opportunity_score", "?")
        parts.append(
            f"  TAM: {tam.get('value','')} {tam.get('unit','')}, "
            f"SOM: {som.get('value','')} {som.get('unit','')} over {som.get('timeline','?')}"
        )
        parts.append(f"  Attractiveness: {market_potential.get('opportunity_attractiveness','?')} (score {score}/10)")
        timing = market_potential.get("timing_assessment", {})
        if timing:
            parts.append(f"  Timing: {'✅ Right time' if timing.get('is_right_time') else '⚠️ Questionable timing'} — {timing.get('reasoning','')}")

    if sources:
        parts += ["", f"=== WEB RESEARCH ({len(sources)} sources — use for positioning benchmarks) ==="]
        for s in sources:
            parts.append(f"[{s['url']}]\n{s['content'][:600]}")
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {source_mode} ===",
            "No web sources. Base strategy on the research data above.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────
def run_idea_strategy(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    competition: dict = None,
    market_potential: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Build a full strategic direction for the saved idea.

    Steps:
      1. Extract region and competitor names from prior analysis
      2. Search for competitor positioning and validation examples
      3. Synthesise all pipeline data + search results into strategy
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id          : str  — for DB persistence
    idea             : str  — saved idea text
    problems         : dict — ProblemDiscovery output
    customers        : dict — FourCustomersAgent output (optional)
    competition      : dict — FiveCompetitionAgent output (optional)
    market_potential : dict — SixMaketPotential output (optional)
    custom_prompt    : str  — extra instruction for regenerate-custom
    """
    # Infer region from market_potential or customers
    region = "Global"
    if market_potential:
        region = market_potential.get("target_region", region)
    elif customers:
        for seg in customers.get("customer_segments", []):
            for kw in ["MENA", "Egypt", "Saudi", "UAE", "Gulf"]:
                if kw.lower() in (seg.get("description", "") + seg.get("size_estimate", "")).lower():
                    region = kw
                    break
            if region != "Global":
                break

    # ── 1. Search ────────────────────────────────────────────────────────────
    sources = []
    source_mode = "synthesis_only"

    if SERPER_API_KEY:
        queries = _build_search_queries(idea, problems, competition or {}, region)
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=10)
        source_mode = "web_enriched" if sources else "synthesis_only"
    else:
        log.warning("SERPER_API_KEY not set — using synthesis only")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(
        idea, problems,
        customers or {}, competition or {}, market_potential or {},
        region, sources, source_mode,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": IDEA_STRATEGY_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.4,
        max_tokens=4000,
    )

    raw    = response.choices[0].message.content
    result = validate_section_output("idea_strategy", parse_llm_json(raw))
    result["source_mode"]  = source_mode
    result["sources_used"] = len(sources)
    result["sources_list"] = [{"url": s["url"], "title": s.get("title", s["url"])} for s in sources]

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_idea_strategy(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────
def chat_idea_strategy(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the idea strategy through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT IDEA STRATEGY ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": IDEA_STRATEGY_CHAT_PROMPT},
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
