"""
agents/FiveCompetitionAgent.py
================================
Pipeline step 5 — Competition Analysis.

Inputs  : saved idea + discovered problems + customer analysis (all from DB)
Outputs : direct_competitors, indirect_alternatives, substitute_solutions,
          positioning_gaps, Porter's Five Forces, VRIO, differentiation_opportunities,
          summary

Search flow (same pattern as TwoProblemDiscovery):
  1. Build competitor-focused search queries from the idea + industry
  2. Search Google via Serper API
  3. Fetch real content (product pages, Reddit threads, review sites)
  4. Feed enriched sources into the LLM prompt
  5. Fall back to profile-only analysis if search returns nothing

Also exposes a section-scoped refinement chat.

DB flow:
  run_competition_analysis() → saves to competition_results + agent_runs
  chat_competition()         → stateless; history managed by the caller
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from db.connection import SessionLocal
from db import crud
from System_Messages.competition_prompt import (
    COMPETITION_ANALYSIS_PROMPT,
    COMPETITION_CHAT_PROMPT,
)

load_dotenv()

log = logging.getLogger(__name__)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_search_queries(
    idea: str,
    problems: dict,
    customers: dict,
    region: str = "",
) -> list:
    """
    Generate targeted search queries to surface real competitors,
    alternative products, and market reviews.
    """
    region_mod = region.strip() if region and region.lower() != "global" else ""

    idea_seed = " ".join(idea.split()[:6]) if idea else ""

    industry_seeds = []
    for p in problems.get("problems", [])[:3]:
        ind = p.get("industry", "")
        if ind and ind not in industry_seeds:
            industry_seeds.append(ind)

    primary_customer = ""
    if customers:
        primary_id  = customers.get("primary_segment", {}).get("id", "")
        for seg in customers.get("customer_segments", []):
            if seg.get("id") == primary_id:
                primary_customer = seg.get("name", "")
                break

    queries = []

    # Competitor discovery
    templates_idea = [
        "{idea} competitors {region}",
        "{idea} alternatives comparison",
        "best {idea} apps platforms {region}",
        "{idea} vs competitors review",
        "{idea} market leaders {region}",
    ]
    for t in templates_idea:
        queries.append(t.format(idea=idea_seed, region=region_mod).strip())

    # Industry-level competitive landscape
    for ind in industry_seeds[:2]:
        queries.append(f"{ind} top startups {region_mod}".strip())
        queries.append(f"{ind} competitive landscape {region_mod}".strip())
        queries.append(f"{ind} market players reddit".strip())

    # Substitute solutions
    if primary_customer:
        queries.append(f"how do {primary_customer} solve {idea_seed} today".strip())
        queries.append(f"{primary_customer} alternatives to {idea_seed}".strip())

    return [q for q in queries if q][:14]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_analysis_context(
    idea: str,
    problems: dict,
    customers: dict,
    sources: list,
    source_mode: str,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        "",
        "=== DISCOVERED PROBLEMS ===",
    ]

    for p in problems.get("problems", []):
        parts.append(
            f"[{p.get('id','?')}] {p.get('title','')}\n"
            f"  Gap opportunity: {p.get('gap_opportunity','')}"
        )

    if customers:
        parts += ["", "=== CUSTOMER ANALYSIS ==="]
        primary = customers.get("primary_segment", {})
        if primary:
            parts.append(
                f"Primary segment: {primary.get('id','')} — {primary.get('reason','')}"
            )
        for seg in customers.get("customer_segments", []):
            parts.append(
                f"  [{seg.get('id')}] {seg.get('name','')} — "
                f"pain: {seg.get('pain_intensity','')}, "
                f"WTP: {seg.get('willingness_to_pay','')}"
            )
        channels = customers.get("acquisition_channels", [])
        if channels:
            parts.append(f"Channels they use: {', '.join(channels[:4])}")

    if sources:
        parts += ["", f"=== WEB RESEARCH ({len(sources)} sources) ==="]
        for s in sources:
            parts.append(f"[{s['url']}]\n{s['content'][:600]}")
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {source_mode} ===",
            "No web sources found. Base analysis on idea and problem data above.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────
def run_competition_analysis(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Generate competition analysis for the saved idea.

    Steps:
      1. Build competitor-focused search queries
      2. Gather real web sources (product pages, reviews, Reddit)
      3. Enrich LLM prompt with fetched content
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id       : str  — for DB persistence
    idea          : str  — saved idea text from idea_results
    problems      : dict — output of ProblemDiscovery
    customers     : dict — output of FourCustomersAgent (optional but recommended)
    custom_prompt : str  — extra instruction for regenerate-custom
    """


    region = ""
    if customers:
        # Try to get region from primary segment or acquisition channels
        for seg in customers.get("customer_segments", []):
            desc = seg.get("description", "")
            if any(r in desc.upper() for r in ["MENA", "EGYPT", "GULF", "GLOBAL"]):
                for keyword in ["MENA", "Egypt", "Gulf", "Global"]:
                    if keyword.upper() in desc.upper():
                        region = keyword
                        break
            if region:
                break

    # ── 1. Search ────────────────────────────────────────────────────────────
    sources = []
    source_mode = "profile_derived"

    if SERPER_API_KEY:
        queries = _build_search_queries(idea, problems, customers or {}, region)
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=12)
        source_mode = "web_sourced" if sources else "profile_derived"
    else:
        log.warning("SERPER_API_KEY not set — skipping web search")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(idea, problems, customers or {}, sources, source_mode)

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": COMPETITION_ANALYSIS_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.4,
        max_tokens=4000,
    )

    raw    = response.choices[0].message.content
    result = parse_llm_json(raw)
    result["source_mode"]  = source_mode
    result["sources_used"] = len(sources)

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_competition(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────
def chat_competition(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the competition analysis through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT COMPETITION ANALYSIS ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": COMPETITION_CHAT_PROMPT},
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
