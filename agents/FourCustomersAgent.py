"""
agents/FourCustomersAgent.py
============================
Pipeline step 4 — Customer Analysis.

Inputs  : saved idea + discovered problems + optional profile (all from DB)
Outputs : customer_segments, primary_segment, CATWOE, personas,
          acquisition_channels, early_adopter_profile, summary

Search flow (same pattern as TwoProblemDiscovery):
  1. Build targeted queries from the idea + problems
  2. Search Google via Serper API
  3. Fetch real content from each result (Reddit + web)
  4. Feed enriched sources into the LLM prompt
  5. Fall back to profile-only analysis if search returns nothing

Also exposes a section-scoped refinement chat.

DB flow:
  run_customers_analysis() → saves to customers_results + agent_runs
  chat_customers()         → stateless; history managed by the caller
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from db.connection import SessionLocal
from db import crud
from System_Messages.customers_prompt import CUSTOMERS_ANALYSIS_PROMPT, CUSTOMERS_CHAT_PROMPT

log = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_search_queries(idea: str, problems: dict, region: str = "") -> list:
    """
    Generate targeted search queries to find real customer research data.
    Extracts keywords from the idea summary and problem titles.
    """
    region_mod = region.strip() if region and region.lower() != "global" else ""

    # Seed keywords from problem titles (first 3 words of each title)
    problem_seeds = []
    for p in problems.get("problems", [])[:3]:
        title = p.get("title", "")
        words = " ".join(title.split()[:4])
        if words:
            problem_seeds.append(words)

    # Seed from idea (first 6 words)
    idea_seed = " ".join(idea.split()[:6]) if idea else ""

    # Get customer segments if already discovered
    segment_seeds = []
    for seg in problems.get("customer_segments", [])[:2]:
        if isinstance(seg, str) and seg:
            segment_seeds.append(seg)

    queries = []

    templates_idea = [
        "{idea} target customers {region}",
        "{idea} user behavior research {region}",
        "{idea} who uses reddit {region}",
    ]
    for t in templates_idea:
        queries.append(t.format(idea=idea_seed, region=region_mod).strip())

    templates_problem = [
        "{problem} customer frustrations {region}",
        "{problem} user pain points reddit",
        "{problem} who buys {region}",
    ]
    for seed in problem_seeds[:2]:
        for t in templates_problem:
            queries.append(t.format(problem=seed, region=region_mod).strip())

    for seg in segment_seeds:
        queries.append(f"{seg} pain points {region_mod}".strip())
        queries.append(f"{seg} spending habits {region_mod}".strip())

    return [q for q in queries if q][:12]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_analysis_context(
    idea: str,
    problems: dict,
    profile: dict,
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
            f"  Target customer : {p.get('target_customer','')}\n"
            f"  Gap opportunity : {p.get('gap_opportunity','')}\n"
            f"  Validation score: {p.get('validation_score', 0)}"
        )

    if problems.get("customer_segments"):
        parts += ["", "=== PREVIOUSLY IDENTIFIED SEGMENTS ==="]
        for seg in problems["customer_segments"]:
            parts.append(f"  - {seg}")

    if profile:
        parts += [
            "",
            "=== FOUNDER CONTEXT ===",
            f"  Industries    : {profile.get('recommended_industries', [])}",
            f"  Problem spaces: {profile.get('recommended_problem_spaces', [])}",
        ]

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
def run_customers_analysis(
    user_id: str,
    idea: str,
    problems: dict,
    profile: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Generate customer analysis for the saved idea.

    Steps:
      1. Extract region from profile (or default to Global)
      2. Build and run targeted web searches
      3. Enrich LLM prompt with fetched content
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id       : str  — for DB persistence
    idea          : str  — saved idea text from idea_results
    problems      : dict — output of ProblemDiscovery
    profile       : dict — optional founder profile for region + context
    custom_prompt : str  — extra instruction for regenerate-custom
    """
    region = ""
    if profile:
        region = (
            profile.get("founder_profile", {}).get("target_region", "")
            or ""
        )

    # ── 1. Search ────────────────────────────────────────────────────────────
    sources = []
    source_mode = "profile_derived"

    if SERPER_API_KEY:
        queries = _build_search_queries(idea, problems, region)
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=10)
        source_mode = "web_sourced" if sources else "profile_derived"
    else:
        log.warning("SERPER_API_KEY not set — skipping web search")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(idea, problems, profile or {}, sources, source_mode)

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": CUSTOMERS_ANALYSIS_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.4,
        max_tokens=4000,
    )

    raw    = response.choices[0].message.content
    result = parse_llm_json(raw)
    result["source_mode"]   = source_mode
    result["sources_used"]  = len(sources)

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_customers(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────
def chat_customers(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the customer analysis through conversation.
    Scoped to this section only — cannot modify any other pipeline data.
    History is owned by the caller (route) and passed on every call.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT CUSTOMER ANALYSIS ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": CUSTOMERS_CHAT_PROMPT},
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
