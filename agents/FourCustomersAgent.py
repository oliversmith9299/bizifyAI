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

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline, SearchResults
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL
from db.connection import SessionLocal
from db import crud
from System_Messages.customers_prompt import CUSTOMERS_ANALYSIS_PROMPT, CUSTOMERS_CHAT_PROMPT

log = logging.getLogger(__name__)

_SEARCH_DOMAINS = [
    "reddit.com", "quora.com", "statista.com", "nielsen.com",
    "medium.com", "thinkwithgoogle.com", "pewresearch.org",
]

_EXTRACTION_SCHEMA = {
    "target_demographics": "age group, location, income level, occupation of main customers",
    "pain_points": "specific problems or frustrations customers face",
    "buying_behavior": "how customers discover, evaluate, and purchase products like this",
    "willingness_to_pay": "price sensitivity or what they currently spend on solutions",
    "market_size": "size or estimated count of this customer segment",
    "acquisition_channels": "where these customers spend time online and how to reach them",
}


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
    search: SearchResults,
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

    web_context = search.to_prompt_context()
    if web_context:
        parts += ["", web_context]
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {search.source_mode} ===",
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

    # ── 1. Search + extract ──────────────────────────────────────────────────
    search = run_search_pipeline(
        queries=_build_search_queries(idea, problems, region),
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=[idea.split()[0] if idea else "", region, "customer", "segment", "pain point"],
        include_domains=_SEARCH_DOMAINS,
        groq_client=client,
        extraction_model=GROQ_EXTRACTION_MODEL,
        serper_fallback_key=SERPER_API_KEY,
    )

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    context = _build_analysis_context(idea, problems, profile or {}, search)

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CUSTOMERS_ANALYSIS_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.4,
            max_tokens=4000,
        )
    except Exception as e:
        log.error("[FourCustomersAgent] LLM call failed: %s", e)
        raise

    raw = response.choices[0].message.content
    try:
        result = validate_section_output("customers", parse_llm_json(raw))
    except ValueError as e:
        log.error("[FourCustomers] JSON parse failed: %s", e)
        raise

    sources_used, sources_list = search.to_sources_meta()
    result["source_mode"]  = search.source_mode
    result["sources_used"] = sources_used
    result["sources_list"] = sources_list

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
