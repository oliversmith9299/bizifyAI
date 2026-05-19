"""
agents/SixMaketPotential.py
============================
Pipeline step 6 — Market Potential Analysis.

Inputs  : saved idea + problems + customers + competition (all from DB)
Outputs : market_definition, TAM, SAM, SOM, market_trends, growth_drivers,
          adoption_barriers, timing_assessment, PESTEL, opportunity_score, summary

Search strategy (most impactful agent for web search):
  Market sizing REQUIRES real data — LLM alone produces unreliable numbers.
  Searches target: industry reports, market size statistics, PESTEL factors,
  regional e-commerce/sector data, trend articles.

DB flow:
  run_market_potential() → saves to market_potential_results + agent_runs
  chat_market_potential() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline, SearchResults
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL

_SEARCH_DOMAINS = [
    "statista.com", "grandviewresearch.com", "mordorintelligence.com",
    "mckinsey.com", "ibisworld.com", "marketsandmarkets.com",
    "businessresearchinsights.com", "alliedmarketresearch.com",
]

_EXTRACTION_SCHEMA = {
    "market_size": "total addressable market size in dollars (TAM)",
    "growth_rate": "annual growth rate or CAGR percentage",
    "market_year": "year the market size figure refers to",
    "forecast": "projected market size in a future year",
    "key_drivers": "main factors driving market growth",
    "region_size": "market size or growth specific to MENA, Egypt, or the target region",
    "adoption_barriers": "main obstacles to market adoption",
}
from db.connection import SessionLocal
from db import crud
from System_Messages.market_potential_prompt import (
    MARKET_POTENTIAL_PROMPT,
    MARKET_POTENTIAL_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_search_queries(
    idea: str,
    problems: dict,
    customers: dict,
    competition: dict,
    region: str,
) -> list:
    """
    Build market-research-focused queries.
    Targets industry reports, TAM data, PESTEL factors, and regional trends.
    """
    region_mod = region.strip() if region and region.lower() != "global" else ""

    idea_seed = " ".join(idea.split()[:6]) if idea else ""

    # Industry from problems
    industries = []
    for p in problems.get("problems", [])[:3]:
        ind = p.get("industry", "").strip()
        if ind and ind not in industries:
            industries.append(ind)
    industry_seed = industries[0] if industries else idea_seed

    # Primary customer name
    primary_customer = ""
    if customers:
        pid = customers.get("primary_segment", {}).get("id", "")
        for seg in customers.get("customer_segments", []):
            if seg.get("id") == pid:
                primary_customer = seg.get("name", "")
                break

    queries = []

    # Market size / TAM queries — highest value
    size_templates = [
        "{industry} market size {region} 2024",
        "{industry} market size statistics report",
        "{industry} TAM total addressable market",
        "{idea} market opportunity {region}",
        "{industry} market growth forecast 2025 2026",
    ]
    for t in size_templates:
        queries.append(t.format(
            industry=industry_seed,
            idea=idea_seed,
            region=region_mod,
        ).strip())

    # Trend and driver queries
    trend_templates = [
        "{industry} market trends {region} 2024",
        "{industry} growth drivers {region}",
        "{industry} e-commerce growth {region}",
    ]
    for t in trend_templates:
        queries.append(t.format(industry=industry_seed, region=region_mod).strip())

    # PESTEL queries
    queries.append(f"{industry_seed} regulatory environment {region_mod}".strip())
    queries.append(f"{industry_seed} {region_mod} economic outlook".strip())
    queries.append(f"{industry_seed} consumer behavior trends {region_mod}".strip())

    # Barrier queries
    if primary_customer:
        queries.append(
            f"barriers to buying {industry_seed} online {region_mod}".strip()
        )
    queries.append(f"{industry_seed} challenges adoption {region_mod}".strip())

    return [q for q in queries if q][:14]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_analysis_context(
    idea: str,
    problems: dict,
    customers: dict,
    competition: dict,
    region: str,
    search: SearchResults,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Target region: {region or 'Global'}",
        "",
        "=== DISCOVERED PROBLEMS ===",
    ]

    for p in problems.get("problems", [])[:4]:
        parts.append(
            f"[{p.get('id','?')}] {p.get('title','')} "
            f"(industry: {p.get('industry','?')}, score: {p.get('validation_score',0)})"
        )

    if customers:
        parts += ["", "=== CUSTOMER SNAPSHOT ==="]
        primary = customers.get("primary_segment", {})
        pid = primary.get("id", "")
        for seg in customers.get("customer_segments", []):
            marker = "★ PRIMARY" if seg.get("id") == pid else ""
            parts.append(
                f"  {marker} [{seg.get('id')}] {seg.get('name','')} — "
                f"size: {seg.get('size_estimate','?')}, "
                f"WTP: {seg.get('willingness_to_pay','?')}"
            )

    if competition:
        parts += ["", "=== COMPETITION SNAPSHOT ==="]
        for c in competition.get("direct_competitors", [])[:3]:
            parts.append(f"  Competitor: {c.get('name','')} — {c.get('market_share_estimate','?')}")
        gaps = competition.get("positioning_gaps", [])
        if gaps:
            parts.append(f"  Key gap: {gaps[0].get('gap','')}")

    web_context = search.to_prompt_context()
    if web_context:
        parts += ["", web_context]
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {search.source_mode} ===",
            "No web sources. Use LLM knowledge for estimates. Mark all numbers as rough estimates.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────
def run_market_potential(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    competition: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Estimate the market opportunity for the saved idea.

    Steps:
      1. Extract region from customers / competition data
      2. Run market-research-focused web searches (TAM data, trends, PESTEL)
      3. Enrich LLM prompt with real sources
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id       : str  — for DB persistence
    idea          : str  — saved idea text from idea_results
    problems      : dict — output of ProblemDiscovery
    customers     : dict — output of FourCustomersAgent (optional)
    competition   : dict — output of FiveCompetitionAgent (optional)
    custom_prompt : str  — extra instruction for regenerate-custom
    """

    # Extract region from customers or problems
    region = "Global"
    if customers:
        for seg in customers.get("customer_segments", []):
            desc = seg.get("description", "") + seg.get("size_estimate", "")
            for keyword in ["MENA", "Egypt", "Saudi", "UAE", "Gulf", "Global"]:
                if keyword.lower() in desc.lower():
                    region = keyword
                    break
            if region != "Global":
                break

    # ── 1. Search + extract ──────────────────────────────────────────────────
    search = run_search_pipeline(
        queries=_build_search_queries(idea, problems, customers or {}, competition or {}, region),
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=[idea.split()[0] if idea else "", region, "market size", "TAM", "growth rate"],
        include_domains=_SEARCH_DOMAINS,
        groq_client=client,
        extraction_model=GROQ_EXTRACTION_MODEL,
        serper_fallback_key=SERPER_API_KEY,
    )

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    context = _build_analysis_context(
        idea, problems, customers or {}, competition or {}, region, search,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": MARKET_POTENTIAL_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,   # lower = more consistent numbers
            max_tokens=4000,
        )

        raw = response.choices[0].message.content
        try:
            result = validate_section_output("market_potential", parse_llm_json(raw))
        except ValueError as e:
            log.error("[SixMarketPotential] JSON parse failed: %s", e)
            raise
        sources_used, sources_list = search.to_sources_meta()
        result["source_mode"]   = search.source_mode
        result["sources_used"]  = sources_used
        result["sources_list"]  = sources_list
        result["target_region"] = result.get("target_region") or region

        # ── 4. Persist ───────────────────────────────────────────────────────────
        db = SessionLocal()
        try:
            crud.save_market_potential(db, user_id, result)
        finally:
            db.close()

        return result
    except Exception as e:
        log.error("[SixMarketPotential] LLM call failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────
def chat_market_potential(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the market potential analysis through conversation.
    Scoped to this section only — cannot modify any other pipeline data.
    """
    context = (
        "=== CURRENT MARKET POTENTIAL ANALYSIS ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": MARKET_POTENTIAL_CHAT_PROMPT},
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
