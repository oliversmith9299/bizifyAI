"""
agents/EightBusinessModel.py
=============================
Pipeline step 8 — Business Model Design.

Inputs  : idea + problems + customers + competition + market_potential +
          idea_strategy (all from DB)
Outputs : business_model_type, Business Model Canvas (all 9 blocks),
          revenue_streams, pricing_strategy, key_metrics,
          business_model_risks, founder_fit_assessment, summary

Search strategy:
  Real pricing benchmarks are critical here — the LLM confabulates numbers.
  Searches target: similar companies' pricing pages, competitor take-rates,
  platform fee comparisons, and industry revenue model case studies.

Also exposes a section-scoped refinement chat and a streaming chat variant.

DB flow:
  run_business_model() → saves to business_model_results + agent_runs
  chat_business_model() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline, SearchResults
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL

_SEARCH_DOMAINS = [
    "saastr.com", "openview.co", "baremetrics.com", "profitwell.com",
    "techcrunch.com", "a16z.com", "firstround.com", "ycombinator.com",
]

_EXTRACTION_SCHEMA = {
    "revenue_model": "how the business makes money (commission, subscription, freemium, etc.)",
    "pricing_strategy": "pricing approach and specific price points mentioned",
    "commission_rate": "commission or transaction fee percentage if applicable",
    "subscription_price": "monthly or annual subscription price if applicable",
    "gross_margin": "typical gross margin percentage for this business type",
    "monetisation_timeline": "when companies typically become profitable or reach break-even",
    "pricing_benchmarks": "industry standard pricing for similar businesses",
}
from db.connection import SessionLocal
from db import crud
from System_Messages.business_model_prompt import (
    BUSINESS_MODEL_PROMPT,
    BUSINESS_MODEL_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_search_queries(
    idea: str,
    strategy: dict,
    competition: dict,
    region: str,
) -> list[str]:
    """
    Build business-model-focused queries targeting real pricing data,
    comparable platform fees, and revenue model benchmarks.
    """
    region_mod = region.strip() if region and region.lower() != "global" else ""
    idea_seed  = " ".join(idea.split()[:6]) if idea else ""

    bm_type = strategy.get("differentiation_strategy", {}).get("approach", "")

    # Extract competitor names for pricing benchmarks
    competitor_names = [
        c.get("name", "")
        for c in competition.get("direct_competitors", [])[:3]
        if c.get("name")
    ]

    queries: list[str] = []

    # Pricing and revenue model queries — highest value
    queries.append(f"{idea_seed} pricing model revenue {region_mod}".strip())
    queries.append(f"{idea_seed} business model how do they make money".strip())
    queries.append(f"marketplace commission rate {idea_seed}".strip())
    queries.append(f"{idea_seed} subscription pricing {region_mod}".strip())

    # Competitor pricing pages
    for name in competitor_names:
        queries.append(f"{name} pricing fees commission rate")
        queries.append(f"{name} business model revenue")

    # Industry cost benchmarks
    queries.append(f"{idea_seed} cost structure startup {region_mod}".strip())
    queries.append(f"{bm_type} business model examples {idea_seed}".strip() if bm_type else "")
    queries.append(f"how much does it cost to run {idea_seed} marketplace".strip())

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
    strategy: dict,
    region: str,
    search: SearchResults,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Region: {region or 'Global'}",
        "",
    ]

    # Strategy signals — the most important input for business model design
    if strategy:
        vp = strategy.get("value_proposition", {})
        diff = strategy.get("differentiation_strategy", {})
        parts += [
            "=== STRATEGY SNAPSHOT ===",
            f"  Value prop : {vp.get('statement', '')}",
            f"  Core promise: {strategy.get('core_promise', '')}",
            f"  Approach   : {diff.get('approach', '')}",
        ]
        for d in diff.get("key_differentiators", [])[:3]:
            parts.append(f"  Differentiator: {d}")

    # Customer snapshot — drives revenue stream design
    if customers:
        primary = customers.get("primary_segment", {})
        pid     = primary.get("id", "")
        parts += ["", "=== CUSTOMER SNAPSHOT ==="]
        for seg in customers.get("customer_segments", []):
            marker = "★ PRIMARY" if seg.get("id") == pid else "  "
            parts.append(
                f"{marker} [{seg.get('id')}] {seg.get('name','')} — "
                f"WTP: {seg.get('willingness_to_pay','?')}, "
                f"size: {seg.get('size_estimate','?')}"
            )
        ea = customers.get("early_adopter_profile", "")
        if ea:
            parts.append(f"  Early adopter: {ea}")

    # Competition — competitor pricing benchmarks
    if competition:
        parts += ["", "=== COMPETITOR PRICING BENCHMARKS ==="]
        for c in competition.get("direct_competitors", [])[:3]:
            parts.append(
                f"  {c.get('name','')} — model: {c.get('pricing_model','?')}"
            )
        for gap in competition.get("positioning_gaps", [])[:2]:
            parts.append(f"  Gap to exploit: {gap.get('opportunity','')}")

    # Market potential — sets scale expectations
    if market_potential:
        som = market_potential.get("som", {})
        parts += [
            "",
            "=== MARKET SCALE ===",
            f"  SOM: {som.get('value','')} {som.get('unit','')} "
            f"over {som.get('timeline','')}",
            f"  Attractiveness: {market_potential.get('opportunity_attractiveness','?')}",
        ]

    # Top problems — inform channel and key activity choices
    parts += ["", "=== TOP PROBLEMS (shape key activities) ==="]
    for p in problems.get("problems", [])[:3]:
        parts.append(f"  [{p.get('id','?')}] {p.get('title','')}")

    # Web research — pricing and cost benchmarks
    web_context = search.to_prompt_context()
    if web_context:
        parts += ["", web_context]
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {search.source_mode} ===",
            "No web sources. Use reasonable estimates — label them clearly as estimates.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_business_model(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    competition: dict = None,
    market_potential: dict = None,
    strategy: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Design the business model for the saved idea.

    Steps:
      1. Extract region and competitor names from prior analysis
      2. Search for real pricing benchmarks and competitor revenue models
      3. Build enriched LLM context from all pipeline data + web sources
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id          : str  — for DB persistence
    idea             : str  — saved idea text
    problems         : dict — ProblemDiscovery output
    customers        : dict — FourCustomersAgent output (optional)
    competition      : dict — FiveCompetitionAgent output (optional)
    market_potential : dict — SixMaketPotential output (optional)
    strategy         : dict — SevenIdeaStrategy output (optional)
    custom_prompt    : str  — extra instruction for regenerate-custom
    """
    # Infer region from market_potential or customers
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

    # ── 1. Search + extract ──────────────────────────────────────────────────
    search = run_search_pipeline(
        queries=_build_search_queries(idea, strategy or {}, competition or {}, region),
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=[idea.split()[0] if idea else "", region, "pricing", "revenue model", "commission"],
        include_domains=_SEARCH_DOMAINS,
        groq_client=client,
        extraction_model=GROQ_EXTRACTION_MODEL,
        serper_fallback_key=SERPER_API_KEY,
    )

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    context = _build_analysis_context(
        idea, problems,
        customers or {}, competition or {},
        market_potential or {}, strategy or {},
        region, search,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": BUSINESS_MODEL_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,   # lower for consistent numbers
            max_tokens=4000,
        )

        raw = response.choices[0].message.content
        try:
            result = validate_section_output("business_model", parse_llm_json(raw))
        except ValueError as e:
            log.error("[EightBusinessModel] JSON parse failed: %s", e)
            raise
        sources_used, sources_list = search.to_sources_meta()
        result["source_mode"]  = search.source_mode
        result["sources_used"] = sources_used
        result["sources_list"] = sources_list

        # ── 4. Persist ───────────────────────────────────────────────────────────
        db = SessionLocal()
        try:
            crud.save_business_model(db, user_id, result)
        finally:
            db.close()

        return result
    except Exception as e:
        log.error("[EightBusinessModel] LLM call failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────

def chat_business_model(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the business model through conversation.
    Scoped to this section only — cannot modify any other pipeline data.
    """
    context = (
        "=== CURRENT BUSINESS MODEL ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": BUSINESS_MODEL_CHAT_PROMPT},
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
