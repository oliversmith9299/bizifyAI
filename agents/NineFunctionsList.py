"""
agents/NineFunctionsList.py
============================
Pipeline step 9 — Product Functions List.

Inputs  : idea + problems + customers + competition + market_potential +
          strategy + business_model (all from DB)
Outputs : product_type, core_functions, nice_to_have_functions,
          future_capabilities, feature_creep_warnings,
          function_to_pain_map, function_to_business_model_map,
          no_code_stack, summary

Search strategy:
  Web search targets competitor feature pages, no-code tool comparisons,
  and product capability examples in the same industry. This helps ground
  the LLM in what similar products actually built vs what founders imagine.

Also exposes a section-scoped refinement chat and a streaming chat variant.

DB flow:
  run_functions_list() → saves to functions_list_results + agent_runs
  chat_functions_list() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline, SearchResults
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL

_SEARCH_DOMAINS = [
    "producthunt.com", "g2.com", "capterra.com", "techcrunch.com",
    "indiehackers.com", "getapp.com", "alternativeto.net",
]

_EXTRACTION_SCHEMA = {
    "core_features": "essential features that all similar products have",
    "differentiating_features": "features that set successful products apart from competitors",
    "mvp_features": "minimum set of features needed to launch",
    "avoided_features": "features that are commonly requested but rarely used or too complex",
    "no_code_tools": "no-code or low-code tools used to build similar products",
    "technical_complexity": "which features are considered technically difficult to build",
}
from db.connection import SessionLocal
from db import crud
from System_Messages.functions_list_prompt import (
    FUNCTIONS_LIST_PROMPT,
    FUNCTIONS_LIST_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_search_queries(
    idea: str,
    competition: dict,
    business_model: dict,
    region: str,
) -> list[str]:
    """
    Build product-feature-focused queries:
    - Competitor feature lists and pricing pages
    - No-code tool comparisons for this product type
    - Feature creep traps in similar products
    """
    region_mod    = region.strip() if region and region.lower() != "global" else ""
    idea_seed     = " ".join(idea.split()[:6]) if idea else ""
    bm_type       = business_model.get("business_model_type", "")
    no_code_stack = [t.get("tool", "") for t in business_model.get("no_code_stack", [])[:3]]

    competitor_names = [
        c.get("name", "")
        for c in competition.get("direct_competitors", [])[:3]
        if c.get("name")
    ]

    queries: list[str] = []

    # Competitor feature comparison — highest value for this agent
    for name in competitor_names:
        queries.append(f"{name} features list product capabilities")
        queries.append(f"{name} product review what does it do")

    # Product type features
    queries.append(f"{bm_type} product must-have features".strip() if bm_type else "")
    queries.append(f"{idea_seed} product features {region_mod}".strip())
    queries.append(f"{idea_seed} what features to build first startup".strip())

    # No-code tool comparisons
    for tool in no_code_stack:
        if tool:
            queries.append(f"{tool} features capabilities limitations")

    # Feature creep and MVP scope
    queries.append(f"{idea_seed} MVP scope minimum viable product features".strip())
    queries.append(f"feature creep {bm_type} startup mistakes".strip() if bm_type else "")

    return [q for q in queries if q][:12]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_analysis_context(
    idea: str,
    problems: dict,
    customers: dict,
    competition: dict,
    strategy: dict,
    business_model: dict,
    region: str,
    search: SearchResults,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Region: {region or 'Global'}",
        "",
    ]

    # Value proposition — tells the LLM what the product MUST deliver
    if strategy:
        vp = strategy.get("value_proposition", {})
        parts += [
            "=== VALUE PROPOSITION (functions must deliver this) ===",
            f"  Statement   : {vp.get('statement', '')}",
            f"  Key benefit : {vp.get('key_benefit', '')}",
            f"  Differentiator: {vp.get('differentiator', '')}",
            f"  Core promise: {strategy.get('core_promise', '')}",
        ]

    # Customer pains — each core function must solve at least one pain
    if customers:
        parts += ["", "=== CUSTOMER PAIN POINTS (functions must solve these) ==="]
        for seg in customers.get("customer_segments", [])[:2]:
            parts.append(
                f"  [{seg.get('id')}] {seg.get('name','')} — "
                f"pain: {seg.get('why_they_care','')[:120]}"
            )
        ea = customers.get("early_adopter_profile", "")
        if ea:
            parts.append(f"  Early adopter: {ea}")

    # Competitor gaps — differentiation functions come from here
    if competition:
        parts += ["", "=== COMPETITOR GAPS (differentiation functions come from here) ==="]
        for c in competition.get("direct_competitors", [])[:3]:
            weaknesses = ", ".join(c.get("weaknesses", [])[:2])
            parts.append(f"  {c.get('name','')} weak at: {weaknesses}")
        for gap in competition.get("positioning_gaps", [])[:2]:
            parts.append(f"  Gap to capture: {gap.get('opportunity','')}")
        diff_opps = competition.get("differentiation_opportunities", [])
        for opp in diff_opps[:2]:
            parts.append(f"  Differentiation: {opp}")

    # Business model — functions must enable revenue streams
    if business_model:
        parts += ["", "=== BUSINESS MODEL (functions must enable these revenue streams) ==="]
        bm_type = business_model.get("business_model_type", "")
        if bm_type:
            parts.append(f"  Model type: {bm_type}")
        for rs in business_model.get("revenue_streams", [])[:3]:
            parts.append(
                f"  Revenue stream [{rs.get('id')}] {rs.get('name','')} "
                f"— {rs.get('type','')}: {rs.get('pricing','')}"
            )
        stack = business_model.get("no_code_stack", [])
        if stack:
            tools = ", ".join(t.get("tool", "") for t in stack[:4])
            parts.append(f"  No-code stack already chosen: {tools}")

    # Top problems — for pain map
    parts += ["", "=== VALIDATED PROBLEMS (for function-to-pain mapping) ==="]
    for p in problems.get("problems", [])[:4]:
        parts.append(
            f"  [{p.get('id','?')}] {p.get('title','')} "
            f"— gap: {p.get('gap_opportunity','')[:100]}"
        )

    # Web research — competitor features and no-code tool capabilities
    web_context = search.to_prompt_context()
    if web_context:
        parts += ["", web_context]
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {search.source_mode} ===",
            "No web sources. Base function list on strategy, pain points, and competitor data above.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_functions_list(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    competition: dict = None,
    market_potential: dict = None,
    strategy: dict = None,
    business_model: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Generate the product functions list for the saved idea.

    Steps:
      1. Extract region, competitor names, and no-code stack from prior analysis
      2. Search for competitor features, tool capabilities, MVP scope examples
      3. Build enriched LLM context from all pipeline data + web sources
      4. Call LLM → parse JSON → save to DB

    Parameters
    ----------
    user_id        : str  — for DB persistence
    idea           : str  — saved idea text
    problems       : dict — ProblemDiscovery output
    customers      : dict — FourCustomersAgent output (optional)
    competition    : dict — FiveCompetitionAgent output (optional)
    market_potential: dict — SixMaketPotential output (optional)
    strategy       : dict — SevenIdeaStrategy output (optional)
    business_model : dict — EightBusinessModel output (optional)
    custom_prompt  : str  — extra instruction for regenerate-custom
    """
    # Infer region
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
        queries=_build_search_queries(idea, competition or {}, business_model or {}, region),
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=[idea.split()[0] if idea else "", region, "features", "MVP", "no-code"],
        include_domains=_SEARCH_DOMAINS,
        groq_client=client,
        extraction_model=GROQ_EXTRACTION_MODEL,
        serper_fallback_key=SERPER_API_KEY,
    )

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    context = _build_analysis_context(
        idea, problems,
        customers or {}, competition or {},
        strategy or {}, business_model or {},
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
                {"role": "system", "content": FUNCTIONS_LIST_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
    except Exception as e:
        log.error("[NineFunctionsList] LLM call failed: %s", e)
        raise

    raw = response.choices[0].message.content
    try:
        result = validate_section_output("functions_list", parse_llm_json(raw))
    except ValueError as e:
        log.error("[NineFunctionsList] JSON parse failed: %s", e)
        raise
    sources_used, sources_list = search.to_sources_meta()
    result["source_mode"]  = search.source_mode
    result["sources_used"] = sources_used
    result["sources_list"] = sources_list

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_functions_list(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────

def chat_functions_list(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the product functions list through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT PRODUCT FUNCTIONS LIST ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": FUNCTIONS_LIST_CHAT_PROMPT},
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
