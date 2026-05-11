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
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from agents.utils import gather_sources, parse_llm_json, truncate_sources
from db.connection import SessionLocal
from db import crud
from System_Messages.functions_list_prompt import (
    FUNCTIONS_LIST_PROMPT,
    FUNCTIONS_LIST_CHAT_PROMPT,
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
    sources: list,
    source_mode: str,
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
    if sources:
        parts += ["", f"=== WEB RESEARCH ({len(sources)} sources — competitor features + tool capabilities) ==="]
        for s in sources:
            parts.append(f"[{s['url']}]\n{s['content'][:600]}")
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {source_mode} ===",
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
    start = time.time()

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

    # ── 1. Search ────────────────────────────────────────────────────────────
    sources: list = []
    source_mode   = "synthesis_only"

    if SERPER_API_KEY:
        queries = _build_search_queries(idea, competition or {}, business_model or {}, region)
        sources = gather_sources(queries, SERPER_API_KEY, max_sources=10)
        source_mode = "web_sourced" if sources else "synthesis_only"
    else:
        log.warning("SERPER_API_KEY not set — building functions from prior pipeline data only")

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    sources = truncate_sources(sources)
    context = _build_analysis_context(
        idea, problems,
        customers or {}, competition or {},
        strategy or {}, business_model or {},
        region, sources, source_mode,
    )

    user_content = context
    if custom_prompt:
        user_content += f"\n\n=== ADDITIONAL INSTRUCTION ===\n{custom_prompt}"

    # ── 3. LLM call ──────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": FUNCTIONS_LIST_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
        max_tokens=4000,
    )

    raw    = response.choices[0].message.content
    result = parse_llm_json(raw)
    result["source_mode"]  = source_mode
    result["sources_used"] = len(sources)

    # ── 4. Persist ───────────────────────────────────────────────────────────
    elapsed_ms = int((time.time() - start) * 1000)

    db = SessionLocal()
    try:
        crud.save_functions_list(db, user_id, result)
        crud.save_agent_run(
            db,
            user_id=user_id,
            agent_name="NineFunctionsList",
            input_data={
                "idea_snippet":    idea[:300],
                "region":          region,
                "has_customers":   customers is not None,
                "has_competition": competition is not None,
                "has_strategy":    strategy is not None,
                "has_biz_model":   business_model is not None,
                "source_mode":     source_mode,
                "sources_used":    len(sources),
            },
            output_data=result,
            status="done",
            execution_time_ms=elapsed_ms,
        )
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
