"""
agents/TwelveGoToMarket.py
===========================
Pipeline step 12 — Go-To-Market Plan. Final agent.

Inputs  : idea + problems + customers + competition + market_potential +
          strategy + business_model + mvp_planning + unit_economics (all from DB)
Outputs : target_launch_segment, positioning_message, marketing_channels,
          funnel_stages, launch_experiments, first_100_customers_plan,
          launch_timeline, success_metrics, cac_tracking,
          feedback_loops, summary

Search strategy:
  GTM requires real-world channel benchmarks and launch playbooks.
  Searches target: similar product launch stories (Indie Hackers, Reddit),
  MENA digital marketing benchmarks, Instagram/WhatsApp acquisition tactics,
  channel CPC/CPM benchmarks, and successful no-code marketplace GTM examples.

Also exposes a section-scoped refinement chat and a streaming chat variant.

DB flow:
  run_go_to_market() → saves to go_to_market_results + agent_runs
  chat_go_to_market() → stateless; history managed by the caller
"""

import json
import logging
import time

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline, SearchResults
from agents.schemas import validate_section_output
from agents.config import client, GROQ_MODEL, SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL

_SEARCH_DOMAINS = [
    "indiehackers.com", "producthunt.com", "ycombinator.com",
    "firstround.com", "a16z.com", "techcrunch.com", "growthhackers.com",
]

_EXTRACTION_SCHEMA = {
    "acquisition_channels": "most effective marketing and sales channels for this type of product",
    "channel_costs": "CAC or cost per lead for specific channels (e.g. Instagram CPL, Google CPC)",
    "launch_strategies": "how similar companies launched successfully (Product Hunt, communities, etc.)",
    "growth_tactics": "specific tactics that drove early user growth",
    "region_channels": "channels that work specifically in MENA, Egypt, or the target region",
    "partnership_types": "strategic partnerships or distribution deals that accelerated growth",
}
from db.connection import SessionLocal
from db import crud
from System_Messages.go_to_market_prompt import (
    GO_TO_MARKET_PROMPT,
    GO_TO_MARKET_CHAT_PROMPT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_search_queries(
    idea: str,
    customers: dict,
    competition: dict,
    business_model: dict,
    region: str,
) -> list[str]:
    """
    Build GTM-focused queries targeting:
    - Similar product launch playbooks (Indie Hackers, Reddit, Product Hunt)
    - MENA digital marketing channel benchmarks
    - WhatsApp/Instagram acquisition tactics for this industry
    - Competitor marketing and acquisition strategies
    - First 100 users tactics for this business model type
    """
    region_mod  = region.strip() if region and region.lower() != "global" else ""
    idea_seed   = " ".join(idea.split()[:6]) if idea else ""
    bm_type     = business_model.get("business_model_type", "")

    primary_channel = ""
    if customers:
        channels = customers.get("acquisition_channels", [])
        primary_channel = channels[0] if channels else ""

    competitor_names = [
        c.get("name", "")
        for c in competition.get("direct_competitors", [])[:2]
        if c.get("name")
    ]

    queries: list[str] = [
        f"{idea_seed} how to get first 100 customers {region_mod}".strip(),
        f"{bm_type} go to market strategy launch playbook".strip(),
        f"{idea_seed} launch story indie hackers reddit {region_mod}".strip(),
        f"Instagram marketing {idea_seed} {region_mod} strategy".strip(),
        f"WhatsApp marketing {bm_type} acquisition {region_mod}".strip(),
        f"{region_mod} digital marketing benchmarks CPC CPM 2024".strip(),
        f"{idea_seed} first users community-led growth {region_mod}".strip(),
        f"{bm_type} {region_mod} customer acquisition channel comparison".strip(),
    ]

    for name in competitor_names:
        queries.append(f"{name} marketing strategy how they got first customers")

    if primary_channel:
        queries.append(f"{primary_channel} customer acquisition tactics {region_mod}".strip())

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
    business_model: dict,
    mvp_planning: dict,
    unit_economics: dict,
    region: str,
    search: SearchResults,
) -> str:
    parts = [
        "=== SAVED IDEA ===",
        idea.strip(),
        f"Region: {region or 'Global'}",
        "",
    ]

    # Primary customer — defines targeting and messaging
    if customers:
        primary = customers.get("primary_segment", {})
        pid     = primary.get("id", "")
        parts += ["=== PRIMARY CUSTOMER (launch target) ==="]
        for seg in customers.get("customer_segments", []):
            if seg.get("id") == pid:
                parts.append(
                    f"  {seg.get('name','')} — {seg.get('description','')[:120]}"
                )
                parts.append(f"  Where to find: {', '.join(seg.get('where_to_find',[])[:4])}")
                parts.append(f"  Pain: {seg.get('why_they_care','')[:120]}")
                break
        ea = customers.get("early_adopter_profile", "")
        if ea:
            parts.append(f"  Early adopter: {ea}")
        ch = customers.get("acquisition_channels", [])
        if ch:
            parts.append(f"  Known channels: {', '.join(ch[:5])}")

    # Positioning from strategy — shapes messaging
    if strategy:
        vp = strategy.get("value_proposition", {})
        parts += [
            "",
            "=== POSITIONING INPUTS ===",
            f"  Statement    : {vp.get('statement','')}",
            f"  Core promise : {strategy.get('core_promise','')}",
            f"  Differentiator: {vp.get('differentiator','')}",
            f"  Pos. statement: {strategy.get('positioning',{}).get('positioning_statement','')}",
        ]
        ua = strategy.get("unfair_advantages", [])
        for a in ua[:3]:
            parts.append(f"  Unfair advantage: {a}")

    # Competition — shapes channel choices and differentiation
    if competition:
        parts += ["", "=== HOW COMPETITORS ACQUIRE CUSTOMERS ==="]
        for c in competition.get("direct_competitors", [])[:3]:
            parts.append(
                f"  {c.get('name','')} — pricing: {c.get('pricing_model','?')}, "
                f"target: {c.get('target_customer','?')[:80]}"
            )
        diff_opps = competition.get("differentiation_opportunities", [])
        for opp in diff_opps[:2]:
            parts.append(f"  Differentiation: {opp}")

    # Business model — shapes monetisation triggers in GTM
    if business_model:
        bm_type = business_model.get("business_model_type", "")
        pricing = business_model.get("pricing_strategy", {})
        parts += [
            "",
            "=== BUSINESS MODEL GTM CONSTRAINTS ===",
            f"  Model type : {bm_type}",
            f"  Commission : {pricing.get('price_points',{}).get('commission_rate','')}",
            f"  Sensitivity: {pricing.get('price_sensitivity_note','')}",
        ]

    # MVP plan — directly feeds launch timeline
    if mvp_planning:
        parts += ["", "=== MVP LAUNCH TARGETS ==="]
        phases = mvp_planning.get("build_plan", {}).get("phases", [])
        for ph in phases[:3]:
            parts.append(f"  Phase {ph.get('phase')}: {ph.get('milestone','')}")
        u100 = mvp_planning.get("first_100_users_plan", "")
        if u100:
            parts.append(f"  First 100 plan: {u100[:250]}")
        lc = mvp_planning.get("launch_criteria", {})
        sm = lc.get("success_metrics", [])
        for m in sm[:3]:
            parts.append(f"  Launch success metric: {m}")

    # Unit economics — CAC targets come from here
    if unit_economics:
        cac_analysis = unit_economics.get("cac_analysis", {})
        parts += [
            "",
            "=== CAC TARGETS FROM UNIT ECONOMICS ===",
            f"  Blended CAC target: ${unit_economics.get('cost_assumptions',{}).get('avg_blended_cac_usd','?')}",
            f"  Organic CAC: ${cac_analysis.get('organic_cac_usd','?')}",
            f"  Paid CAC: ${cac_analysis.get('paid_cac_usd','?')}",
            f"  LTV: ${unit_economics.get('ltv_analysis',{}).get('ltv_usd','?')}",
            f"  LTV/CAC viable ratio: {unit_economics.get('ltv_cac_ratio',{}).get('target_ratio','3')}x",
        ]

    # Market potential — sets ambition of targets
    if market_potential:
        som = market_potential.get("som", {})
        parts += [
            "",
            "=== MARKET SCALE ===",
            f"  SOM: {som.get('value','')} {som.get('unit','')} over {som.get('timeline','')}",
        ]

    # Top problems — fuel messaging
    parts += ["", "=== TOP PAIN POINTS (inform messaging) ==="]
    for p in problems.get("problems", [])[:3]:
        parts.append(f"  [{p.get('id','?')}] {p.get('title','')}")

    web_context = search.to_prompt_context()
    if web_context:
        parts += ["", web_context]
    else:
        parts += [
            "",
            f"=== SOURCE MODE: {search.source_mode} ===",
            "No web sources. Base GTM plan on customer, strategy, and business model data above.",
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def run_go_to_market(
    user_id: str,
    idea: str,
    problems: dict,
    customers: dict = None,
    competition: dict = None,
    market_potential: dict = None,
    strategy: dict = None,
    business_model: dict = None,
    mvp_planning: dict = None,
    unit_economics: dict = None,
    custom_prompt: str = None,
) -> dict:
    """
    Generate the go-to-market plan for the saved idea.

    Steps:
      1. Extract region, channels, CAC targets from prior pipeline data
      2. Search for launch playbooks, channel benchmarks, acquisition tactics
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
    business_model   : dict — EightBusinessModel output (optional)
    mvp_planning     : dict — TenMVPPlanning output (optional)
    unit_economics   : dict — ElevenUnitEconomicsAgent output (optional)
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

    # ── 1. Search + extract ──────────────────────────────────────────────────
    search = run_search_pipeline(
        queries=_build_search_queries(idea, customers or {}, competition or {}, business_model or {}, region),
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=[idea.split()[0] if idea else "", region, "acquisition", "launch", "growth", "marketing"],
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
        business_model or {}, mvp_planning or {},
        unit_economics or {},
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
                {"role": "system", "content": GO_TO_MARKET_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.4,
            max_tokens=4500,
        )
    except Exception as e:
        log.error("[TwelveGoToMarket] LLM call failed: %s", e)
        raise

    raw = response.choices[0].message.content
    try:
        result = validate_section_output("go_to_market", parse_llm_json(raw))
    except ValueError as e:
        log.error("[TwelveGoToMarket] JSON parse failed: %s", e)
        raise
    sources_used, sources_list = search.to_sources_meta()
    result["source_mode"]  = search.source_mode
    result["sources_used"] = sources_used
    result["sources_list"] = sources_list

    # ── 4. Persist ───────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        crud.save_go_to_market(db, user_id, result)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section refinement chat  (stateless — caller owns history)
# ─────────────────────────────────────────────────────────────────────────────

def chat_go_to_market(
    current_analysis: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Refine the go-to-market plan through conversation.
    Scoped to this section only — cannot modify any other pipeline data.

    Returns
    -------
    str — assistant reply (may contain a ```json block with updated sections)
    """
    context = (
        "=== CURRENT GO-TO-MARKET PLAN ===\n"
        + json.dumps(current_analysis, indent=2)
    )

    messages = [
        {"role": "system", "content": GO_TO_MARKET_CHAT_PROMPT},
        {"role": "system", "content": context},
        *history[-20:],
        {"role": "user",   "content": user_message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1400,
    )

    return response.choices[0].message.content.strip()
