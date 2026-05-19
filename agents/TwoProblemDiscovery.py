"""
agents/TwoProblemDiscovery.py
==============================
Pipeline Step 2 — Problem Discovery.

Input  : founder profile (from OneProfileAnalysis) + questionnaire
Output : validated problems JSON — real consumer pain points with sources,
         validation scores, customer segments, and personas.

Called by
---------
  orchestrator/orchestrator.py  → run_new_user_pipeline()
  orchestrator/orchestrator.py  → run_returning_user_pipeline()
  agents/generalBot.py          → _run_new_user_pipeline_inline()

Search strategy
---------------
Uses Tavily (domain-filtered to Reddit, Quora, Trustpilot — where real users
complain) + Groq LLM extraction to pull structured pain points from each source.
Falls back to Serper + BeautifulSoup if Tavily is unavailable.
"""

import json
import logging

from agents.utils import parse_llm_json
from agents.search_pipeline import run_search_pipeline
from agents.config import (
    client, GROQ_MODEL,
    SERPER_API_KEY, TAVILY_API_KEY, GROQ_EXTRACTION_MODEL,
)

log = logging.getLogger(__name__)

# Platforms where real users voice frustrations and pain points
_SEARCH_DOMAINS = [
    "reddit.com", "quora.com", "trustpilot.com",
    "medium.com", "producthunt.com", "getapp.com",
]

_EXTRACTION_SCHEMA = {
    "pain_points":       "specific problems or complaints customers express about this space",
    "frustrations":      "what users find broken or frustrating about current solutions",
    "unmet_needs":       "things users wish existed but can't find",
    "current_solutions": "what people currently use to solve this problem",
    "complaints":        "specific negative experiences mentioned by users",
    "frequency":         "how often this problem occurs (daily, weekly, occasionally)",
}


def _build_queries(keywords: list, curiosity_domain: str, region_mod: str,
                   is_marketplace: bool) -> list[str]:
    """Build targeted search queries from the profile's keyword seeds."""
    if is_marketplace and curiosity_domain:
        templates = [
            "{k} buyer problems {r}",
            "{k} consumer frustrations buying online {r}",
            "{k} marketplace trust issues {r}",
            "{k} online shopping pain points {r}",
            "{k} community marketplace challenges {r}",
        ]
    elif is_marketplace:
        templates = [
            "{k} marketplace buyer problems {r}",
            "{k} consumer trust online marketplace {r}",
            "{k} online marketplace pain points {r}",
            "{k} marketplace challenges {r}",
        ]
    else:
        templates = [
            "{k} problems small business {r}",
            "{k} user pain points {r}",
            "{k} challenges startup {r}",
            "{k} complaints reddit {r}",
        ]

    queries = []
    for k in keywords:
        for t in templates:
            queries.append(t.format(k=k, r=region_mod).strip())
            if len(queries) >= 20:
                return queries
    return queries


def run_problem_discovery(profile: dict, questionnaire: dict) -> dict:
    """
    Search the web for real consumer problems and extract structured pain points.

    The LLM receives ranked, extracted source data (not raw scraped text) so
    the problems it generates are grounded in real evidence.
    """
    u                  = questionnaire.get("user_profile", {})
    business_interests = u.get("business_interests", [])
    target_region      = u.get("target_region", "")
    curiosity_domain   = u.get("curiosity_domain", "")
    region_mod         = {"MENA": "MENA Middle East", "Egypt": "Egypt",
                          "Global": ""}.get(target_region, target_region)
    is_marketplace     = any("marketplace" in b.lower() for b in business_interests)
    keywords           = profile.get("search_direction", {}).get("keywords", [])

    queries = _build_queries(keywords, curiosity_domain, region_mod, is_marketplace)

    search = run_search_pipeline(
        queries=queries,
        tavily_api_key=TAVILY_API_KEY,
        extraction_schema=_EXTRACTION_SCHEMA,
        keywords=keywords + [curiosity_domain, target_region, "problems", "pain points", "complaints"],
        include_domains=_SEARCH_DOMAINS,
        groq_client=client,
        extraction_model=GROQ_EXTRACTION_MODEL,
        serper_fallback_key=SERPER_API_KEY,
        max_sources=10,
    )

    # Explicit B2B guard for marketplace founders — prevents seller-tooling problems
    b2b_guard = ""
    if is_marketplace:
        b2b_guard = f"""
=== FORBIDDEN PROBLEM TYPES (DO NOT GENERATE) ===
❌ "Sellers struggle with X" — the founder is NOT building seller tools
❌ "Marketplace sellers face X" — wrong customer
❌ Any B2B SaaS, inventory management, pricing tool, or logistics problem
❌ Problems where the TARGET CUSTOMER is a business or seller

✅ REQUIRED: Problems where the target customer is a CONSUMER or BUYER
✅ The curiosity_domain is "{curiosity_domain}" — all problems MUST relate to this niche
✅ Example correct: "Art buyers struggle to discover authentic handmade art online"
✅ Example correct: "Design enthusiasts can't find a trusted global marketplace for indie creators"
"""

    profile_summary = {
        "curiosity_domain":           curiosity_domain,
        "recommended_industries":     profile.get("recommended_industries", []),
        "recommended_problem_spaces": profile.get("recommended_problem_spaces", []),
        "founder_strengths":          profile.get("personality_insights", {}).get("strengths", []),
        "key_skill_gaps":             profile.get("founder_profile", {}).get("key_skill_gaps", []),
        "target_region":              target_region,
        "business_type_interest":     business_interests,
        "experience_level":           u.get("experience_level", ""),
        "risk_tolerance":             u.get("risk_tolerance", ""),
        "founder_setup":              u.get("founder_setup", ""),
        "career_profile":             questionnaire.get("career_profile", {}),
    }

    source_context = (
        search.to_prompt_context()
        if search.sources
        else "No web sources — use profile to derive realistic consumer problems."
    )

    prompt = f"""
You are a startup problem discovery AI. Extract REAL, SPECIFIC customer problems.

=== CONTEXT ===
Founder's curiosity domain : {curiosity_domain}
Business model interest    : {business_interests}
Target region              : {target_region}
{b2b_guard}

=== RULES ===
1. Title = specific problem sentence from the CONSUMER's perspective
2. All problems MUST relate to "{curiosity_domain}" niche
3. Region = "{target_region}" — ground problems here
4. Extract at least 4 problems
5. validation_score = min(85, sources*25 + evidence*15)
6. customer_segments: 3-5 specific CONSUMER types (not seller types)
{"Source mode: web_sourced. Use provided sources." if search.source_mode == "web_sourced" else "Source mode: profile_derived. Validation score = 35."}

Founder Profile:
{json.dumps(profile_summary, indent=2)}

{source_context}

Return ONLY valid JSON:
{{"problems":[{{"id":"P1","title":"","description":"","industry":"","target_customer":"",
"pain_level":"high|medium|low","frequency":"high|medium|low","current_solutions":"",
"gap_opportunity":"","source_type":"{search.source_mode}","sources":[],"evidence":[],"validation_score":0}}],
"customer_segments":[],"personas":[],"summary_insight":""}}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation. No markdown fences."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.5,
        max_tokens=8000,
    )

    try:
        result = parse_llm_json(response.choices[0].message.content)
    except ValueError as e:
        log.error("[ProblemDiscovery] JSON parse failed: %s", e)
        raise

    for p in result.get("problems", []):
        if p.get("source_type") == "profile_derived":
            p["validation_score"] = 35
        else:
            p["validation_score"] = min(
                85,
                len(p.get("sources", [])) * 25 + len(p.get("evidence", [])) * 15,
            )

    sources_used, sources_list = search.to_sources_meta()
    result["source_mode"]  = search.source_mode
    result["sources_used"] = sources_used
    result["sources_list"] = sources_list
    return result
