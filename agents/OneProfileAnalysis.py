"""
agents/OneProfileAnalysis.py
=============================
Pipeline Step 1 — Founder Profile Analysis.

Input  : questionnaire (user_profile + career_profile) + skills list
Output : founder profile JSON — personality insights, recommended industries,
         problem spaces, and search keywords for the next agent.

Called by
---------
  orchestrator/orchestrator.py  → run_new_user_pipeline()
  agents/generalBot.py          → _run_new_user_pipeline_inline()

The output feeds directly into TwoProblemDiscovery.run_problem_discovery().
"""

import json
import logging

from agents.utils import parse_llm_json
from agents.config import client, GROQ_MODEL

log = logging.getLogger(__name__)


def run_profile_analysis(questionnaire: dict, skills: list) -> dict:
    """
    Analyse the founder's questionnaire and skills to produce a structured
    profile used by all downstream agents.

    Key guarantees
    --------------
    - keyword_seeds are always present in the output regardless of LLM behaviour
      (merged after the LLM call so the next agent always has search queries)
    - Marketplace detection steers keywords toward consumer-facing problems
    - Region and curiosity domain are injected into keyword seeds explicitly
    """
    u = questionnaire.get("user_profile", {})
    curiosity_domain   = u.get("curiosity_domain", "")
    business_interests = u.get("business_interests", [])
    target_region      = u.get("target_region", "")
    is_marketplace     = any("marketplace" in b.lower() for b in business_interests)

    # Build deterministic keyword seeds so the LLM must expand them, not replace them
    keyword_seeds = []
    if curiosity_domain:
        if is_marketplace:
            keyword_seeds.append(f"{curiosity_domain} marketplace")
            keyword_seeds.append(f"{curiosity_domain} online marketplace")
            if target_region and target_region.lower() != "global":
                keyword_seeds.append(f"{curiosity_domain} marketplace {target_region}")
        else:
            keyword_seeds.append(f"{curiosity_domain} e-commerce")
            keyword_seeds.append(f"{curiosity_domain} online store")
    for bi in business_interests:
        keyword_seeds.append(f"{curiosity_domain} {bi}".strip())

    combined = {
        "questionnaire":  questionnaire,
        "skills":         {"skills": skills},
        "keyword_seeds":  keyword_seeds,
    }

    prompt = f"""
You are a senior startup advisor and venture builder.
Analyze this founder and determine what kind of business they can realistically build.

=== KEYWORD GENERATION RULES (STRICT) ===
You are given `keyword_seeds` — these are NON-NEGOTIABLE starting points.
You MUST include ALL keyword_seeds in your `search_direction.keywords` output.
You may ADD 2-4 more specific variants, but NEVER remove or ignore the seeds.

The seeds were derived from:
  curiosity_domain   = "{curiosity_domain}"   ← this is the NICHE/ANGLE
  business_interests = {json.dumps(business_interests)}  ← this is the MODEL
  target_region      = "{target_region}"

Keywords MUST be domain-specific search queries, e.g.:
  "Art & Design marketplace pain points"
  "handmade art online marketplace problems"
  "digital art platform buyer frustrations"
NOT generic: "E-commerce Marketplace", "Global Digital Marketplace"

=== EXECUTION RULES ===
- skills = {json.dumps(skills)} → if empty, operator/no-code models ONLY
- Founder setup: {u.get("founder_setup", "")}
- If "Marketplace" in business_interests → recommended_industries must be consumer-facing marketplace
- Avoid recommending industries that require skills the user does not have

INPUT:
{json.dumps(combined, indent=2)}

Return ONLY valid JSON:
{{
  "personality_insights": {{"type":"","motivation":"","traits":[],"strengths":[],"weaknesses":[]}},
  "founder_profile": {{"experience_level":"","execution_style":"","risk_level":"","readiness":"","skill_level_summary":"","key_skill_gaps":[]}},
  "recommended_industries": [],
  "recommended_problem_spaces": [],
  "search_direction": {{"keywords": []}},
  "system_flags": {{"needs_guidance": true, "should_suggest_learning": true}}
}}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation. No markdown."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.5,
        max_tokens=2000,
    )

    try:
        result = parse_llm_json(response.choices[0].message.content)
    except ValueError as e:
        log.error("[ProfileAnalysis] JSON parse failed: %s", e)
        raise

    # Guarantee seeds are always in the output regardless of what the LLM did
    existing_kw = result.get("search_direction", {}).get("keywords", [])
    merged = list(dict.fromkeys(keyword_seeds + existing_kw))   # seeds first, deduped
    result.setdefault("search_direction", {})["keywords"] = merged[:10]

    return result
