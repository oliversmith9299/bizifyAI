"""
agents/PipelineRunner.py
========================
Background task — runs all 3 AI pipeline steps and saves to Supabase.
Called by routes/main.py as a FastAPI BackgroundTask.

FIXES APPLIED:
  1. Profile Analysis: curiosity_domain is now a PRIMARY keyword driver,
     not a secondary shaper. Keywords are always domain-specific + region.
  2. Problem Discovery: problems now match the founder's curiosity domain
     (Art & Design), target CONSUMER pain points (not seller/B2B tooling),
     and inject the full user_profile so the LLM has real context.
  3. _build_context: risk constraint now checks for actual user phrase,
     co-founder setup no longer wrongly triggers solo-only rule,
     curiosity_domain and career_profile injected for richer idea context.
  4. Idea generation opening prompt references Art & Design + co-founder
     explicitly so the first idea is domain-matched.
"""

import json
import logging
import os
from typing import Dict, List

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import requests as req

load_dotenv()

log = logging.getLogger("pipeline_runner")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# In-memory chat sessions per user
# user_id → {"context": str, "history": list}
_user_sessions: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: Profile Analysis
# ─────────────────────────────────────────────────────────────────────────────
def run_profile_analysis(questionnaire: dict, skills: list) -> dict:
    u = questionnaire.get("user_profile", {})
    curiosity_domain = u.get("curiosity_domain", "")
    business_interests = u.get("business_interests", [])
    target_region = u.get("target_region", "")
    is_marketplace = any("marketplace" in b.lower() for b in business_interests)

    # FIX 1: Build deterministic keyword seeds so the LLM can't ignore them
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
        "questionnaire": questionnaire,
        "skills": {"skills": skills},
        "keyword_seeds": keyword_seeds,  # hard-injected so the LLM expands them, not replaces them
    }

    prompt = f"""
You are a senior startup advisor and venture builder.
Analyze this user and determine what kind of business they can realistically build.

=== KEYWORD GENERATION RULES (STRICT) ===
You are given `keyword_seeds` — these are NON-NEGOTIABLE starting points.
You MUST include ALL keyword_seeds in your `search_direction.keywords` output.
You may ADD 2-4 more specific variants, but NEVER remove or ignore the seeds.

The seeds were derived from:
  curiosity_domain  = "{curiosity_domain}"   ← this is the NICHE/ANGLE
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
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation. No markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,  # lower = more deterministic, less likely to ignore instructions
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())

    # FIX 1b: Guarantee seeds are always present regardless of what the LLM did
    existing_kw = result.get("search_direction", {}).get("keywords", [])
    merged = list(dict.fromkeys(keyword_seeds + existing_kw))  # seeds first, deduped
    result.setdefault("search_direction", {})["keywords"] = merged[:10]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: Problem Discovery
# ─────────────────────────────────────────────────────────────────────────────
def _search_serper(query: str) -> dict:
    if not SERPER_API_KEY:
        return {}
    try:
        res = req.post(
            "https://google.serper.dev/search",
            json={"q": query},
            headers={"X-API-KEY": SERPER_API_KEY},
            timeout=10
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        log.warning(f"Search failed: {e}")
        return {}


def _fetch_page(url: str, snippet: str = "") -> str:
    try:
        r = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        )
        return text[:800]
    except Exception:
        return snippet[:800]


def run_problem_discovery(profile: dict, questionnaire: dict) -> dict:
    u = questionnaire.get("user_profile", {})
    business_interests = u.get("business_interests", [])
    target_region = u.get("target_region", "")
    curiosity_domain = u.get("curiosity_domain", "")  # FIX 2: extracted here
    region_mod = {"MENA": "MENA Middle East", "Egypt": "Egypt", "Global": ""}.get(target_region, target_region)
    is_marketplace = any("marketplace" in b.lower() for b in business_interests)
    keywords = profile.get("search_direction", {}).get("keywords", [])

    # FIX 2: Search templates now target CONSUMER pain points in the curiosity domain,
    # not generic seller/B2B tooling problems
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
            q = t.format(k=k, r=region_mod).strip()
            queries.append(q)
            if len(queries) >= 20:
                break
        if len(queries) >= 20:
            break

    seen, all_sources = set(), []
    for q in queries:
        for r in _search_serper(q).get("organic", []):
            url = r.get("link", "")
            if url and url not in seen:
                seen.add(url)
                all_sources.append({
                    "title": r.get("title", ""), "url": url,
                    "snippet": r.get("snippet", ""),
                    "type": "reddit" if "reddit.com" in url else "web"
                })

    enriched = []
    for s in all_sources[:20]:
        content = _fetch_page(s["url"], s.get("snippet", ""))
        if len(content) > 100:
            enriched.append({"title": s["title"], "url": s["url"], "content": content})
    enriched = enriched[:10]
    source_mode = "web_sourced" if enriched else "profile_derived"

    # FIX 2b: Full user profile injected so LLM can't generate off-topic problems
    profile_summary = {
        "curiosity_domain": curiosity_domain,
        "recommended_industries": profile.get("recommended_industries", []),
        "recommended_problem_spaces": profile.get("recommended_problem_spaces", []),
        "founder_strengths": profile.get("personality_insights", {}).get("strengths", []),
        "key_skill_gaps": profile.get("founder_profile", {}).get("key_skill_gaps", []),
        "target_region": target_region,
        "business_type_interest": business_interests,
        "experience_level": u.get("experience_level", ""),
        "risk_tolerance": u.get("risk_tolerance", ""),
        "founder_setup": u.get("founder_setup", ""),
        "career_profile": questionnaire.get("career_profile", {}),
    }

    # FIX 2c: Prompt explicitly forbids B2B seller-tooling problems
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
✅ Example correct problem: "Art buyers struggle to discover authentic handmade art online"
✅ Example correct problem: "Design enthusiasts can't find a trusted global marketplace for indie creators"
"""

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
{"Source mode: web_sourced. Use provided sources." if source_mode == "web_sourced" else "Source mode: profile_derived. Validation score = 35."}

Founder Profile:
{json.dumps(profile_summary, indent=2)}

{"Sources:\n" + json.dumps(enriched, indent=2) if enriched else "No web sources — use profile to derive realistic consumer problems."}

Return ONLY valid JSON:
{{"problems":[{{"id":"P1","title":"","description":"","industry":"","target_customer":"",
"pain_level":"high|medium|low","frequency":"high|medium|low","current_solutions":"",
"gap_opportunity":"","source_type":"{source_mode}","sources":[],"evidence":[],"validation_score":0}}],
"customer_segments":[],"personas":[],"summary_insight":""}}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation. No markdown fences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=8000,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]

    result = json.loads(raw.strip())
    for p in result.get("problems", []):
        if p.get("source_type") == "profile_derived":
            p["validation_score"] = 35
        else:
            p["validation_score"] = min(85, len(p.get("sources", [])) * 25 + len(p.get("evidence", [])) * 15)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: Idea Chat
# ─────────────────────────────────────────────────────────────────────────────
IDEA_SYSTEM_PROMPT = """
You are a sharp, practical startup advisor helping a founder find their best startup idea.

RULES (non-negotiable — read before every response):
1. Read HARD EXECUTION CONSTRAINTS in context first.
2. "Marketplace" = B2C platform where CONSUMERS buy. NOT B2B tools.
3. No coding skills = no-code tools only (Sharetribe, Notion, WhatsApp, Webflow, Airtable).
4. Ideas MUST match the founder's curiosity domain — it is the core niche.
5. ANTI-LOOP: If user wants something different → change BOTH the problem AND the business model.

FORMAT every idea exactly like this — no exceptions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 IDEA: [Specific Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem it solves : [consumer problem in the founder's curiosity domain]
Target customer   : [specific consumer type, e.g. "Art lovers globally buying handmade pieces"]
How it works      : [2-3 sentences — what the founder does day-to-day, no software building]
Launch stack      : [exact no-code tools, e.g. "Sharetribe for marketplace, WhatsApp for seller onboarding"]
Business model    : [commission %, listing fee, or subscription — be specific]
Why you can do it : [tied to their actual strengths — creativity, ops, co-founder leverage]
First 7-day test  : [one concrete WhatsApp/DM/form action to validate with real people]
Startup cost      : [realistic USD estimate — no-code should be under $500]
Risk level        : Low / Medium / High — [one sentence why]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

When user seems satisfied → remind them: Type 'save' to save this idea.
Max 400 words per response unless user asks for detail.
"""


def _build_context(profile: dict, problems: dict, questionnaire: dict, skills: list) -> str:
    u = questionnaire.get("user_profile", {})
    career = questionnaire.get("career_profile", {})

    has_tech = any(s.lower() in ["python", "machine learning", "apis", "backend development",
                                  "software development", "coding"] for s in skills)
    bi = u.get("business_interests", [])
    region = u.get("target_region", "")
    setup = u.get("founder_setup", "")
    risk = u.get("risk_tolerance", "")
    curiosity_domain = u.get("curiosity_domain", "")

    # FIX 3a: Constraints derived from actual field values, not fragile string checks
    is_solo = "solo" in setup.lower()
    is_cofounder = "co-founder" in setup.lower() or "partner" in setup.lower()
    is_marketplace = "marketplace" in " ".join(bi).lower()
    # FIX 3b: risk constraint matches actual user phrasing
    low_capital = "low" in risk.lower() or "minimal" in risk.lower()

    c = ["=== ⚠️ HARD EXECUTION CONSTRAINTS ==="]
    if not has_tech:
        c += [
            "❌ NO SOFTWARE BUILDING — founder has no coding skills",
            "✅ ALLOWED TOOLS: Sharetribe, Webflow, Notion, Airtable, WhatsApp Business, Typeform",
        ]
    if is_solo:
        c.append("❌ SOLO FOUNDER — no team, laptop-only operations")
    if is_cofounder:
        c.append("✅ CO-FOUNDER SETUP — can split ops, creative, and biz dev roles")
    if low_capital:
        c.append("❌ LOW CAPITAL — keep launch cost under $300")
    else:
        c.append("✅ MODERATE RISK TOLERANCE — calculated risks ok, keep launch under $500")
    if is_marketplace:
        c += [
            "✅ B2C MARKETPLACE ONLY — end user is a CONSUMER, not a business",
            "❌ FORBIDDEN: B2B SaaS, seller tools, logistics consulting, inventory management",
            "❌ FORBIDDEN: Any idea where the paying customer is a merchant/seller/business",
        ]
    c.append(f"✅ TARGET REGION: {region}")

    # FIX 3c: Inject curiosity domain prominently — it's the product niche
    if curiosity_domain:
        c.append(f"✅ FOUNDER NICHE / CURIOSITY DOMAIN: {curiosity_domain}")
        c.append(f"   → All ideas MUST operate in or around the {curiosity_domain} space")

    # FIX 3d: Inject career profile signals to enrich idea generation
    if career:
        desired_impact = career.get("desired_impact", [])
        work_types = career.get("preferred_work_types", [])
        if desired_impact:
            c.append(f"✅ FOUNDER WANTS TO: {', '.join(desired_impact)}")
        if work_types:
            c.append(f"✅ WORKS BEST WITH: {', '.join(work_types)}")

    p = ["=== VALIDATED PROBLEMS (use these as idea sources) ==="]
    for prob in problems.get("problems", []):
        if prob.get("validation_score", 0) >= 35:
            p.append(f"[{prob['id']}] {prob['title']}")
            p.append(f"  Target customer  : {prob.get('target_customer', '')}")
            p.append(f"  Gap opportunity  : {prob.get('gap_opportunity', '')}")

    return "\n".join(c) + "\n\n" + "\n".join(p)


def get_user_context(user_id: str) -> str:
    """Get stored context for a user (used by /pipeline/chat route)."""
    return _user_sessions.get(user_id, {}).get("context", "")


def chat_with_idea_agent(user_id: str, message: str, context: str) -> str:
    if user_id not in _user_sessions:
        _user_sessions[user_id] = {"context": context, "history": []}

    session = _user_sessions[user_id]
    session["history"].append({"role": "user", "content": message})

    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        *session["history"][-20:],
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1000,
    )
    reply = response.choices[0].message.content.strip()
    session["history"].append({"role": "assistant", "content": reply})
    return reply


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Full Pipeline Background Task
# ─────────────────────────────────────────────────────────────────────────────
async def run_full_pipeline(user_id: str, questionnaire: dict, skills: list):
    """
    Runs all 3 steps and saves each result to Supabase DB.
    Called as a FastAPI BackgroundTask from routes/main.py.
    """
    from db import crud
    from db.connection import get_session

    u = questionnaire.get("user_profile", {})
    curiosity_domain = u.get("curiosity_domain", "")
    founder_setup = u.get("founder_setup", "")
    business_interests = u.get("business_interests", [])

    try:
        # ── Step 1: Profile Analysis ─────────────────────────────────────────
        with get_session() as db:
            crud.upsert_pipeline_status(db, user_id, "running", "profile_analysis")

        log.info(f"[{user_id}] Running ProfileAnalysis...")
        profile = run_profile_analysis(questionnaire, skills)

        with get_session() as db:
            crud.save_profile(db, user_id, profile)
        log.info(f"[{user_id}] ✅ ProfileAnalysis saved to DB")

        # ── Step 2: Problem Discovery ────────────────────────────────────────
        with get_session() as db:
            crud.upsert_pipeline_status(db, user_id, "running", "problem_discovery")

        log.info(f"[{user_id}] Running ProblemDiscovery...")
        problems = run_problem_discovery(profile, questionnaire)

        with get_session() as db:
            crud.save_problems(db, user_id, problems)
        log.info(f"[{user_id}] ✅ Problems saved ({len(problems.get('problems', []))} found)")

        # ── Step 3: Generate opening idea ────────────────────────────────────
        with get_session() as db:
            crud.upsert_pipeline_status(db, user_id, "running", "idea_generation")

        log.info(f"[{user_id}] Generating opening idea...")
        context = _build_context(profile, problems, questionnaire, skills)
        _user_sessions[user_id] = {"context": context, "history": []}

        # FIX 4: Opening prompt is domain-specific and references the actual user signals
        is_marketplace = any("marketplace" in b.lower() for b in business_interests)
        model_type = "B2C marketplace" if is_marketplace else "e-commerce"
        setup_note = (
            "They have a co-founder, so can split creative and ops roles."
            if "co-founder" in founder_setup.lower() or "partner" in founder_setup.lower()
            else "They are a solo founder."
        )
        domain_note = f"Their curiosity domain is {curiosity_domain}." if curiosity_domain else ""

        opening = (
            f"Generate the ONE best startup idea for this founder. "
            f"{domain_note} {setup_note} "
            f"Business model: {model_type}. "
            f"No coding skills — no-code tools only. "
            f"Use the exact structured format. "
            f"The idea MUST be in the {curiosity_domain or 'stated'} niche."
        )
        idea = chat_with_idea_agent(user_id, opening, context)

        # Don't store the system prompt message in chat history
        history = [m for m in _user_sessions[user_id]["history"]
                   if not m["content"].startswith("Generate the ONE best")]

        with get_session() as db:
            crud.save_idea(db, user_id, idea, history)
            crud.upsert_pipeline_status(db, user_id, "done", None)

        log.info(f"[{user_id}] ✅ Pipeline complete — idea saved to DB")

    except Exception as e:
        log.error(f"[{user_id}] ❌ Pipeline error: {e}", exc_info=True)
        try:
            with get_session() as db:
                crud.upsert_pipeline_status(db, user_id, "error", None, str(e))
        except Exception as db_err:
            log.error(f"[{user_id}] Also failed to save error status: {db_err}")