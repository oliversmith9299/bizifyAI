"""
agents/PipelineRunner.py
========================
Background task — runs all 3 AI pipeline steps and saves to Supabase.
Called by routes/main.py as a FastAPI BackgroundTask.

Fix applied: _get_db() now uses get_session() context manager
so the DB session is always properly committed and closed,
even when an exception occurs mid-pipeline.
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
    combined = {"questionnaire": questionnaire, "skills": {"skills": skills}}

    prompt = f"""
You are a senior startup advisor and venture builder.
Analyze this user and determine what kind of business they can realistically build.

PRIORITY ORDER for search_direction keywords:
1. business_interests (HIGHEST) — always marketplace-related if Marketplace selected
2. target_region — always append to keywords
3. curiosity_domain — shapes the angle
4. skills — only determines HOW they execute, not WHAT space to search
NEVER generate keywords about raw technical skills (e.g. "Python for X").

INPUT:
{json.dumps(combined, indent=2)}

RULES:
- Base ALL on actual skills and questionnaire answers
- Empty skills list = operator/business/no-code models ONLY
- Prefer platform, service, operational models

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
            {"role": "system", "content": "Return ONLY valid JSON. No explanation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


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
    region_mod = {"MENA": "MENA Middle East", "Egypt": "Egypt", "Global": ""}.get(target_region, target_region)
    is_marketplace = any("marketplace" in b.lower() for b in business_interests)
    keywords = profile.get("search_direction", {}).get("keywords", [])

    templates = (
        ["{k} marketplace seller problems {r}", "{k} marketplace buyer trust {r}",
         "{k} marketplace startup challenges", "{k} pain points {r}"]
        if is_marketplace else
        ["{k} problems small business {r}", "{k} user pain points {r}",
         "{k} challenges startup", "{k} complaints reddit"]
    )

    queries = []
    for k in keywords:
        for t in templates:
            queries.append(t.format(k=k, r=region_mod).strip())
            if len(queries) >= 16:
                break
        if len(queries) >= 16:
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

    profile_summary = {
        "recommended_industries": profile.get("recommended_industries", []),
        "recommended_problem_spaces": profile.get("recommended_problem_spaces", []),
        "founder_strengths": profile.get("personality_insights", {}).get("strengths", []),
        "key_skill_gaps": profile.get("founder_profile", {}).get("key_skill_gaps", []),
        "target_region": target_region,
        "business_type_interest": business_interests,
        "experience_level": u.get("experience_level", ""),
        "risk_tolerance": u.get("risk_tolerance", ""),
        "founder_setup": u.get("founder_setup", ""),
    }

    prompt = f"""
You are a startup problem discovery AI. Extract REAL, SPECIFIC customer problems.

RULES:
1. Title = specific problem sentence, not a category
2. business_type = {business_interests} — if Marketplace → marketplace dynamics only
3. region = "{target_region}" — ground all problems here
4. Extract at least 4 problems
5. validation_score = min(85, sources*25 + quotes*15)
6. customer_segments: 3-5 specific types
{"Source mode: web_sourced. Use provided sources." if source_mode == "web_sourced" else "Source mode: profile_derived. Validation score = 35."}

Founder Profile: {json.dumps(profile_summary)}
{"Sources: " + json.dumps(enriched) if enriched else "No sources available."}

Return ONLY valid JSON:
{{"problems":[{{"id":"P1","title":"","description":"","industry":"","target_customer":"",
"pain_level":"high|medium|low","frequency":"high|medium|low","current_solutions":"",
"gap_opportunity":"","source_type":"{source_mode}","sources":[],"evidence":[],"validation_score":0}}],
"customer_segments":[],"personas":[],"summary_insight":""}}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
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

RULES:
1. Read HARD EXECUTION CONSTRAINTS in context first — non-negotiable.
2. "Marketplace" = B2C platform where consumers buy. NOT B2B tools.
3. No coding skills = no-code tools only (Sharetribe, Notion, WhatsApp, Webflow).
4. Solo founder = laptop-only, no warehouse, no team.

FORMAT every idea exactly:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 IDEA: [Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem it solves : ...
Target customer   : ...
How it works      : ...
Launch stack      : ...
Business model    : ...
Why you can do it : ...
First 7-day test  : ...
Startup cost      : ...
Risk level        : Low/Medium/High — reason
━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANTI-LOOP: Never repeat the same B2B/logistics idea. If user wants B2C — switch problem space.
"""


def _build_context(profile: dict, problems: dict, questionnaire: dict, skills: list) -> str:
    u = questionnaire.get("user_profile", {})
    has_tech = any(s.lower() in ["python", "machine learning", "apis", "backend development",
                                  "software development", "coding"] for s in skills)
    bi = u.get("business_interests", [])
    region = u.get("target_region", "")
    setup = u.get("founder_setup", "")
    risk = u.get("risk_tolerance", "")

    c = ["=== ⚠️ HARD EXECUTION CONSTRAINTS ==="]
    if not has_tech:
        c += ["❌ NO SOFTWARE BUILDING — no coding skills",
              "✅ ALLOWED: Sharetribe, Notion, WhatsApp, Webflow, Airtable"]
    if "solo" in setup.lower():
        c += ["❌ NO LARGE TEAM — solo founder, laptop only"]
    if "moderate" in risk.lower():
        c += ["❌ NO HIGH CAPITAL — under $500 to launch"]
    if "marketplace" in " ".join(bi).lower():
        c += ["✅ B2C MARKETPLACE ONLY — consumers as end users",
              "❌ FORBIDDEN: B2B SaaS, logistics tools, consulting"]
    c += [f"✅ REGION: {region}"]

    p = ["=== VALIDATED PROBLEMS ==="]
    for prob in problems.get("problems", []):
        if prob.get("validation_score", 0) >= 40:
            p.append(f"[{prob['id']}] {prob['title']}")
            p.append(f"  Gap: {prob.get('gap_opportunity', '')}")

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

    FIX: Uses get_session() context manager so the DB session is always
    properly committed and closed, even when an exception occurs.
    """
    from db import crud
    from db.connection import get_session   # ← FIX: use context manager, not raw SessionLocal

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

        opening = (
            "Generate the ONE best startup idea for this founder. "
            "Check constraints: no coding, solo, B2C marketplace, under $500. "
            "Use the exact structured format."
        )
        idea = chat_with_idea_agent(user_id, opening, context)
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