"""
#2 Problem Discovery (Analysis, not chatbot)
Pipeline: ProfileAnalysis → ProblemDiscovery → IdeaAgent

Fixes applied:
- Bare except → specific exception handling with logging
- Reddit URL validation before .json fetch attempt
- Query expansion now covers ALL keywords, not just first
- Graceful fallback when no enriched sources found (profile-derived mode)
- JSON parse failure now saves partial output instead of crashing
- Added snippet-only fallback when full page fetch fails
- Scoring improved: rewards evidence diversity
- FIX: Context window overflow → content truncated to 600 chars per source
- FIX: Token estimation guard before sending to LLM
- FIX: finish_reason check — detects silent truncation
- FIX: Auto-retry with fewer sources if prompt is too large
- FIX: Switched default model to llama3-groq-70b-8192-tool-use-preview (same window, more reliable JSON)
"""

import os
import json
import time
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# -------------------------
# Load ENV
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
# FIX: Use llama-3.3-70b-versatile — 128k context window vs llama3-70b-8192's 8k
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Safe token budget: leave 2000 tokens for output
# llama-3.3-70b-versatile = 128k context, llama3-70b-8192 = 8k context
MAX_PROMPT_CHARS = 80_000   # ~20k tokens, well within 128k window
CONTENT_CHARS_PER_SOURCE = 800  # per source content truncation

if not GROQ_API_KEY or not SERPER_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY or SERPER_API_KEY in .env")

# -------------------------
# Init LLM Client
# -------------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "profileAnalysis.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "problems.json")

# -------------------------
# Load Profile
# -------------------------
with open(INPUT_PATH, "r") as f:
    profile = json.load(f)

keywords = profile.get("search_direction", {}).get("keywords", [])
if not keywords:
    raise RuntimeError("profileAnalysis.json has no search_direction.keywords")

# -------------------------
# Query Expansion
# FIX: Was slicing to [:5] before iterating all keywords.
# Now generates queries for ALL keywords, then limits total.
# -------------------------
def expand_queries(keywords: list, max_total: int = 12) -> list:
    templates = [
        "{k} problems small business",
        "{k} reddit complaints",
        "{k} user pain points",
        "{k} challenges startup",
    ]
    queries = []
    for k in keywords:
        for t in templates:
            queries.append(t.format(k=k))
            if len(queries) >= max_total:
                return queries
    return queries

# -------------------------
# Search via Serper
# -------------------------
def search_google(query: str) -> dict:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        res = requests.post(url, json={"q": query}, headers=headers, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        log.warning(f"Search failed for '{query}': {e}")
        return {}

# -------------------------
# Extract Sources from Search Results
# -------------------------
def extract_sources(results: dict) -> list:
    sources = []
    for r in results.get("organic", []):
        link = r.get("link")
        if not link:
            continue
        source_type = "reddit" if "reddit.com" in link else "web"
        sources.append({
            "title": r.get("title", ""),
            "url": link,
            "snippet": r.get("snippet", ""),
            "type": source_type,
        })
    return sources

# -------------------------
# Reddit Content Fetcher
# FIX: Validate URL is a /comments/ post before trying .json trick.
# Subreddit listing pages, search pages, etc. don't work.
# -------------------------
def fetch_reddit_content(url: str) -> str:
    if "/comments/" not in url:
        log.debug(f"Skipping non-post Reddit URL: {url}")
        return ""
    try:
        json_url = url.rstrip("/") + ".json"
        headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
        res = requests.get(json_url, headers=headers, timeout=8)
        res.raise_for_status()
        data = res.json()

        post = data[0]["data"]["children"][0]["data"]
        comments = data[1]["data"]["children"]

        content = post.get("selftext", "") + "\n"
        for c in comments[:5]:
            body = c.get("data", {}).get("body", "")
            if body:
                content += body + "\n"

        return content[:2500]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        log.warning(f"Reddit fetch failed for {url}: {e}")
        return ""

# -------------------------
# Web Page Fetcher
# FIX: Specific exception handling + snippet fallback
# -------------------------
def fetch_web_content(url: str, snippet_fallback: str = "") -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
        res = requests.get(url, headers=headers, timeout=8)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove nav/footer/script noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
        return text[:2500]
    except requests.RequestException as e:
        log.warning(f"Web fetch failed for {url}: {e}")
        # FIX: Fall back to the Serper snippet — still useful signal
        return snippet_fallback

# -------------------------
# Pipeline: Search → Deduplicate → Enrich
# -------------------------
queries = expand_queries(keywords, max_total=12)
log.info(f"Running {len(queries)} search queries...")

all_sources = []
for q in queries:
    results = search_google(q)
    sources = extract_sources(results)
    all_sources.extend(sources)
    time.sleep(0.3)  # be polite to the API

# Deduplicate by URL
seen = set()
unique_sources = []
for s in all_sources:
    if s["url"] not in seen:
        unique_sources.append(s)
        seen.add(s["url"])

log.info(f"Found {len(unique_sources)} unique sources, enriching top 20...")

enriched_sources = []
for s in unique_sources[:20]:
    if s["type"] == "reddit":
        content = fetch_reddit_content(s["url"])
    else:
        content = fetch_web_content(s["url"], snippet_fallback=s.get("snippet", ""))

    # FIX: Lower threshold to 100 chars (snippet fallbacks are short but useful)
    if len(content) > 100:
        enriched_sources.append({
            "title": s["title"],
            "url": s["url"],
            "type": s["type"],
            # FIX: Truncate content per source — prevents context window overflow.
            # llama3-70b-8192 has only 8192 tokens. 10 sources × 2500 chars = ~25k tokens = crash.
            # 800 chars per source × 10 sources = ~8000 chars total for source content (~2k tokens).
            "content": content[:CONTENT_CHARS_PER_SOURCE],
        })

enriched_sources = enriched_sources[:10]
log.info(f"Enriched {len(enriched_sources)} sources with content")

# -------------------------
# FIX: Fallback mode when no sources found
# Instead of forcing the LLM to return empty, let it use profile context
# and clearly flag those problems as unvalidated.
# -------------------------
source_mode = "web_sourced" if enriched_sources else "profile_derived"
if source_mode == "profile_derived":
    log.warning("No enriched sources found — switching to profile-derived mode (unvalidated problems)")

# -------------------------
# LLM Prompt
# -------------------------
# Compact profile — only what the LLM needs, strips heavy fields to save tokens
profile_summary = {
    "recommended_industries": profile.get("recommended_industries", []),
    "recommended_problem_spaces": profile.get("recommended_problem_spaces", []),
    "founder_strengths": profile.get("personality_insights", {}).get("strengths", []),
    "target_customer_types": profile.get("personas", [{}])[0].get("type", "") if profile.get("personas") else "",
}

if source_mode == "web_sourced":
    prompt = f"""
You are a startup problem discovery AI. Extract REAL, SPECIFIC customer problems from the sources.

STRICT RULES:
1. Title must be a specific problem statement, NOT a category.
   BAD:  "Inventory Management"
   GOOD: "Small e-commerce stores lose revenue from stockouts due to manual inventory tracking"

2. Evidence quotes must be DIRECT PAIN expressions from customers or reports — not article descriptions.
   BAD:  "Discover how AI revolutionizes ecommerce..."
   GOOD: "We lost $12k last quarter because we oversold on Amazon while our Shopify was out of stock"

3. customer_segments MUST be filled — list 3-5 specific buyer types (not generic).

4. Extract AT LEAST 4 problems if the sources support it. Cover different problem spaces.

5. validation_score = min(100, unique_sources_count * 25 + evidence_quotes_count * 15)
   This means a problem with 1 source + 1 quote = 40. Max realistic score ~85.

6. Only include problems that match the founder's recommended industries.

Founder Profile (condensed):
{json.dumps(profile_summary, indent=2)}

Sources:
{json.dumps(enriched_sources, indent=2)}

Return ONLY valid JSON, no explanation, no markdown fences:

{{
  "problems": [
    {{
      "id": "P1",
      "title": "Specific problem statement as a sentence",
      "description": "2-3 sentences explaining the problem, who has it, and its business impact",
      "industry": "one of the founder's recommended industries",
      "target_customer": "specific role/type (e.g. Shopify merchant with <$500k/yr revenue)",
      "pain_level": "high | medium | low",
      "frequency": "high | medium | low",
      "current_solutions": "what they use today and why it fails",
      "gap_opportunity": "specific product/feature opportunity that doesn't exist yet",
      "source_type": "web_sourced",
      "sources": [
        {{ "title": "...", "url": "..." }}
      ],
      "evidence": [
        {{ "quote": "direct pain quote from source content", "source_url": "..." }}
      ],
      "validation_score": 0
    }}
  ],
  "customer_segments": [
    "Shopify merchants with under $500k annual revenue",
    "Independent fashion retailers using WooCommerce",
    "... (3-5 specific segments)"
  ],
  "personas": [
    {{
      "name": "...",
      "type": "...",
      "goal": "...",
      "pain": "..."
    }}
  ],
  "summary_insight": "2-3 sentence synthesis of the strongest opportunity based on the sources"
}}
"""
else:
    prompt = f"""
You are a startup problem discovery AI.

No web sources were available. Generate REALISTIC, SPECIFIC problems based on the founder's profile.
Mark all as source_type "profile_derived" with validation_score 35 (unvalidated).

STRICT RULES:
1. Title must be a specific problem statement sentence, NOT a category name.
2. Generate exactly 4 problems covering different recommended problem spaces.
3. customer_segments MUST be filled with 3-5 specific buyer types.
4. gap_opportunity must be concrete — what specific product/API/feature would solve it.

Founder Profile:
{json.dumps(profile, indent=2)}

Return ONLY valid JSON, no explanation, no markdown fences:

{{
  "problems": [
    {{
      "id": "P1",
      "title": "Specific problem statement as a sentence",
      "description": "2-3 sentences: who has the problem, what causes it, what it costs them",
      "industry": "...",
      "target_customer": "specific role/type",
      "pain_level": "high | medium | low",
      "frequency": "high | medium | low",
      "current_solutions": "what they use today and why it fails",
      "gap_opportunity": "specific product/API/feature opportunity",
      "source_type": "profile_derived",
      "sources": [],
      "evidence": [],
      "validation_score": 35
    }}
  ],
  "customer_segments": [
    "3-5 specific buyer types"
  ],
  "personas": [
    {{
      "name": "...",
      "type": "...",
      "goal": "...",
      "pain": "..."
    }}
  ],
  "summary_insight": "2-3 sentence synthesis of the strongest opportunity"
}}
"""

# -------------------------
# Call LLM — with token guard and auto-retry
# -------------------------
def call_llm(prompt: str, sources_used: list) -> str:
    """
    Call the LLM with a token size guard.
    If the prompt is too large, reduce sources and retry.
    Checks finish_reason to detect silent truncation.
    """
    estimated_chars = len(prompt)
    log.info(f"Prompt size: ~{estimated_chars:,} chars (~{estimated_chars // 4:,} tokens)")

    # If over budget, progressively reduce sources and rebuild prompt
    current_sources = sources_used[:]
    current_prompt = prompt

    while estimated_chars > MAX_PROMPT_CHARS and len(current_sources) > 2:
        current_sources = current_sources[:-2]  # drop 2 sources at a time
        log.warning(f"Prompt too large — reducing to {len(current_sources)} sources and retrying")
        current_prompt = current_prompt.replace(
            json.dumps(sources_used, indent=2),
            json.dumps(current_sources, indent=2)
        )
        estimated_chars = len(current_prompt)

    log.info(f"Calling LLM ({GROQ_MODEL}) with {len(current_sources)} sources...")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
            {"role": "user", "content": current_prompt},
        ],
        temperature=0.5,
        max_tokens=8000,  # FIX: 2000 was too small — full JSON output needs up to 4-6k tokens
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    raw = choice.message.content
    if not raw or not raw.strip():
        raise RuntimeError(
            f"LLM returned empty content. finish_reason='{finish_reason}'. "
            f"Model='{GROQ_MODEL}'. Prompt was ~{estimated_chars} chars (~{estimated_chars//4} tokens). "
            f"Fix: set GROQ_MODEL=llama-3.3-70b-versatile in your .env (128k context window)."
        )

    # FIX: Detect and repair truncated JSON (finish_reason=length means cut mid-response)
    if finish_reason == "length":
        log.warning("⚠️  LLM hit max_tokens (finish_reason=length) — attempting to repair truncated JSON")
        raw = _repair_truncated_json(raw.strip())

    return raw.strip()


def _repair_truncated_json(raw: str) -> str:
    """
    When the model is cut off mid-JSON, try to salvage whatever complete
    problem objects were already generated before the truncation point.
    Strategy: find the last complete problem block and close the JSON cleanly.
    """
    # Find the last complete problem object — ends with a closing }
    # Walk backwards to find the last well-formed closing brace before truncation
    last_complete = raw.rfind('},\n    {')   # between two problem objects
    if last_complete == -1:
        last_complete = raw.rfind('}')       # last closing brace of any kind

    if last_complete == -1:
        log.error("Cannot repair truncated JSON — no closing brace found")
        return raw  # let safe_parse_json handle the error

    # Truncate to last complete object and close the array + wrapper
    salvaged = raw[:last_complete + 1]

    # Remove trailing comma if present (would be invalid JSON)
    salvaged = salvaged.rstrip().rstrip(',')

    # Close the problems array and the root object with minimal required fields
    salvaged += (
        '\n  ],\n'
        '  "customer_segments": [],\n'
        '  "personas": [],\n'
        '  "summary_insight": "Truncated — partial results recovered."\n'
        '}'
    )

    log.info(f"Repaired truncated JSON — salvaged {salvaged.count('\"id\"')} problem(s)")
    return salvaged

log.info("Calling LLM for problem extraction...")
raw_output = call_llm(prompt, enriched_sources)

# -------------------------
# Parse JSON
# FIX: Strip markdown fences if LLM wraps output in ```json ... ```
# FIX: Save error output for debugging instead of just crashing
# -------------------------
def safe_parse_json(raw: str) -> dict:
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed: {e}")
        log.error(f"Raw output was:\n{raw[:1000]}")
        # Save raw output for debugging
        debug_path = OUTPUT_PATH.replace(".json", "_debug_raw.txt")
        with open(debug_path, "w") as f:
            f.write(raw)
        log.info(f"Raw LLM output saved to {debug_path}")
        raise

result = safe_parse_json(raw_output)

# -------------------------
# Post-process: Recalculate validation scores
# -------------------------
for p in result.get("problems", []):
    if p.get("source_type") == "profile_derived":
        p["validation_score"] = 35
    else:
        num_sources = len(p.get("sources", []))
        num_quotes = len(p.get("evidence", []))
        # 25 per unique source + 15 per evidence quote, max 85
        # (100 is reserved for problems with interview validation in later pipeline stages)
        p["validation_score"] = min(85, num_sources * 25 + num_quotes * 15)

# -------------------------
# Save Output
# -------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

problem_count = len(result.get("problems", []))
log.info(f"✅ problems.json created — {problem_count} problems found (mode: {source_mode})")
if problem_count == 0:
    log.warning("⚠️  Zero problems returned. Check: (1) SERPER_API_KEY quota, (2) network access, (3) LLM raw output in _debug_raw.txt")