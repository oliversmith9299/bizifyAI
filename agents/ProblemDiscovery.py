import os
import json
import time
import logging
import requests
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

MAX_PROMPT_CHARS = 80_000
CONTENT_CHARS_PER_SOURCE = 800

if not GROQ_API_KEY or not SERPER_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY or SERPER_API_KEY")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)


# ─────────────────────────────────────────────────────────
# Search Layer
# ─────────────────────────────────────────────────────────
def expand_queries(keywords: List[str], max_total=12) -> List[str]:
    templates = [
        "{k} problems small business",
        "{k} reddit complaints",
        "{k} user pain points",
        "{k} startup challenges",
    ]
    queries = []
    for k in keywords:
        for t in templates:
            queries.append(t.format(k=k))
            if len(queries) >= max_total:
                return queries
    return queries


def search_google(query: str) -> dict:
    try:
        res = requests.post(
            "https://google.serper.dev/search",
            json={"q": query},
            headers={"X-API-KEY": SERPER_API_KEY},
            timeout=10,
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        log.warning(f"Search failed: {e}")
        return {}


def extract_sources(results: dict) -> List[dict]:
    sources = []
    for r in results.get("organic", []):
        url = r.get("link")
        if not url:
            continue
        sources.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("snippet", ""),
            "type": "reddit" if "reddit.com" in url else "web",
        })
    return sources


# ─────────────────────────────────────────────────────────
# Content Fetchers
# ─────────────────────────────────────────────────────────
def fetch_reddit(url: str) -> str:
    if "/comments/" not in url:
        return ""
    try:
        res = requests.get(url.rstrip("/") + ".json", timeout=8)
        data = res.json()
        post = data[0]["data"]["children"][0]["data"]
        comments = data[1]["data"]["children"]

        content = post.get("selftext", "")
        for c in comments[:5]:
            content += "\n" + c["data"].get("body", "")

        return content[:CONTENT_CHARS_PER_SOURCE]
    except Exception:
        return ""


def fetch_web(url: str, fallback="") -> str:
    try:
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = " ".join(
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        )

        return text[:CONTENT_CHARS_PER_SOURCE]
    except Exception:
        return fallback[:CONTENT_CHARS_PER_SOURCE]


# ─────────────────────────────────────────────────────────
# LLM Helpers
# ─────────────────────────────────────────────────────────
def call_llm(prompt: str, sources: list) -> str:

    while len(prompt) > MAX_PROMPT_CHARS and len(sources) > 2:
        sources = sources[:-2]
        prompt = prompt.replace(
            json.dumps(sources, indent=2),
            json.dumps(sources[:-2], indent=2),
        )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY JSON"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=8000,
    )

    return response.choices[0].message.content.strip()


def safe_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────
# MAIN FUNCTION (PIPELINE READY)
# ─────────────────────────────────────────────────────────
def run_problem_discovery(profile: dict, questionnaire: dict) -> dict:

    keywords = profile.get("search_direction", {}).get("keywords", [])
    if not keywords:
        raise ValueError("No keywords found")

    # ── SEARCH ─────────────────────────
    queries = expand_queries(keywords)
    all_sources = []

    for q in queries:
        results = search_google(q)
        all_sources.extend(extract_sources(results))
        time.sleep(0.3)

    # Deduplicate
    seen = set()
    unique = []
    for s in all_sources:
        if s["url"] not in seen:
            unique.append(s)
            seen.add(s["url"])

    # ── ENRICH ─────────────────────────
    enriched = []
    for s in unique[:20]:
        content = fetch_reddit(s["url"]) if s["type"] == "reddit" \
                  else fetch_web(s["url"], s["snippet"])

        if len(content) > 100:
            enriched.append({
                "title": s["title"],
                "url": s["url"],
                "content": content
            })

    enriched = enriched[:10]

    source_mode = "web_sourced" if enriched else "profile_derived"

    # ── PROMPT ─────────────────────────
    prompt = f"""
Extract real startup problems.

Profile:
{json.dumps(profile, indent=2)}

Sources:
{json.dumps(enriched, indent=2)}

Return JSON with problems.
"""

    raw = call_llm(prompt, enriched)
    result = safe_json(raw)

    # ── SCORING ────────────────────────
    for p in result.get("problems", []):
        if source_mode == "profile_derived":
            p["validation_score"] = 35
        else:
            p["validation_score"] = min(
                85,
                len(p.get("sources", [])) * 25 +
                len(p.get("evidence", [])) * 15
            )

    return result