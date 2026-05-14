# NOTE: This is the standalone CLI version of ProblemDiscovery.
# The FastAPI pipeline uses agents/PipelineRunner.run_problem_discovery instead,
# which has more features (B2B guard, curiosity-domain templates, region modifiers).
# Keep the two in sync when changing the output schema.

import json
import time
import logging
import requests
from typing import Dict, List
from bs4 import BeautifulSoup
from agents.utils import (
    parse_llm_json,
    search_serper as _search_serper_util,
    extract_sources,
    fetch_reddit,
    fetch_web,
)
from agents.config import client, GROQ_MODEL, SERPER_API_KEY


# ─────────────────────────────────────────────────────────
# Setupp
# ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 80_000
CONTENT_CHARS_PER_SOURCE = 800


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


# search_google, extract_sources, fetch_reddit, fetch_web
# are now shared from agents.utils — imported at the top.
def search_google(query: str) -> dict:
    return _search_serper_util(query, SERPER_API_KEY)


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
    result = parse_llm_json(raw)

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