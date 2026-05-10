"""
Shared utilities for all agents.
Import from here instead of duplicating per-agent.
"""

import json
import logging
import time

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

CONTENT_CHARS_PER_SOURCE = 800


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_json(raw: str) -> dict:
    """Strip markdown code fences and parse JSON from an LLM response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def truncate_sources(sources: list, max_chars: int = 80_000) -> list:
    """Drop sources from the tail until the JSON fits inside max_chars."""
    while len(sources) > 2:
        if len(json.dumps(sources)) <= max_chars:
            break
        sources = sources[:-2]
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Web search — Serper API
# ─────────────────────────────────────────────────────────────────────────────

def search_serper(query: str, api_key: str) -> dict:
    """Call Serper Google Search API. Returns raw response dict."""
    try:
        res = requests.post(
            "https://google.serper.dev/search",
            json={"q": query},
            headers={"X-API-KEY": api_key},
            timeout=10,
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        log.warning(f"[search_serper] failed for '{query}': {e}")
        return {}


def extract_sources(results: dict) -> list:
    """Pull title, URL, snippet, and type from a Serper response."""
    sources = []
    for r in results.get("organic", []):
        url = r.get("link")
        if not url:
            continue
        sources.append({
            "title":   r.get("title", ""),
            "url":     url,
            "snippet": r.get("snippet", ""),
            "type":    "reddit" if "reddit.com" in url else "web",
        })
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Content fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit(url: str) -> str:
    """Fetch the post body and top comments from a Reddit thread."""
    if "/comments/" not in url:
        return ""
    try:
        res = requests.get(
            url.rstrip("/") + ".json",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        data = res.json()
        post     = data[0]["data"]["children"][0]["data"]
        comments = data[1]["data"]["children"]
        content  = post.get("selftext", "")
        for c in comments[:5]:
            content += "\n" + c["data"].get("body", "")
        return content[:CONTENT_CHARS_PER_SOURCE]
    except Exception:
        return ""


def fetch_web(url: str, fallback: str = "") -> str:
    """Fetch and clean the main text from a web page."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        r.raise_for_status()
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


# ─────────────────────────────────────────────────────────────────────────────
# High-level helper used by section agents
# ─────────────────────────────────────────────────────────────────────────────

def gather_sources(
    queries: list,
    api_key: str,
    max_sources: int = 10,
    delay: float = 0.3,
) -> list:
    """
    Run all queries through Serper, deduplicate URLs, fetch content.

    Returns a list of enriched source dicts:
      { title, url, content }

    Only sources with >100 characters of real content are included.
    """
    seen: set = set()
    all_sources = []

    for q in queries:
        results = search_serper(q, api_key)
        for s in extract_sources(results):
            if s["url"] not in seen:
                seen.add(s["url"])
                all_sources.append(s)
        time.sleep(delay)

    enriched = []
    for s in all_sources[:max_sources * 2]:   # fetch more, keep best
        content = (
            fetch_reddit(s["url"])
            if s["type"] == "reddit"
            else fetch_web(s["url"], s.get("snippet", ""))
        )
        if len(content) > 100:
            enriched.append({"title": s["title"], "url": s["url"], "content": content})
        if len(enriched) >= max_sources:
            break

    return enriched
