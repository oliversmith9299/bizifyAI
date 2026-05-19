"""
agents/search_pipeline.py
==========================
Centralized Search + Extraction Pipeline.

Architecture
------------
                  queries + schema + domains
                           │
                    ┌──────▼──────┐
                    │   Tavily    │  advanced search, domain-filtered
                    │  (primary)  │  returns pre-cleaned text
                    └──────┬──────┘
                           │ list of clean sources
                    ┌──────▼──────┐
                    │    Groq     │  llama-3.1-8b-instant (fast + cheap)
                    │ extraction  │  extracts structured JSON per source
                    └──────┬──────┘
                           │ list of ExtractedSource
                    ┌──────▼──────┐
                    │SearchResults│  .to_prompt_context() → LLM-ready string
                    └─────────────┘  .to_sources_meta()  → DB storage tuple

Fallback
--------
If TAVILY_API_KEY is missing → falls back to gather_sources() (Serper + BS4).
If LLM extraction fails for a source → falls back to regex data-point extraction.

Agent contract
--------------
Each agent supplies:
    queries          : list[str]   — built by the agent's _build_search_queries()
    tavily_api_key   : str         — from agents/config.py
    extraction_schema: dict        — what fields to extract, per-agent definition
    keywords         : list[str]   — used only in fallback relevance scoring
    include_domains  : list[str]   — trusted domains for this agent's topic
    groq_client      : OpenAI      — shared client from agents/config.py
    extraction_model : str         — GROQ_EXTRACTION_MODEL from agents/config.py

Each agent receives:
    SearchResults with:
        .sources              — list[ExtractedSource], best-first by Tavily score
        .source_mode          — "web_sourced" | "profile_derived"
        .to_prompt_context()  — structured string ready for the main LLM prompt
        .to_sources_meta()    — (count, [{url, title}]) for DB storage
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Thread pool for parallel LLM extraction calls
_executor = ThreadPoolExecutor(max_workers=6)

# Domains considered authoritative — shown with a ★ badge in prompt context
_TRUSTED_DOMAINS = {
    "statista.com", "crunchbase.com", "mckinsey.com", "deloitte.com",
    "pwc.com", "gartner.com", "forrester.com", "ibisworld.com",
    "grandviewresearch.com", "mordorintelligence.com", "marketsandmarkets.com",
    "techcrunch.com", "bloomberg.com", "reuters.com", "ft.com",
    "saastr.com", "openview.co", "baremetrics.com", "a16z.com",
    "ycombinator.com", "indiehackers.com", "producthunt.com",
    "similarweb.com", "g2.com", "capterra.com",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedSource:
    """One source: Tavily content + LLM-structured extraction."""
    url: str
    title: str
    tavily_score: float          # relevance score from Tavily (0–1)
    raw_content: str             # cleaned text Tavily returned
    llm_extracted: dict          # structured fields from LLM extraction
    is_trusted: bool             # True if domain is in _TRUSTED_DOMAINS

    def to_context_block(self) -> str:
        """Compact, structured block ready for the main LLM prompt."""
        badge = " ★" if self.is_trusted else ""
        domain = self.url.split("/")[2].replace("www.", "") if "/" in self.url else self.url
        lines = [f"[{self.title}]({self.url})  {domain}{badge}"]

        # LLM-extracted structured fields — skip nulls
        for key, value in self.llm_extracted.items():
            if value is not None and value != "" and value != [] and value != {}:
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    lines.append(f"  {label}: {', '.join(str(v) for v in value[:4])}")
                else:
                    lines.append(f"  {label}: {value}")

        return "\n".join(lines)


@dataclass
class SearchResults:
    """Full output of run_search_pipeline() — what every agent receives."""
    sources: list[ExtractedSource]
    source_mode: str      # "web_sourced" | "profile_derived"
    query_count: int

    def to_prompt_context(self) -> str:
        """Render all sources as a structured, LLM-ready context string."""
        if not self.sources:
            return ""
        trusted   = [s for s in self.sources if s.is_trusted]
        untrusted = [s for s in self.sources if not s.is_trusted]
        ordered   = trusted + untrusted       # trusted sources first

        blocks = [s.to_context_block() for s in ordered]
        header = f"=== WEB RESEARCH ({len(self.sources)} sources) ==="
        return header + "\n\n" + "\n\n".join(blocks)

    def to_sources_meta(self) -> tuple[int, list[dict]]:
        """Return (sources_used, sources_list) for DB storage."""
        return (
            len(self.sources),
            [{"url": s.url, "title": s.title} for s in self.sources],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tavily search
# ─────────────────────────────────────────────────────────────────────────────

def _tavily_search(
    queries: list[str],
    api_key: str,
    include_domains: list[str],
    max_results_per_query: int = 3,
) -> list[dict]:
    """
    Run all queries through Tavily, deduplicate by URL, return raw results.

    Uses advanced search depth for richer content.
    Domain filtering steers results toward authoritative sources.
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        log.error("tavily-python not installed. Run: pip install tavily-python")
        return []

    client = TavilyClient(api_key=api_key)
    seen: set[str] = set()
    all_results: list[dict] = []

    for query in queries:
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results_per_query,
                include_domains=include_domains or [],
                include_answer=False,
            )
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    all_results.append(r)
            time.sleep(0.2)   # light rate limiting
        except Exception as e:
            log.warning("Tavily query failed for '%s': %s", query, e)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_prompt(content: str, title: str, schema: dict) -> str:
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in schema.items())
    return f"""You are a data extraction assistant for startup market research.

From the source text below, extract ONLY the specific data points listed.
Return valid JSON with exactly those keys. Use null for any field not found.
Do not add extra fields. Do not explain. Return JSON only.

FIELDS TO EXTRACT:
{{
{schema_lines}
}}

SOURCE: {title}
TEXT:
{content[:2500]}"""


def _extract_one_source(
    raw: dict,
    schema: dict,
    groq_client: Any,
    model: str,
) -> ExtractedSource:
    """Extract structured data from one Tavily result using Groq llama-8b."""
    from agents.utils import parse_llm_json

    url     = raw.get("url", "")
    title   = raw.get("title", "")
    content = raw.get("content", "")
    score   = float(raw.get("score", 0.0))
    domain  = url.split("/")[2].replace("www.", "") if "/" in url else ""

    # LLM extraction
    extracted: dict = {}
    try:
        prompt   = _build_extraction_prompt(content, title, schema)
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
        extracted = parse_llm_json(response.choices[0].message.content)
    except ValueError as e:
        # LLM returned malformed JSON — regex fallback is safe here
        log.warning("LLM extraction parse error for '%s': %s — using regex fallback", url, e)
        extracted = _regex_fallback(content, schema)
    except Exception as e:
        # API error, timeout, etc. — still use regex so the source isn't lost
        log.error("LLM extraction failed for '%s' (%s: %s) — using regex fallback",
                  url, type(e).__name__, e)
        extracted = _regex_fallback(content, schema)

    return ExtractedSource(
        url=url,
        title=title,
        tavily_score=score,
        raw_content=content,
        llm_extracted=extracted,
        is_trusted=any(t in domain for t in _TRUSTED_DOMAINS),
    )


def _regex_fallback(content: str, schema: dict) -> dict:
    """Pull percentages, monetary values, and numbers when LLM extraction fails."""
    percentages     = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", content)
    monetary_values = re.findall(
        r"\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T)\b)?",
        content, re.IGNORECASE,
    )
    return {k: None for k in schema} | {
        "_percentages":     percentages[:5]     or None,
        "_monetary_values": monetary_values[:5] or None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_search_pipeline(
    queries: list[str],
    tavily_api_key: str,
    extraction_schema: dict,
    keywords: list[str],
    include_domains: list[str] | None = None,
    max_sources: int = 8,
    groq_client: Any = None,
    extraction_model: str = "llama-3.1-8b-instant",
    serper_fallback_key: str | None = None,
) -> SearchResults:
    """
    Full pipeline: Tavily search → parallel LLM extraction → ranked results.

    Args:
        queries           : Search queries built by the calling agent.
        tavily_api_key    : TAVILY_API_KEY from config.
        extraction_schema : Dict mapping field_name → description of what to extract.
                            Defined per-agent. e.g. {"market_size": "TAM in dollars"}
        keywords          : Fallback relevance scoring keywords (used only when
                            Tavily is unavailable and old gather_sources runs).
        include_domains   : Preferred authoritative domains for this agent's topic.
                            Tavily will prioritise results from these domains.
        max_sources       : Maximum enriched sources to keep.
        groq_client       : Shared OpenAI-compatible Groq client from config.
        extraction_model  : Fast Groq model for extraction (default: llama-3.1-8b-instant).
        serper_fallback_key: SERPER_API_KEY — used if Tavily is unavailable.

    Returns:
        SearchResults — call .to_prompt_context() for the LLM prompt string,
                        .to_sources_meta() for DB storage.
    """
    # ── Tavily unavailable → fall back to Serper + regex ─────────────────────
    if not tavily_api_key:
        log.warning("TAVILY_API_KEY not set — falling back to Serper search")
        return _serper_fallback(queries, serper_fallback_key, keywords, max_sources)

    # ── 1. Tavily search ──────────────────────────────────────────────────────
    raw_results = _tavily_search(
        queries=queries,
        api_key=tavily_api_key,
        include_domains=include_domains or [],
        max_results_per_query=3,
    )

    if not raw_results:
        log.warning("Tavily returned no results — falling back to Serper")
        return _serper_fallback(queries, serper_fallback_key, keywords, max_sources)

    # Cap to max_sources before paying for LLM extraction calls
    raw_results = raw_results[:max_sources]

    # ── 2. Parallel LLM extraction ────────────────────────────────────────────
    extracted: list[ExtractedSource] = []

    if groq_client:
        futures = {
            _executor.submit(_extract_one_source, raw, extraction_schema, groq_client, extraction_model): raw
            for raw in raw_results
        }
        for future in as_completed(futures):
            try:
                extracted.append(future.result(timeout=15))
            except Exception as e:
                log.warning("Extraction future failed: %s", e)
    else:
        # No groq client — regex fallback for all sources
        for raw in raw_results:
            url    = raw.get("url", "")
            domain = url.split("/")[2].replace("www.", "") if "/" in url else ""
            extracted.append(ExtractedSource(
                url=url,
                title=raw.get("title", ""),
                tavily_score=float(raw.get("score", 0.0)),
                raw_content=raw.get("content", ""),
                llm_extracted=_regex_fallback(raw.get("content", ""), extraction_schema),
                is_trusted=any(t in domain for t in _TRUSTED_DOMAINS),
            ))

    # Trusted sources first, then by Tavily relevance score
    extracted.sort(key=lambda s: (s.is_trusted, s.tavily_score), reverse=True)

    return SearchResults(
        sources=extracted,
        source_mode="web_sourced" if extracted else "profile_derived",
        query_count=len(queries),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serper fallback (keeps old behaviour when Tavily is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _serper_fallback(
    queries: list[str],
    serper_key: str | None,
    keywords: list[str],
    max_sources: int,
) -> SearchResults:
    """Use the old Serper + BeautifulSoup pipeline as a fallback."""
    if not serper_key:
        return SearchResults(sources=[], source_mode="profile_derived", query_count=len(queries))

    from agents.utils import gather_sources

    raw = gather_sources(queries, serper_key, max_sources=max_sources)

    sources = []
    for r in raw:
        url    = r.get("url", "")
        domain = url.split("/")[2].replace("www.", "") if "/" in url else ""
        sources.append(ExtractedSource(
            url=url,
            title=r.get("title", ""),
            tavily_score=0.5,
            raw_content=r.get("content", ""),
            llm_extracted={"_content": r.get("content", "")[:600]},
            is_trusted=any(t in domain for t in _TRUSTED_DOMAINS),
        ))

    return SearchResults(
        sources=sources,
        source_mode="web_sourced" if sources else "profile_derived",
        query_count=len(queries),
    )
