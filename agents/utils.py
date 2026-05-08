"""
Shared utilities for all agents.
Import from here instead of duplicating per-agent.
"""

import json
import logging

log = logging.getLogger(__name__)


def parse_llm_json(raw: str) -> dict:
    """
    Strip markdown code fences and parse the JSON an LLM returned.
    Raises json.JSONDecodeError if the content is not valid JSON after cleaning.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def truncate_sources(sources: list, max_chars: int = 80_000) -> list:
    """
    Drop sources from the tail until the JSON serialisation of the list
    fits inside max_chars. Prevents LLM context overflow on long search results.
    """
    while len(sources) > 2:
        if len(json.dumps(sources)) <= max_chars:
            break
        sources = sources[:-2]
    return sources
