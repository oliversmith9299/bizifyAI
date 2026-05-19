"""
agents/config.py
================
Single source of truth for all configuration loaded from .env.

Every agent imports from here instead of calling os.getenv() directly.
No values are written in this file — everything comes from the .env file.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Groq / LLM ───────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE")
GROQ_MODEL    = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")
if not GROQ_API_BASE:
    raise RuntimeError("GROQ_API_BASE is not set in .env")
if not GROQ_MODEL:
    raise RuntimeError("GROQ_MODEL is not set in .env")

# Shared OpenAI-compatible client pointed at Groq
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# ── Search ────────────────────────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ── Security ──────────────────────────────────────────────────────────────────
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of origins allowed to call the AI service.
# In production this should be set to the backend's origin only, e.g.:
#   ALLOWED_ORIGINS=https://api.bizify.com
# Defaults to empty (blocks all browser-direct calls) when not set.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
