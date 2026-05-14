"""
tests/conftest.py
=================
Shared test configuration, engine setup, and fixtures.

Order of operations matters here:
  1. Set env vars before any project imports (so db.connection sees them).
  2. Import db.connection (which calls load_dotenv(override=True) from .env).
  3. Force-set critical test env vars again so .env values don't bleed in.
  4. Swap engine and session factory to SQLite in-memory.
  5. Create all AI-owned tables in the test DB once.

Individual test fixtures then provide isolated DB sessions and a TestClient
with the get_db dependency overridden.
"""

import os

# ── 1. Pre-seed env vars so db.connection can import without error ─────────────
os.environ.setdefault("DATABASE_URL",   "postgresql://test:test@localhost/testdb")
os.environ.setdefault("GROQ_API_KEY",   "test-groq-key")
os.environ.setdefault("GROQ_API_BASE",  "https://api.groq.com/openai/v1")
os.environ.setdefault("GROQ_MODEL",     "llama-3.3-70b-versatile")
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
os.environ.setdefault("API_SECRET_KEY", "test-secret-key")

# ── 2. Import db.connection (triggers load_dotenv(override=True) from .env) ───
import db.connection as _db_conn

# ── 3. Force-reset critical vars after .env may have overwritten them ─────────
#       agents/config.py reads these at import time with load_dotenv() (no override),
#       so whatever os.environ holds at that moment is what agents see.
os.environ["GROQ_API_KEY"]   = "test-groq-key"
os.environ["GROQ_API_BASE"]  = "https://api.groq.com/openai/v1"
os.environ["GROQ_MODEL"]     = "llama-3.3-70b-versatile"
os.environ["SERPER_API_KEY"] = "test-serper-key"
os.environ["API_SECRET_KEY"] = "test-secret-key"

# ── 4. Replace the PostgreSQL engine with SQLite in-memory ────────────────────
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)

_db_conn.engine       = _test_engine
_db_conn.SessionLocal = _TestSessionLocal
_db_conn.db           = _TestSessionLocal

# ── 5. Create all model tables once per test session ─────────────────────────
from db.models import Base
Base.metadata.create_all(bind=_test_engine)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
import pytest
from fastapi.testclient import TestClient
from db.connection import get_db

TEST_API_KEY = "test-secret-key"


@pytest.fixture
def db_session():
    """
    Yields a SQLAlchemy session backed by the in-memory SQLite DB.
    Rolls back after each test to keep tests fully isolated.
    """
    session = _TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    """
    FastAPI TestClient with:
      - get_db overridden to use the same test session as the test
      - API_SECRET_KEY patched in routes.dependencies so auth works with
        the hardcoded test key without needing the real .env key
    """
    from main import app
    import routes.dependencies as _deps

    _original_key = _deps.API_SECRET_KEY
    _deps.API_SECRET_KEY = TEST_API_KEY

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
    _deps.API_SECRET_KEY = _original_key


@pytest.fixture
def api_headers():
    """Default authenticated headers matching the test secret key."""
    return {"X-API-KEY": TEST_API_KEY}
