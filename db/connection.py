import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(override=True)

# ── NEVER hardcode credentials here — always use .env ────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to your .env file:\n"
        "DATABASE_URL=postgresql://user:password@host:5432/dbname"
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"sslmode": "require"},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Single Base shared across ALL models in the project
# Import this Base in every models.py file — never create a new one
Base = declarative_base()

# Backwards-compatible alias
db = SessionLocal


def _safe_database_url() -> str:
    """Return the database URL with password hidden for logs."""
    try:
        scheme, rest = DATABASE_URL.split("://", 1)
        credentials, host = rest.rsplit("@", 1)
        username = credentials.split(":", 1)[0]
        return f"{scheme}://{username}:****@{host}"
    except ValueError:
        return "postgresql://****"


def test_connection() -> bool:
    """Test the PostgreSQL connection."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print(f"Connected to PostgreSQL: {_safe_database_url()}")
        return True
    except Exception as exc:
        print(f"Failed to connect to PostgreSQL: {exc}")
        return False


def get_db():
    """FastAPI dependency — provides a SQLAlchemy session, always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_session():
    """Context manager for background tasks and scripts."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine():
    return engine