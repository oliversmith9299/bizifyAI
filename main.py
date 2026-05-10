from fastapi import FastAPI

from db.connection import Base, engine
from db import crud, models
from routes.main import router as pipeline_router


app = FastAPI(
    title="Bizify AI Service",
    version="1.0.0",
)

app.include_router(pipeline_router)


@app.on_event("startup")
def ensure_database_tables():
    """Create missing tables for newly added pipeline models."""
    Base.metadata.create_all(bind=engine)
    from db.connection import SessionLocal

    session = SessionLocal()
    try:
        crud.seed_agents(session)
    finally:
        session.close()


@app.get("/")
def health():
    return {"status": "AI service running"}
