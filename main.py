from fastapi import FastAPI

from routes.main import router as pipeline_router


app = FastAPI(
    title="Bizify AI Service",
    version="1.0.0",
)

app.include_router(pipeline_router)


@app.get("/")
def health():
    return {"status": "AI service running"}
