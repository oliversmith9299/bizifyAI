import time

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}


@router.get("/version-check")
def version_check():
    return {"version": "NEW_CODE_123"}
