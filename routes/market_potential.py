import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from db.connection import get_db, SessionLocal
from db import crud
from routes.dependencies import (
    verify_api_key,
    sse_chat_stream,
    SectionChatInput,
    RegenerateCustomInput,
)

router = APIRouter()


@router.post("/market-potential/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_market_potential(user_id: str, db=Depends(get_db)):
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")

    customers_row   = crud.get_customers(db, user_id)
    competition_row = crud.get_competition(db, user_id)

    from agents.SixMaketPotential import run_market_potential
    result = run_market_potential(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data   if customers_row   else None,
        competition=competition_row.data if competition_row else None,
    )
    return {"user_id": user_id, "status": "done", "market_potential": result}


@router.post("/market-potential/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_market_potential(user_id: str, db=Depends(get_db)):
    return generate_market_potential(user_id, db)


@router.post("/market-potential/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_market_potential_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    idea_row     = crud.get_idea(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not idea_row or not problems_row:
        raise HTTPException(status_code=425, detail="Idea or problems not ready.")

    customers_row   = crud.get_customers(db, data.user_id)
    competition_row = crud.get_competition(db, data.user_id)

    from agents.SixMaketPotential import run_market_potential
    result = run_market_potential(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data   if customers_row   else None,
        competition=competition_row.data if competition_row else None,
        custom_prompt=data.custom_prompt,
    )
    return {"user_id": data.user_id, "status": "done", "market_potential": result}


@router.post("/market-potential/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_market_potential(data: SectionChatInput, db=Depends(get_db)):
    mp_row = crud.get_market_potential(db, data.user_id)
    if not mp_row:
        raise HTTPException(status_code=425, detail="Market potential not generated yet. Call POST /market-potential/{user_id} first.")

    from agents.SixMaketPotential import chat_market_potential as _chat
    reply = _chat(current_analysis=mp_row.data, user_message=data.message, history=data.history)

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_market_potential(db, data.user_id, mp_row.data, updated_history)

    return {"user_id": data.user_id, "reply": reply, "chat_history_length": len(updated_history)}


@router.post("/market-potential/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_market_potential_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    mp_row = crud.get_market_potential(db, data.user_id)
    if not mp_row:
        raise HTTPException(status_code=425, detail="Market potential not generated yet.")

    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.market_potential_prompt import MARKET_POTENTIAL_CHAT_PROMPT

    current_data = mp_row.data
    user_id      = data.user_id
    context      = "=== CURRENT MARKET POTENTIAL ANALYSIS ===\n" + json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": MARKET_POTENTIAL_CHAT_PROMPT},
        {"role": "system", "content": context},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        updated = data.history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_market_potential(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/market-potential/{user_id}", dependencies=[Depends(verify_api_key)])
def get_market_potential(user_id: str, db=Depends(get_db)):
    row = crud.get_market_potential(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No market potential analysis found for this user.")

    return {"user_id": user_id, "market_potential": row.data, "chat_history": row.chat_history or []}
