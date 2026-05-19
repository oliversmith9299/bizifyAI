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


def _load_deps(user_id: str, db: Any) -> tuple:
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=409, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=409, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row, problems_row,
        crud.get_customers(db, user_id),
        crud.get_competition(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
        crud.get_business_model(db, user_id),
    )


@router.post("/functions-list/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_functions_list(user_id: str, db=Depends(get_db)):
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row, bm_row = _load_deps(user_id, db)

    from agents.NineFunctionsList import run_functions_list
    result = run_functions_list(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
    )
    return {"user_id": user_id, "status": "done", "functions_list": result}


@router.post("/functions-list/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_functions_list(user_id: str, db=Depends(get_db)):
    return generate_functions_list(user_id, db)


@router.post("/functions-list/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_functions_list_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row, bm_row = _load_deps(data.user_id, db)

    from agents.NineFunctionsList import run_functions_list
    result = run_functions_list(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        custom_prompt=data.custom_prompt,
    )
    return {"user_id": data.user_id, "status": "done", "functions_list": result}


@router.post("/functions-list/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_functions_list(data: SectionChatInput, db=Depends(get_db)):
    fl_row = crud.get_functions_list(db, data.user_id)
    if not fl_row:
        raise HTTPException(status_code=409, detail="Functions list not generated yet. Call POST /functions-list/{user_id} first.")

    from agents.NineFunctionsList import chat_functions_list as _chat
    reply = _chat(current_analysis=fl_row.data, user_message=data.message, history=data.history)

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_functions_list(db, data.user_id, fl_row.data, updated_history)

    return {"user_id": data.user_id, "reply": reply, "chat_history_length": len(updated_history)}


@router.post("/functions-list/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_functions_list_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    fl_row = crud.get_functions_list(db, data.user_id)
    if not fl_row:
        raise HTTPException(status_code=409, detail="Functions list not generated yet.")

    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.functions_list_prompt import FUNCTIONS_LIST_CHAT_PROMPT

    current_data = fl_row.data
    user_id      = data.user_id
    context      = "=== CURRENT PRODUCT FUNCTIONS LIST ===\n" + json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": FUNCTIONS_LIST_CHAT_PROMPT},
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
            crud.save_functions_list(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/functions-list/{user_id}", dependencies=[Depends(verify_api_key)])
def get_functions_list(user_id: str, db=Depends(get_db)):
    row = crud.get_functions_list(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No functions list found for this user.")

    return {"user_id": user_id, "functions_list": row.data, "chat_history": row.chat_history or []}