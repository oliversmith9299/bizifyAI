from fastapi import APIRouter

from routes import (
    service,
    pipeline,
    idea_intake,
    customers,
    competition,
    market_potential,
    idea_strategy,
    business_model,
    functions_list,
    mvp_planning,
    unit_economics,
    go_to_market,
    general_chat,
)

router = APIRouter(prefix="/pipeline", tags=["AI Pipeline"])

router.include_router(service.router)
router.include_router(pipeline.router)
router.include_router(idea_intake.router)
router.include_router(customers.router)
router.include_router(competition.router)
router.include_router(market_potential.router)
router.include_router(idea_strategy.router)
router.include_router(business_model.router)
router.include_router(functions_list.router)
router.include_router(mvp_planning.router)
router.include_router(unit_economics.router)
router.include_router(go_to_market.router)
router.include_router(general_chat.router)