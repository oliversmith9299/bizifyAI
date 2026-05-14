"""
agents/schemas.py
=================
Pydantic output schemas for every pipeline agent.

Purpose
-------
Every agent calls parse_llm_json() which returns a plain untyped dict.
These schemas coerce that dict to validated, typed data before it is
saved to the database.

Design decisions
----------------
- All fields are Optional with sensible defaults so a partially-valid
  LLM response is still usable (graceful degradation, not hard failure).
- extra="ignore" — LLM output often contains extra keys; we discard them.
- AliasChoices — some LLM responses use different key names than the DB
  schema expects (e.g. "pestel_analysis" vs "pestel", "catwoe_analysis"
  vs "catwoe"). The alias layer normalises these transparently.
- model_dump() — callers always receive a plain dict so no downstream
  code needs to change.
"""

import logging
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = ConfigDict(
        extra="ignore",          # silently drop unknown keys from LLM output
        populate_by_name=True,   # allow field name OR alias when building
        coerce_numbers_to_str=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — FourCustomersAgent
# ─────────────────────────────────────────────────────────────────────────────

class CustomersOutput(_Base):
    customer_segments:     list[dict[str, Any]] = Field(default_factory=list)
    primary_segment:       dict[str, Any]        = Field(default_factory=dict)
    # LLM sometimes returns "catwoe_analysis"; normalise to "catwoe"
    catwoe:                dict[str, Any]        = Field(
        default_factory=dict,
        validation_alias=AliasChoices("catwoe", "catwoe_analysis"),
    )
    personas:              list[dict[str, Any]] = Field(default_factory=list)
    acquisition_channels:  list[Any]            = Field(default_factory=list)
    early_adopter_profile: str                  = ""
    summary:               str                  = ""
    source_mode:           str                  = ""
    sources_used:          int                  = 0
    sources_list:          list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — FiveCompetitionAgent
# ─────────────────────────────────────────────────────────────────────────────

class CompetitionOutput(_Base):
    direct_competitors:            list[dict[str, Any]] = Field(default_factory=list)
    indirect_alternatives:         list[dict[str, Any]] = Field(default_factory=list)
    substitute_solutions:          list[Any]            = Field(default_factory=list)
    positioning_gaps:              list[dict[str, Any]] = Field(default_factory=list)
    porters_five_forces:           dict[str, Any]       = Field(default_factory=dict)
    vrio_analysis:                 list[dict[str, Any]] = Field(default_factory=list)
    differentiation_opportunities: list[Any]            = Field(default_factory=list)
    summary:                       str                  = ""
    source_mode:                   str                  = ""
    sources_used:                  int                  = 0
    sources_list:                  list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — SixMarketPotential
# ─────────────────────────────────────────────────────────────────────────────

class MarketPotentialOutput(_Base):
    market_definition:          str           = ""
    target_region:              str           = ""
    tam:                        dict[str, Any] = Field(default_factory=dict)
    sam:                        dict[str, Any] = Field(default_factory=dict)
    som:                        dict[str, Any] = Field(default_factory=dict)
    market_trends:              list[Any]      = Field(default_factory=list)
    growth_drivers:             list[Any]      = Field(default_factory=list)
    adoption_barriers:          list[Any]      = Field(default_factory=list)
    timing_assessment:          dict[str, Any] = Field(default_factory=dict)
    # LLM returns "pestel_analysis"; DB column is "pestel"
    pestel:                     dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("pestel", "pestel_analysis"),
    )
    opportunity_score:          Optional[float] = 0.0
    opportunity_attractiveness: str             = ""
    summary:                    str             = ""
    source_mode:                str             = ""
    sources_used:               int             = 0
    sources_list:               list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 — SevenIdeaStrategy
# ─────────────────────────────────────────────────────────────────────────────

class IdeaStrategyOutput(_Base):
    value_proposition:        dict[str, Any]       = Field(default_factory=dict)
    positioning:              dict[str, Any]        = Field(default_factory=dict)
    core_promise:             str                   = ""
    differentiation_strategy: dict[str, Any]        = Field(default_factory=dict)
    key_assumptions:          list[dict[str, Any]]  = Field(default_factory=list)
    validation_priorities:    list[dict[str, Any]]  = Field(default_factory=list)
    strategic_direction:      dict[str, Any]        = Field(default_factory=dict)
    unfair_advantages:        list[Any]             = Field(default_factory=list)
    strategic_risks:          list[dict[str, Any]]  = Field(default_factory=list)
    summary:                  str                   = ""
    source_mode:              str                   = ""
    sources_used:             int                   = 0
    sources_list:             list[dict[str, Any]]  = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 — EightBusinessModel
# ─────────────────────────────────────────────────────────────────────────────

class BusinessModelOutput(_Base):
    business_model_type:    str                   = ""
    business_model_canvas:  dict[str, Any]         = Field(default_factory=dict)
    revenue_streams:        list[dict[str, Any]]   = Field(default_factory=list)
    pricing_strategy:       dict[str, Any]         = Field(default_factory=dict)
    key_metrics:            list[Any]              = Field(default_factory=list)
    business_model_risks:   list[dict[str, Any]]   = Field(default_factory=list)
    founder_fit_assessment: dict[str, Any]         = Field(default_factory=dict)
    summary:                str                    = ""
    source_mode:            str                    = ""
    sources_used:           int                    = 0
    sources_list:           list[dict[str, Any]]   = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 9 — NineFunctionsList
# ─────────────────────────────────────────────────────────────────────────────

class FunctionsListOutput(_Base):
    product_type:                   str                   = ""
    core_functions:                 list[dict[str, Any]]  = Field(default_factory=list)
    nice_to_have_functions:         list[dict[str, Any]]  = Field(default_factory=list)
    future_capabilities:            list[dict[str, Any]]  = Field(default_factory=list)
    feature_creep_warnings:         list[Any]             = Field(default_factory=list)
    function_to_pain_map:           list[dict[str, Any]]  = Field(default_factory=list)
    function_to_business_model_map: list[dict[str, Any]]  = Field(default_factory=list)
    no_code_stack:                  list[dict[str, Any]]  = Field(default_factory=list)
    summary:                        str                   = ""
    source_mode:                    str                   = ""
    sources_used:                   int                   = 0
    sources_list:                   list[dict[str, Any]]  = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 10 — TenMVPPlanning
# ─────────────────────────────────────────────────────────────────────────────

class MVPPlanningOutput(_Base):
    mvp_goal:               str                   = ""
    riskiest_assumptions:   list[dict[str, Any]]  = Field(default_factory=list)
    scope:                  dict[str, Any]         = Field(default_factory=dict)
    core_user_flows:        list[dict[str, Any]]  = Field(default_factory=list)
    build_plan:             dict[str, Any]         = Field(default_factory=dict)
    validation_experiments: list[dict[str, Any]]  = Field(default_factory=list)
    launch_criteria:        dict[str, Any]         = Field(default_factory=dict)
    testing_plan:           list[dict[str, Any]]  = Field(default_factory=list)
    qa_checklist:           list[Any]             = Field(default_factory=list)
    first_100_users_plan:   str                   = ""
    summary:                str                   = ""
    source_mode:            str                   = ""
    sources_used:           int                   = 0
    sources_list:           list[dict[str, Any]]  = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 11 — ElevenUnitEconomics
# ─────────────────────────────────────────────────────────────────────────────

class UnitEconomicsOutput(_Base):
    revenue_model_summary: str                   = ""
    pricing_assumptions:   dict[str, Any]         = Field(default_factory=dict)
    cost_assumptions:      dict[str, Any]         = Field(default_factory=dict)
    gross_margin:          dict[str, Any]         = Field(default_factory=dict)
    cac_analysis:          dict[str, Any]         = Field(default_factory=dict)
    ltv_analysis:          dict[str, Any]         = Field(default_factory=dict)
    ltv_cac_ratio:         dict[str, Any]         = Field(default_factory=dict)
    payback_period:        dict[str, Any]         = Field(default_factory=dict)
    break_even:            dict[str, Any]         = Field(default_factory=dict)
    monthly_projections:   list[dict[str, Any]]  = Field(default_factory=list)
    weak_assumptions:      list[dict[str, Any]]  = Field(default_factory=list)
    pricing_tests:         list[dict[str, Any]]  = Field(default_factory=list)
    overall_viability:     dict[str, Any]         = Field(default_factory=dict)
    summary:               str                   = ""
    source_mode:           str                   = ""
    sources_used:          int                   = 0
    sources_list:          list[dict[str, Any]]  = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 12 — TwelveGoToMarket
# ─────────────────────────────────────────────────────────────────────────────

class GoToMarketOutput(_Base):
    target_launch_segment:    dict[str, Any]        = Field(default_factory=dict)
    positioning_message:      dict[str, Any]        = Field(default_factory=dict)
    marketing_channels:       list[dict[str, Any]]  = Field(default_factory=list)
    funnel_stages:            list[dict[str, Any]]  = Field(default_factory=list)
    launch_experiments:       list[dict[str, Any]]  = Field(default_factory=list)
    first_100_customers_plan: dict[str, Any]        = Field(default_factory=dict)
    launch_timeline:          list[dict[str, Any]]  = Field(default_factory=list)
    success_metrics:          dict[str, Any]        = Field(default_factory=dict)
    cac_tracking:             dict[str, Any]        = Field(default_factory=dict)
    feedback_loops:           list[dict[str, Any]]  = Field(default_factory=list)
    summary:                  str                   = ""
    source_mode:              str                   = ""
    sources_used:             int                   = 0
    sources_list:             list[dict[str, Any]]  = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Registry + validation entry point
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_MAP: dict[str, type[_Base]] = {
    "customers":        CustomersOutput,
    "competition":      CompetitionOutput,
    "market_potential": MarketPotentialOutput,
    "idea_strategy":    IdeaStrategyOutput,
    "business_model":   BusinessModelOutput,
    "functions_list":   FunctionsListOutput,
    "mvp_planning":     MVPPlanningOutput,
    "unit_economics":   UnitEconomicsOutput,
    "go_to_market":     GoToMarketOutput,
}


def validate_section_output(section: str, raw: dict) -> dict:
    """
    Coerce and validate a raw agent output dict against the section's schema.

    - Fills missing fields with their defaults.
    - Normalises field name aliases (e.g. pestel_analysis → pestel).
    - Strips unknown keys returned by the LLM.
    - On validation failure logs a warning and returns the original dict
      so the pipeline is never hard-blocked by a schema mismatch.
    """
    schema_cls = _SCHEMA_MAP.get(section)
    if schema_cls is None:
        return raw
    try:
        return schema_cls.model_validate(raw).model_dump(by_alias=False)
    except Exception as exc:
        log.warning(
            "[validate_section_output] section=%s validation warning: %s",
            section, exc,
        )
        return raw
