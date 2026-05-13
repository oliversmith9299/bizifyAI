"""
factories/agent_factory.py
===========================
Returns the callable run-function for a given agent name.
The orchestrator uses this instead of hard-coding imports.

Agent names correspond to their dedicated result tables in the database.
"""

from typing import Callable


def get_agent_runner(agent_name: str) -> Callable:
    """
    Return the main run-function for the named agent.

    Example
    -------
    run = get_agent_runner("OneProfileAnalysis")
    result = run(questionnaire, skills)
    """
    if agent_name == "OneProfileAnalysis":
        from agents.PipelineRunner import run_profile_analysis
        return run_profile_analysis

    if agent_name == "TwoProblemDiscovery":
        from agents.PipelineRunner import run_problem_discovery
        return run_problem_discovery

    if agent_name == "ThreeIdeaIntakeAgent":
        from agents.ThreeIdeaIntakeAgent import run_idea_intake
        return run_idea_intake

    if agent_name == "ThreePersonalizeIdeaChat":
        from agents.PipelineRunner import generate_opening_idea
        return generate_opening_idea

    if agent_name == "FourCustomersAgent":
        from agents.FourCustomersAgent import run_customers_analysis
        return run_customers_analysis

    if agent_name == "FiveCompetitionAgent":
        from agents.FiveCompetitionAgent import run_competition_analysis
        return run_competition_analysis

    if agent_name == "SixMaketPotential":
        from agents.SixMaketPotential import run_market_potential
        return run_market_potential

    if agent_name == "SevenIdeaStrategy":
        from agents.SevenIdeaStrategy import run_idea_strategy
        return run_idea_strategy

    if agent_name == "EightBusinessModel":
        from agents.EightBusinessModel import run_business_model
        return run_business_model

    if agent_name == "NineFunctionsList":
        from agents.NineFunctionsList import run_functions_list
        return run_functions_list

    if agent_name == "TenMVPPlanning":
        from agents.TenMVPPlanning import run_mvp_planning
        return run_mvp_planning

    if agent_name == "ElevenUnitEconomicsAgent":
        from agents.ElevenUnitEconomicsAgent import run_unit_economics
        return run_unit_economics

    if agent_name == "TwelveGoToMarket":
        from agents.TwelveGoToMarket import run_go_to_market
        return run_go_to_market

    raise KeyError(f"Unknown agent: '{agent_name}'")
