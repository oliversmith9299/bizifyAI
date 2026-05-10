"""
factories/agent_factory.py
===========================
Returns the callable run-function for a given agent name.
The orchestrator uses this instead of hard-coding imports.

All agent names match the keys in crud.AGENT_DEFINITIONS.
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

    # Agents 8-12: return a stub that makes the gap obvious at runtime
    _STUBS = {
        "SevenIdeaStrategy",
        "EightBusinessModel",
        "NineFunctionsList",
        "TenMVPPlanning",
        "ElevenUnitEconomicsAgent",
        "TwelveGoToMarket",
    }
    if agent_name in _STUBS:
        def _not_implemented(*args, **kwargs):
            raise NotImplementedError(
                f"Agent '{agent_name}' is not implemented yet. "
                "Create the agent file and register it in agent_factory.py."
            )
        return _not_implemented

    raise KeyError(f"Unknown agent: '{agent_name}'")
