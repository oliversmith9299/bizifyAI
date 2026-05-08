"""
orchestrator/session.py
=======================
Pipeline state machine for bizifyAI.

Tracks which step the user is on and which agent to run next.
Both flows (new-user and returning-user) share the same steps after the
idea is saved, so the router only diverges at the beginning.

Usage
-----
    session = PipelineSession(user_id="abc", flow="new_user")
    while not session.is_done():
        step = session.current_step
        # run the agent for `step`, pass result to session
        session.advance(result)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Step registry
# ─────────────────────────────────────────────────────────────────────────────

class Step(str, Enum):
    # New-user only
    PROFILE_ANALYSIS  = "profile_analysis"
    # Returning-user only
    IDEA_INTAKE       = "idea_intake"
    # Shared from here on
    PROBLEM_DISCOVERY = "problem_discovery"
    IDEA_CHAT         = "idea_chat"
    CUSTOMERS         = "customers"
    COMPETITION       = "competition"
    MARKET_POTENTIAL  = "market_potential"
    IDEA_STRATEGY     = "idea_strategy"
    BUSINESS_MODEL    = "business_model"
    FUNCTIONS_LIST    = "functions_list"
    MVP_PLANNING      = "mvp_planning"
    UNIT_ECONOMICS    = "unit_economics"
    GO_TO_MARKET      = "go_to_market"
    DONE              = "done"


class Flow(str, Enum):
    NEW_USER       = "new_user"
    RETURNING_USER = "returning_user"


# Ordered step sequence for each flow
_NEW_USER_STEPS: list[Step] = [
    Step.PROFILE_ANALYSIS,
    Step.PROBLEM_DISCOVERY,
    Step.IDEA_CHAT,
    Step.CUSTOMERS,
    Step.COMPETITION,
    Step.MARKET_POTENTIAL,
    Step.IDEA_STRATEGY,
    Step.BUSINESS_MODEL,
    Step.FUNCTIONS_LIST,
    Step.MVP_PLANNING,
    Step.UNIT_ECONOMICS,
    Step.GO_TO_MARKET,
    Step.DONE,
]

_RETURNING_USER_STEPS: list[Step] = [
    Step.IDEA_INTAKE,
    Step.PROBLEM_DISCOVERY,
    Step.IDEA_CHAT,
    Step.CUSTOMERS,
    Step.COMPETITION,
    Step.MARKET_POTENTIAL,
    Step.IDEA_STRATEGY,
    Step.BUSINESS_MODEL,
    Step.FUNCTIONS_LIST,
    Step.MVP_PLANNING,
    Step.UNIT_ECONOMICS,
    Step.GO_TO_MARKET,
    Step.DONE,
]

# Which agent file/function handles each step
STEP_AGENT_MAP: Dict[Step, str] = {
    Step.PROFILE_ANALYSIS:  "agents.PipelineRunner.run_profile_analysis",
    Step.IDEA_INTAKE:       "agents.ThreeIdeaIntakeAgent.run_idea_intake",
    Step.PROBLEM_DISCOVERY: "agents.PipelineRunner.run_problem_discovery",
    Step.IDEA_CHAT:         "agents.PipelineRunner.generate_opening_idea",
    Step.CUSTOMERS:         "agents.FourCustomersAgent",
    Step.COMPETITION:       "agents.FiveCompetitionAgent",
    Step.MARKET_POTENTIAL:  "agents.SixMaketPotential",
    Step.IDEA_STRATEGY:     "agents.SevenIdeaStrategy",
    Step.BUSINESS_MODEL:    "agents.EightBusinessModel",
    Step.FUNCTIONS_LIST:    "agents.NineFunctionsList",
    Step.MVP_PLANNING:      "agents.TenMVPPlanning",
    Step.UNIT_ECONOMICS:    "agents.ElevenUnitEconomicsAgent",
    Step.GO_TO_MARKET:      "agents.TwelveGoToMarket",
}

# Human-readable labels shown in status responses and logs
STEP_LABELS: Dict[Step, str] = {
    Step.PROFILE_ANALYSIS:  "Profile Analysis",
    Step.IDEA_INTAKE:       "Idea Intake",
    Step.PROBLEM_DISCOVERY: "Problem Discovery",
    Step.IDEA_CHAT:         "Idea Generation & Chat",
    Step.CUSTOMERS:         "Customer Analysis",
    Step.COMPETITION:       "Competition Analysis",
    Step.MARKET_POTENTIAL:  "Market Potential",
    Step.IDEA_STRATEGY:     "Idea Strategy",
    Step.BUSINESS_MODEL:    "Business Model",
    Step.FUNCTIONS_LIST:    "Product Functions List",
    Step.MVP_PLANNING:      "MVP Planning",
    Step.UNIT_ECONOMICS:    "Unit Economics",
    Step.GO_TO_MARKET:      "Go-to-Market Plan",
    Step.DONE:              "Done",
}


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineSession:
    """
    Tracks the state of one user's pipeline run.

    Attributes
    ----------
    user_id      : Unique user identifier.
    flow         : "new_user" or "returning_user".
    current_step : The step the session is currently on.
    results      : Accumulated agent outputs keyed by step name.
    error        : Last error message, if any.
    """

    user_id:      str
    flow:         Flow
    current_step: Step = field(init=False)
    results:      Dict[str, Any] = field(default_factory=dict)
    error:        Optional[str] = field(default=None)

    def __post_init__(self):
        steps = self._steps()
        self.current_step = steps[0]

    # ── Navigation ────────────────────────────────────────────────────────────

    def _steps(self) -> list[Step]:
        return (
            _NEW_USER_STEPS
            if self.flow == Flow.NEW_USER
            else _RETURNING_USER_STEPS
        )

    def advance(self, result: Any = None) -> "PipelineSession":
        """Save the current step's result and move to the next step."""
        if result is not None:
            self.results[self.current_step.value] = result

        steps = self._steps()
        idx = steps.index(self.current_step)
        if idx + 1 < len(steps):
            self.current_step = steps[idx + 1]
        return self

    def go_to(self, step: Step) -> "PipelineSession":
        """Jump to a specific step (e.g. after a re-run or recovery)."""
        if step not in self._steps():
            raise ValueError(f"Step {step!r} is not valid for flow {self.flow!r}")
        self.current_step = step
        return self

    def fail(self, message: str) -> "PipelineSession":
        """Mark the session as failed without advancing."""
        self.error = message
        return self

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self.current_step == Step.DONE

    def is_failed(self) -> bool:
        return self.error is not None

    @property
    def step_label(self) -> str:
        return STEP_LABELS.get(self.current_step, self.current_step.value)

    @property
    def agent(self) -> Optional[str]:
        """Dotted path to the agent function for the current step."""
        return STEP_AGENT_MAP.get(self.current_step)

    def remaining_steps(self) -> list[Step]:
        steps = self._steps()
        idx = steps.index(self.current_step)
        return steps[idx:]

    # ── Serialisation (for DB storage) ────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id":      self.user_id,
            "flow":         self.flow.value,
            "current_step": self.current_step.value,
            "error":        self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineSession":
        session = cls(
            user_id=data["user_id"],
            flow=Flow(data["flow"]),
        )
        session.current_step = Step(data["current_step"])
        session.error        = data.get("error")
        return session


# ─────────────────────────────────────────────────────────────────────────────
# Router helper
# ─────────────────────────────────────────────────────────────────────────────

def make_session(user_id: str, has_idea: bool) -> PipelineSession:
    """
    Factory used by the API routes to create the right session type.

    Parameters
    ----------
    user_id  : the user's ID
    has_idea : True if the user is in the returning-user flow (already has an idea)
    """
    flow = Flow.RETURNING_USER if has_idea else Flow.NEW_USER
    return PipelineSession(user_id=user_id, flow=flow)
