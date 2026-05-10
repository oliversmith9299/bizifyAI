"""
agents/base_agent.py
====================
Abstract contract that every agent in the pipeline must satisfy.
Concrete agents inherit from BaseAgent and implement run().
"""

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Shared interface for all bizifyAI pipeline agents.

    Attributes
    ----------
    name  : Registry name — must match the key in AGENT_DEFINITIONS in crud.py
            (e.g. "OneProfileAnalysis", "TwoProblemDiscovery")
    phase : Pipeline phase label (discovery / ideation / planning / strategy /
            business / product / finance / launch)
    """

    name:  str = ""
    phase: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(f"{cls.__name__} must define a non-empty class attribute 'name'")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute the agent and return a structured dict.
        The dict shape must match the agent's JSON output schema in data/.
        """

    # ── Logging helper ────────────────────────────────────────────────────────
    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(self.name)
