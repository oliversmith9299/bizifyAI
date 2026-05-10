"""
services/base_service.py
========================
Abstract base for all bizifyAI services.

A service sits between a route and the orchestrator/agents.
It validates inputs, calls the orchestrator, and translates results
into the response shape the route needs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    """
    Override handle() in each concrete service.
    Routes should call service.handle() — not the orchestrator directly.
    """

    service_name: str = ""

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(self.service_name or self.__class__.__name__)

    @abstractmethod
    def handle(self, *args: Any, **kwargs: Any) -> dict:
        """Process the request and return a serialisable result dict."""
