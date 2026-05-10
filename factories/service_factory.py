"""
factories/service_factory.py
=============================
Single place to get any service instance.
Routes import get_service() instead of instantiating services directly.

Usage:
    from factories.service_factory import get_service
    svc = get_service("pipeline")
    result = svc.handle_start(user_id, questionnaire, skills)
"""

from services.pipeline_service import IdeaIntakeService, PipelineService

_REGISTRY = {
    "pipeline":     PipelineService,
    "idea_intake":  IdeaIntakeService,
}


def get_service(name: str):
    """Return a fresh service instance by name."""
    cls = _REGISTRY.get(name)
    if not cls:
        raise KeyError(f"No service registered under '{name}'. "
                       f"Available: {list(_REGISTRY)}")
    return cls()
