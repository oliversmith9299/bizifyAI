"""
agents/PipelineRunner.py
========================
Backward-compatibility shim.

All agent logic now lives in dedicated agent files consistent with steps 4-12:

  Step 1 → OneProfileAnalysis.py       run_profile_analysis()
  Step 2 → TwoProblemDiscovery.py      run_problem_discovery()
  Step 3 → ThreePersonalizeIdeaChat.py build_context(), generate_opening_idea(),
                                        generate_idea(), chat_idea(), IDEA_SYSTEM_PROMPT

This file re-exports everything so existing callers (orchestrator, generalBot,
routes/pipeline.py) continue to work without any changes.
"""

from agents.OneProfileAnalysis import run_profile_analysis          # noqa: F401
from agents.TwoProblemDiscovery import run_problem_discovery        # noqa: F401
from agents.ThreePersonalizeIdeaChat import (                       # noqa: F401
    build_context,
    generate_opening_idea,
    generate_idea,
    chat_idea,
    IDEA_SYSTEM_PROMPT,
)

# Re-export the Groq client and model that routes/pipeline.py imports from here
from agents.config import client as groq_client, GROQ_MODEL         # noqa: F401
