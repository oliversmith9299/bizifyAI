# ThreeIdeaIntakeAgent
# =================
#
# Returning-user flow only.
# User describes their raw idea → this agent structures it → ProblemDiscovery researches it.
#
# Multi-turn: if the idea is too vague, the agent asks 2-3 short clarification
# questions and waits for answers before producing the structured output.

import os

from dotenv import load_dotenv
from openai import OpenAI

from db.connection import SessionLocal
from db import crud
from agents.utils import parse_llm_json

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """
You are a startup analyst helping a founder structure their raw idea.

Your job: read what the founder tells you and decide one of two things:

OPTION A — READY
The idea has enough information to extract all key fields.
An idea is ready if you know: what it does, who it's for, and what problem it solves.
Make reasonable assumptions for anything not stated (region → "Global", business_model → infer from idea).

OPTION B — NEEDS_CLARIFICATION
The idea is too vague to extract even a basic summary.
Only choose this if at least 2 of the 3 critical fields (what, who, problem) are completely unknown.
Ask a maximum of 3 short, specific questions. Never ask more than 3.

=== OUTPUT FORMAT (STRICT — return ONLY valid JSON, nothing else) ===

If READY:
{
  "decision": "ready",
  "intake": {
    "idea_summary": "One sentence describing what the product/service does",
    "target_users": ["specific user type 1", "specific user type 2"],
    "industry": "e.g. EdTech, HealthTech, Marketplace, SaaS, etc.",
    "problem_assumption": "The core pain point this idea addresses",
    "solution_assumption": "How the idea solves the problem",
    "business_model": "e.g. Subscription, Commission, Freemium, One-time purchase",
    "region": "Target market region or Global",
    "keywords_for_problem_discovery": [
      "4 to 7 specific search terms that reveal real pain points around this idea"
    ],
    "unclear_questions": []
  },
  "reply": "A 1-2 sentence confirmation message telling the user their idea has been understood and you are now researching real problems around it."
}

If NEEDS_CLARIFICATION:
{
  "decision": "needs_clarification",
  "partial_intake": {
    "idea_summary": "fill what you know, leave empty string if unknown",
    "target_users": [],
    "industry": "",
    "problem_assumption": "",
    "solution_assumption": "",
    "business_model": "",
    "region": "Global",
    "keywords_for_problem_discovery": [],
    "unclear_questions": ["question 1", "question 2"]
  },
  "reply": "A friendly message with 2-3 short clarification questions (inline, not as a list)."
}

RULES:
- Return ONLY valid JSON — no markdown, no explanation outside the JSON
- keywords_for_problem_discovery must be 4-7 specific search phrases, not single words
  Good: "study partner apps student engagement problems"
  Bad:  "education"
- Be generous with assumptions — prefer "ready" over asking questions unless truly lost
- If the user answers your clarification questions, use their answers to produce the "ready" output
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_profile_for_problem_discovery(intake: dict) -> dict:
    """Map IdeaIntake → the profile shape that run_problem_discovery expects."""
    return {
        "search_direction": {
            "keywords": intake.get("keywords_for_problem_discovery", [])
        },
        "recommended_industries":    [intake.get("industry", "")],
        "recommended_problem_spaces": [intake.get("problem_assumption", "")],
        "personality_insights": {"strengths": []},
        "founder_profile":      {"key_skill_gaps": []},
    }


def _build_questionnaire_for_problem_discovery(intake: dict) -> dict:
    """Map IdeaIntake → the questionnaire shape that run_problem_discovery expects."""
    return {
        "user_profile": {
            "target_region":      intake.get("region", "Global"),
            "business_interests": [intake.get("business_model", "")],
            "curiosity_domain":   intake.get("industry", ""),
            "founder_setup":      "solo",
            "risk_tolerance":     "medium",
            "experience_level":   "intermediate",
        },
        "career_profile": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────
def run_idea_intake(user_id: str, user_message: str, history: list = None) -> dict:
    """
    Process a raw idea message (or clarification answer) and return structured data.

    Parameters
    ----------
    user_id      : str  — the founder's user ID (for DB persistence)
    user_message : str  — raw idea text or answer to clarification questions
    history      : list — prior turns [{"role": ..., "content": ...}]
                          Pass the history returned by a previous needs_clarification call.

    Returns
    -------
    Ready:
      { "status": "ready", "intake": {...}, "reply": "...",
        "profile_for_problem_discovery": {...},
        "questionnaire_for_problem_discovery": {...} }

    Needs clarification:
      { "status": "needs_clarification", "reply": "...",
        "questions": [...], "history": [...] }
    """
    if history is None:
        history = []

    db = SessionLocal()
    try:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )

        raw    = response.choices[0].message.content
        result = parse_llm_json(raw)

        updated_history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": raw},
        ]

        if result.get("decision") == "ready":
            intake               = result["intake"]
            profile_compat       = _build_profile_for_problem_discovery(intake)
            questionnaire_compat = _build_questionnaire_for_problem_discovery(intake)

            crud.save_idea_intake(db, user_id, intake)
            print(f"[IdeaIntakeAgent] Intake saved for user {user_id}")

            return {
                "status": "ready",
                "intake": intake,
                "reply":  result.get("reply", "Your idea has been structured. Researching real problems now..."),
                "profile_for_problem_discovery":       profile_compat,
                "questionnaire_for_problem_discovery": questionnaire_compat,
            }

        else:
            partial = result.get("partial_intake", {})
            crud.save_idea_intake(db, user_id, {
                **partial,
                "_status":  "pending_clarification",
                "_history": updated_history,
            })

            return {
                "status":    "needs_clarification",
                "reply":     result.get("reply", ""),
                "questions": partial.get("unclear_questions", []),
                "history":   updated_history,
            }

    except Exception as e:
        print(f"[IdeaIntakeAgent] Error: {e}")
        raise

    finally:
        db.close()
