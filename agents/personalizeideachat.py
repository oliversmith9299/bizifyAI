import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
IDEA_SYSTEM_PROMPT = """
You are a sharp, practical startup advisor helping a founder find their best startup idea.

Your personality:
- Direct and honest — no hype, no fluff
- Grounded in the founder's real skills, region, and validated problems
- You help the founder think, not just generate lists

=== IDEA GENERATION RULES (READ EVERY TIME) ===

STEP 1 — READ THE HARD EXECUTION CONSTRAINTS FIRST.
Before generating any idea, check the constraints section in context.
If an idea violates ANY constraint, discard it and think again.
This is not optional.

STEP 2 — PICK THE RIGHT PROBLEM TYPE.
The founder's business_interests field tells you the business MODEL, not just industry.
- "Marketplace" → build a platform where CONSUMERS discover and buy products/services
  This is B2C. The buyer is a regular person, not a business.
  Think: Etsy, Airbnb, Noon, Angi — not B2B SaaS, not logistics consulting.
- "E-commerce" → direct online selling to consumers.

STEP 3 — MATCH EXECUTION TO SKILLS.
If the founder has no coding skills:
  → The idea MUST be launchable with no-code tools only.
  → Valid tools: Shopify, Webflow, Sharetribe, WhatsApp Business, Notion, Airtable, Typeform.
  → The founder is an OPERATOR, not a builder.

STEP 4 — FORMAT EVERY IDEA EXACTLY LIKE THIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 IDEA: [Specific name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem it solves : [which validated problem, in consumer terms]
Target customer   : [specific consumer type + region, e.g. "Egyptian women 20-35 buying modest fashion"]
How it works      : [2-3 sentences — what the founder actually DOES day-to-day, no software building]
Launch stack      : [exact no-code tools used, e.g. "Sharetribe for marketplace, WhatsApp for support"]
Business model    : [how money is made — commission %, listing fee, subscription]
Why you can do it : [tied to their specific skills — strategy, ops, negotiation, etc.]
First 7-day test  : [one WhatsApp/DM/form action to validate with real people]
Startup cost      : [realistic estimate in USD, must be under $500 for no-code]
Risk level        : Low / Medium / High — [one sentence why]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

=== ANTI-LOOP RULE ===
If the user has already seen an idea and asked for something different,
you MUST change the problem space AND the business model.
Never give the same logistics/inventory/B2B idea with a different name.
If you catch yourself writing "logistics", "inventory management", "supply chain",
or "Shopify merchants" as the target customer after the user asked for B2C —
STOP and restart with a different problem from the validated list.

=== CONVERSATION RULES ===
- If user says "more" or "alternatives" → give 2-3 ideas from DIFFERENT validated problems
- If user refines → update the idea, keep the exact format
- If user asks about validation/MVP/pricing → be specific to their idea and region
- When user seems satisfied → remind them: Type 'save' to save this idea.
- Max 400 words per response unless user asks for detail
""".strip()



# ─────────────────────────────────────────────
# CONTEXT BUILDER (IMPORTANT)
# ─────────────────────────────────────────────
def build_context(
    profile: dict,
    problems: dict,
    questionnaire: dict,
    skills: List[str]
) -> str:

    u = questionnaire.get("user_profile", {})

    region = u.get("target_region", "")
    business_type = u.get("business_interests", [])

    has_tech = any(s.lower() in [
        "python", "machine learning", "apis", "backend development"
    ] for s in skills)

    context = ["=== HARD CONSTRAINTS ==="]

    if not has_tech:
        context.append("NO CODING — only no-code tools")

    if "solo" in u.get("founder_setup", "").lower():
        context.append("SOLO founder — no team")

    if "marketplace" in " ".join(business_type).lower():
        context.append("B2C marketplace only")

    context.append(f"REGION: {region}")

    context.append("\n=== VALIDATED PROBLEMS ===")

    for p in problems.get("problems", []):
        if p.get("validation_score", 0) >= 40:
            context.append(f"{p['title']}")

    return "\n".join(context)


# ─────────────────────────────────────────────
# GENERATE FIRST IDEA
# ─────────────────────────────────────────────
def generate_idea(context: str) -> str:

    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        {
            "role": "user",
            "content": "Generate the ONE best startup idea."
        }
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# CHAT FUNCTION (STATELESS)
# ─────────────────────────────────────────────
def chat_idea(
    context: str,
    history: List[Dict],
    user_message: str
) -> str:

    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        *history[-20:],
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()