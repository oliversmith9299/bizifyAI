import os
from typing import Dict, List

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
- Grounded in the founder's real curiosity domain, skills, region, and validated problems
- You help the founder think, not just generate lists

=== IDEA GENERATION RULES (READ EVERY TIME) ===

STEP 1 — READ THE HARD EXECUTION CONSTRAINTS FIRST.
Before generating any idea, check the constraints section in context.
If an idea violates ANY constraint, discard it and think again.
This is not optional.

STEP 2 — STAY IN THE FOUNDER'S CURIOSITY DOMAIN.
The context includes a FOUNDER NICHE / CURIOSITY DOMAIN field.
ALL ideas MUST operate in or around that domain.
- If the domain is "Art & Design" → ideas must involve art, design, creativity, or visual culture
- If the domain is "Technology" → ideas must involve tech products or digital tools
- NEVER suggest an idea outside the stated domain unless the user explicitly asks

STEP 3 — PICK THE RIGHT PROBLEM TYPE.
The founder's business_interests field tells you the business MODEL, not just industry.
- "Marketplace" → build a platform where CONSUMERS discover and buy products/services
  This is B2C. The buyer is a regular person, not a business.
  Think: Etsy, Society6, Redbubble — not B2B SaaS, not seller tools, not logistics.
- "E-commerce" → direct online selling to consumers.

STEP 4 — MATCH EXECUTION TO SKILLS.
If the founder has no coding skills:
  → The idea MUST be launchable with no-code tools only.
  → Valid tools: Shopify, Webflow, Sharetribe, WhatsApp Business, Notion, Airtable, Typeform.
  → The founder is an OPERATOR, not a builder.

STEP 5 — FORMAT EVERY IDEA EXACTLY LIKE THIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 IDEA: [Specific name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem it solves : [which validated consumer problem in the curiosity domain]
Target customer   : [specific consumer type + region, e.g. "Art lovers globally buying indie prints"]
How it works      : [2-3 sentences — what the founder actually DOES day-to-day, no software building]
Launch stack      : [exact no-code tools used]
Business model    : [how money is made — commission %, listing fee, subscription]
Why you can do it : [tied to their specific strengths — creativity, ops, co-founder leverage]
First 7-day test  : [one WhatsApp/DM/form action to validate with real people]
Startup cost      : [realistic estimate in USD, must be under $500 for no-code]
Risk level        : Low / Medium / High — [one sentence why]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

=== ANTI-LOOP RULE ===
If the user has already seen an idea and asked for something different,
you MUST change the problem space AND the business model.
Never give the same logistics/inventory/B2B/seller-tooling idea with a different name.
If you catch yourself writing about "sellers", "inventory", "pricing tools",
or any B2B problem after the user asked for B2C — STOP and restart.

=== CONVERSATION RULES ===
- If user says "more" or "alternatives" → give 2-3 ideas from DIFFERENT validated problems
- If user refines → update the idea, keep the exact format
- If user asks about validation/MVP/pricing → be specific to their idea and region
- When user seems satisfied → remind them: Type 'save' to save this idea.
- Max 400 words per response unless user asks for detail
""".strip()


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_context(
    profile: dict,
    problems: dict,
    questionnaire: dict,
    skills: List[str]
) -> str:
    u = questionnaire.get("user_profile", {})
    career = questionnaire.get("career_profile", {})

    region = u.get("target_region", "")
    business_type = u.get("business_interests", [])
    curiosity_domain = u.get("curiosity_domain", "")  # FIX: primary niche
    setup = u.get("founder_setup", "")
    risk = u.get("risk_tolerance", "")

    has_tech = any(s.lower() in [
        "python", "machine learning", "apis", "backend development",
        "software development", "coding"
    ] for s in skills)

    is_solo = "solo" in setup.lower()
    is_cofounder = "co-founder" in setup.lower() or "partner" in setup.lower()
    is_marketplace = "marketplace" in " ".join(business_type).lower()
    low_capital = "low" in risk.lower() or "minimal" in risk.lower()

    context = ["=== ⚠️ HARD EXECUTION CONSTRAINTS ==="]

    if not has_tech:
        context += [
            "❌ NO SOFTWARE BUILDING — no coding skills",
            "✅ ALLOWED: Sharetribe, Webflow, Notion, Airtable, WhatsApp Business",
        ]

    if is_solo:
        context.append("❌ SOLO FOUNDER — no team, laptop only")
    if is_cofounder:
        context.append("✅ CO-FOUNDER — can split creative and ops roles")

    if low_capital:
        context.append("❌ LOW CAPITAL — under $300 to launch")
    else:
        context.append("✅ MODERATE RISK — calculated risks ok, under $500")

    if is_marketplace:
        context += [
            "✅ B2C MARKETPLACE ONLY — consumers as end users",
            "❌ FORBIDDEN: B2B SaaS, seller tools, logistics, inventory, consulting",
        ]

    context.append(f"✅ TARGET REGION: {region}")

    # FIX: Inject curiosity domain as the primary niche signal
    if curiosity_domain:
        context.append(f"✅ FOUNDER NICHE / CURIOSITY DOMAIN: {curiosity_domain}")
        context.append(f"   → ALL ideas MUST be in the {curiosity_domain} space")

    # FIX: Inject career profile for richer idea alignment
    desired_impact = career.get("desired_impact", [])
    work_types = career.get("preferred_work_types", [])
    if desired_impact:
        context.append(f"✅ FOUNDER WANTS TO: {', '.join(desired_impact)}")
    if work_types:
        context.append(f"✅ WORKS BEST WITH: {', '.join(work_types)}")

    context.append("\n=== VALIDATED PROBLEMS ===")
    for p in problems.get("problems", []):
        if p.get("validation_score", 0) >= 35:
            context.append(f"[{p['id']}] {p['title']}")
            context.append(f"  Target customer : {p.get('target_customer', '')}")
            context.append(f"  Gap opportunity : {p.get('gap_opportunity', '')}")

    return "\n".join(context)


# ─────────────────────────────────────────────
# GENERATE FIRST IDEA
# ─────────────────────────────────────────────
def generate_idea(
    context: str,
    curiosity_domain: str = "",
    founder_setup: str = "",
    business_interests: List[str] = None,
) -> str:
    """
    Generate the first idea with a domain-specific opening prompt.
    """
    if business_interests is None:
        business_interests = []

    is_marketplace = any("marketplace" in b.lower() for b in business_interests)
    model_type = "B2C marketplace" if is_marketplace else "e-commerce"
    setup_note = (
        "They have a co-founder, so can split creative and ops roles."
        if "co-founder" in founder_setup.lower() or "partner" in founder_setup.lower()
        else "They are a solo founder."
    )
    domain_note = f"Their curiosity domain is {curiosity_domain}." if curiosity_domain else ""

    opening = (
        f"Generate the ONE best startup idea for this founder. "
        f"{domain_note} {setup_note} "
        f"Business model: {model_type}. "
        f"No coding skills — no-code tools only. "
        f"Use the exact structured format. "
        f"The idea MUST be in the {curiosity_domain or 'stated'} niche."
    )

    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        {"role": "user", "content": opening},
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
    history: List[dict],
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