"""
#3 Personalized Idea Chat Agent
Pipeline position: ProfileAnalysis → ProblemDiscovery → IdeaAgent (this)

What it does:
1. Loads ALL upstream context: questionnaireOutput, skills, profileAnalysis, problems
2. On startup: generates ONE focused, best-fit startup idea for the user (not a list)
3. Saves that idea to idea_output.json
4. Opens a chat loop where the user can:
   - Ask to see alternative ideas
   - Refine/edit the suggested idea
   - Ask about validation, MVP, pricing, go-to-market
   - Add new constraints ("I want something cheaper to start")
5. Every accepted idea/refinement is saved back to idea_output.json

Inputs  (all from data/):
  - questionnaireOutput.json  → raw user answers
  - skills.json               → user's skills
  - profileAnalysis.json      → LLM-derived founder profile + search keywords
  - problems.json             → validated problems from ProblemDiscovery

Output (data/):
  - idea_output.json          → current best idea + full chat history
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# -------------------------
# Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Input paths (read-only) ──────────────────────────────────────────────────
PATH_QUESTIONNAIRE   = os.path.join(BASE_DIR, "data", "questionnaireOutput.json")
PATH_SKILLS          = os.path.join(BASE_DIR, "data", "skills.json")
PATH_PROFILE         = os.path.join(BASE_DIR, "data", "profileAnalysis.json")
PATH_PROBLEMS        = os.path.join(BASE_DIR, "data", "problems.json")

# ── Output path ──────────────────────────────────────────────────────────────
PATH_IDEA_OUTPUT     = os.path.join(BASE_DIR, "data", "idea_output.json")


# -------------------------
# Data Loader
# -------------------------
def load_json(path: str, label: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"✅ Loaded {label} from {path}")
        return data
    except FileNotFoundError:
        log.warning(f"⚠️  {label} not found at {path} — skipping")
        return None
    except json.JSONDecodeError as e:
        log.warning(f"⚠️  {label} is invalid JSON: {e} — skipping")
        return None


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------------
# Context Builder
# Assembles a clean, token-efficient system context from all 4 input files.
# Each section is clearly labeled so the LLM can reason over them separately.
# -------------------------
def build_system_context(
    questionnaire: Optional[Dict],
    skills: Optional[Dict],
    profile: Optional[Dict],
    problems: Optional[Dict],
) -> str:

    sections = []

    # ── 1. Raw user answers ──────────────────────────────────────────────────
    if questionnaire:
        u = questionnaire.get("user_profile", {})
        c = questionnaire.get("career_profile", {})
        sections.append(f"""
=== FOUNDER QUESTIONNAIRE (raw answers) ===
Curiosity domain    : {u.get('curiosity_domain', 'N/A')}
Experience level    : {u.get('experience_level', 'N/A')}
Business interests  : {', '.join(u.get('business_interests', []))}
Target region       : {u.get('target_region', 'N/A')}
Founder setup       : {u.get('founder_setup', 'N/A')}
Risk tolerance      : {u.get('risk_tolerance', 'N/A')}
Free day preferences: {', '.join(c.get('free_day_preferences', []))}
Preferred work types: {', '.join(c.get('preferred_work_types', []))}
Problem solving     : {', '.join(c.get('problem_solving_styles', []))}
Desired impact      : {', '.join(c.get('desired_impact', []))}
""".strip())

    # ── 2. Skills ────────────────────────────────────────────────────────────
    if skills:
        skill_list = skills.get("skills", [])
        if skill_list:
            sections.append(f"=== FOUNDER SKILLS ===\n{', '.join(skill_list)}")
        else:
            sections.append("=== FOUNDER SKILLS ===\nNo technical skills declared. Recommend operator/business/no-code models only.")

    # ── 3. Profile analysis (LLM-derived insights) ───────────────────────────
    if profile:
        pi = profile.get("personality_insights", {})
        fp = profile.get("founder_profile", {})
        sections.append(f"""
=== FOUNDER PROFILE ANALYSIS ===
Personality type    : {pi.get('type', 'N/A')}
Motivation          : {pi.get('motivation', 'N/A')}
Strengths           : {', '.join(pi.get('strengths', []))}
Weaknesses          : {', '.join(pi.get('weaknesses', []))}
Key skill gaps      : {', '.join(fp.get('key_skill_gaps', []))}
Execution style     : {fp.get('execution_style', 'N/A')}
Risk level          : {fp.get('risk_level', 'N/A')}
Recommended industries    : {', '.join(profile.get('recommended_industries', []))}
Recommended problem spaces: {', '.join(profile.get('recommended_problem_spaces', []))}
""".strip())

    # ── 4. Hard execution constraints (READ BEFORE PROBLEMS) ─────────────────
    # Derived directly from questionnaire + skills — not negotiable
    if questionnaire and skills:
        u = questionnaire.get("user_profile", {})
        fp = profile.get("founder_profile", {}) if profile else {}
        skill_list = skills.get("skills", [])
        has_tech = any(s.lower() in ["python", "machine learning", "apis", "backend development",
                                      "software development", "programming", "coding", "data engineering"]
                       for s in skill_list)
        founder_setup = u.get("founder_setup", "")
        business_interests = u.get("business_interests", [])
        risk = u.get("risk_tolerance", "")
        region = u.get("target_region", "")

        constraint_lines = ["=== ⚠️ HARD EXECUTION CONSTRAINTS (NON-NEGOTIABLE) ===",
                            "These constraints MUST be satisfied before any idea is suggested.",
                            "Violating ANY of these is a critical failure.\n"]

        # Software constraint
        if not has_tech:
            constraint_lines.append("❌ NO SOFTWARE BUILDING: Founder has zero technical/coding skills.")
            constraint_lines.append("   → FORBIDDEN: SaaS platforms, apps, custom software, APIs, tech infrastructure")
            constraint_lines.append("   → ALLOWED   : No-code tools (Shopify, Notion, Airtable, WhatsApp, Webflow, Sharetribe)")
            constraint_lines.append("   → ALLOWED   : Operator models (curate, connect, manage, aggregate, resell)\n")
        else:
            constraint_lines.append("✅ Founder CAN build software — tech ideas are permitted.\n")

        # Solo constraint
        if "solo" in founder_setup.lower():
            constraint_lines.append("❌ NO LARGE TEAM IDEAS: Founder is SOLO with no co-founder.")
            constraint_lines.append("   → FORBIDDEN: Ideas requiring warehouse staff, sales team, dev team, ops team")
            constraint_lines.append("   → ALLOWED   : Ideas one person can run from a laptop in early stage\n")

        # Capital constraint (infer from risk tolerance)
        if "moderate" in risk.lower() or "stability" in risk.lower():
            constraint_lines.append("❌ NO HIGH CAPITAL IDEAS: Founder is moderate-risk, no physical infrastructure.")
            constraint_lines.append("   → FORBIDDEN: Physical warehouse, retail space, logistics fleet, hardware")
            constraint_lines.append("   → ALLOWED   : Lean, digital-first, under $500 to start\n")

        # Business type constraint — MOST IMPORTANT
        if business_interests:
            bi = ", ".join(business_interests)
            constraint_lines.append(f"✅ BUSINESS TYPE INTEREST: {bi}")
            if "marketplace" in bi.lower():
                constraint_lines.append("   → Ideas MUST be marketplace models: connect buyers and sellers,")
                constraint_lines.append("     curated product/service discovery, platform where consumers browse and buy.")
                constraint_lines.append("   → The END USER must be a CONSUMER (B2C), not another business.")
                constraint_lines.append("   → FORBIDDEN: B2B consulting, B2B SaaS, logistics services to merchants")
                constraint_lines.append("   → ALLOWED   : Niche product marketplace, service marketplace, curated discovery platform\n")
            if "e-commerce" in bi.lower():
                constraint_lines.append("   → Focus on direct-to-consumer online selling models.\n")

        # Region
        constraint_lines.append(f"✅ REGION: All ideas must be grounded in {region} market context.")
        constraint_lines.append(f"   → Use {region}-specific platforms, payment methods, consumer behavior.\n")

        constraint_lines.append("SUMMARY — The ideal idea is:")
        constraint_lines.append("  A B2C marketplace the founder OPERATES (not builds from code),")
        constraint_lines.append(f"  targeting consumers in {region}, launchable solo from a laptop under $500,")
        constraint_lines.append("  using no-code tools, earning via commission or listing fees.")

        sections.append("\n".join(constraint_lines))

    # ── 5. Validated problems (filtered through constraints above) ────────────
    if problems:
        prob_list = problems.get("problems", [])
        if prob_list:
            lines = ["=== VALIDATED PROBLEMS FROM RESEARCH ==="]
            lines.append("Use these problems to identify WHAT to solve.")
            lines.append("But the HOW must respect the hard constraints above.")
            lines.append("Do NOT copy the gap_opportunity verbatim if it violates the constraints.\n")
            for p in prob_list:
                score = p.get("validation_score", 0)
                if score < 40:
                    continue
                lines.append(f"[{p['id']}] {p['title']}")
                lines.append(f"  Industry       : {p.get('industry', '')}")
                lines.append(f"  Target customer: {p.get('target_customer', '')}")
                lines.append(f"  Pain level     : {p.get('pain_level', '')} | Frequency: {p.get('frequency', '')}")
                lines.append(f"  Current fix    : {p.get('current_solutions', '')}")
                lines.append(f"  Gap/Opportunity: {p.get('gap_opportunity', '')}  ← reframe for B2C marketplace if needed")
                lines.append(f"  Validation score: {score}/85")
                lines.append("")
            sections.append("\n".join(lines))
        else:
            sections.append("=== VALIDATED PROBLEMS ===\nNo validated problems found. Generate ideas from profile + constraints only.")

    return "\n\n".join(sections)


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """
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


# -------------------------
# Groq API Client
# -------------------------
class GroqClient:
    def __init__(self):
        self.endpoint = f"{GROQ_API_BASE.rstrip('/')}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.4,
    ) -> str:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    self.endpoint, headers=self.headers, json=payload, timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                if resp.status_code in (429, 502, 503, 504):
                    wait = 15 ** attempt
                    log.warning(f"Rate limited / server error {resp.status_code} — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Groq API {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                time.sleep(2 ** attempt)
                if attempt == 3:
                    raise RuntimeError(f"Groq network error: {e}") from e
        raise RuntimeError("Groq API failed after 3 retries")


# -------------------------
# Conversation Memory
# Keeps last N messages to stay within context window
# -------------------------
class Memory:
    def __init__(self, max_messages: int = 30):
        self.history: List[Dict[str, str]] = []
        self.max = max_messages

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max:
            # Drop oldest pairs (user+assistant) to stay within limit
            self.history = self.history[-self.max:]

    def as_messages(self) -> List[Dict[str, str]]:
        return list(self.history)


# -------------------------
# Idea Agent
# -------------------------
class IdeaAgent:
    def __init__(self, system_context: str):
        self.client = GroqClient()
        self.memory = Memory(max_messages=30)
        self.system_context = system_context
        self.current_idea: Optional[str] = None

    def _messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self.system_context},
            *self.memory.as_messages(),
        ]

    def generate_opening_idea(self) -> str:
        """
        First turn: generate the ONE best startup idea.
        The kickoff prompt explicitly names the business type and execution constraint
        so the LLM cannot default to a generic software/logistics idea.
        """
        # Extract key signals to embed directly in the kickoff
        q = {}
        for msg in self._messages():
            if "FOUNDER QUESTIONNAIRE" in msg.get("content", ""):
                q = msg["content"]
                break

        kickoff = (
            "Generate the ONE best startup idea for this founder. "
            "Before writing, confirm to yourself:\n"
            "1. Did I read the HARD EXECUTION CONSTRAINTS section? "
            "The founder has NO coding skills and is SOLO — no software building, no team.\n"
            "2. Is this a B2C MARKETPLACE idea where consumers are the end users? "
            "The founder's business interest is Marketplace, which means buyers and sellers, "
            "not B2B tools or logistics services.\n"
            "3. Can this be launched from a laptop under $500 using no-code tools?\n\n"
            "If all 3 are yes — write the idea in the exact structured format. "
            "If not — think again until you have one that passes all 3 checks."
        )
        self.memory.add("user", kickoff)
        reply = self.client.chat(self._messages(), max_tokens=1000, temperature=0.35)
        self.memory.add("assistant", reply)
        self.current_idea = reply
        return reply

    def chat(self, user_message: str) -> str:
        self.memory.add("user", user_message)
        reply = self.client.chat(self._messages(), max_tokens=1000, temperature=0.4)
        self.memory.add("assistant", reply)

        # Track the latest idea if the reply contains the idea header
        if "💡 IDEA:" in reply or "IDEA:" in reply:
            self.current_idea = reply

        return reply


# -------------------------
# Save idea output
# -------------------------
def save_idea_output(agent: IdeaAgent, session_start: int) -> None:
    output = {
        "generated_at": session_start,
        "updated_at": int(time.time()),
        "current_idea": agent.current_idea,
        "chat_history": [
            msg for msg in agent.memory.as_messages()
            # Skip the system kickoff prompt from history
            if not msg["content"].startswith("Based on everything you know about this founder")
        ],
    }
    save_json(PATH_IDEA_OUTPUT, output)
    log.info(f"💾 Idea saved to {PATH_IDEA_OUTPUT}")


# -------------------------
# Main REPL
# -------------------------
def main():
    print("\n" + "━"*50)
    print("  💡 Startup Idea Discovery Agent")
    print("━"*50)
    print("Commands: 'more' = more ideas | 'save' = save idea | 'exit' = quit\n")

    # Load all upstream data
    questionnaire = load_json(PATH_QUESTIONNAIRE, "questionnaireOutput")
    skills        = load_json(PATH_SKILLS,        "skills")
    profile       = load_json(PATH_PROFILE,       "profileAnalysis")
    problems      = load_json(PATH_PROBLEMS,      "problems")

    if not any([questionnaire, profile, problems]):
        print("❌ No input data found. Run ProfileAnalysis.py and ProblemDiscovery.py first.")
        return

    # Build context and agent
    system_context = build_system_context(questionnaire, skills, profile, problems)
    agent = IdeaAgent(system_context=system_context)
    session_start = int(time.time())

    # ── Opening: generate the first idea automatically ───────────────────────
    print("Agent: Analyzing your profile and validated problems...\n")
    try:
        opening = agent.generate_opening_idea()
        print(f"Agent:\n{opening}\n")
        save_idea_output(agent, session_start)
    except Exception as e:
        print(f"❌ Could not generate opening idea: {e}")
        return

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit"}:
            print("Goodbye. Your idea is saved.")
            break

        if cmd == "save":
            save_idea_output(agent, session_start)
            print("✅ Idea saved to idea_output.json\n")
            continue

        if cmd in {"more", "more ideas", "show more", "alternatives"}:
            user_input = (
                "Show me 2-3 alternative startup ideas based on my profile and the other validated problems. "
                "Keep the same structured format."
            )

        # Regular chat turn
        try:
            print("Agent: (thinking...)\n")
            reply = agent.chat(user_input)
            print(f"Agent:\n{reply}\n")
            save_idea_output(agent, session_start)
        except Exception as e:
            print(f"Agent: Sorry, something went wrong: {e}\n")


if __name__ == "__main__":
    main()