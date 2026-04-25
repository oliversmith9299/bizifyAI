###1 profile analysis (Analysis not chatbot)
#Assess founder suitability
#Build personal runway 
#chat to founder to get more ideas if those are not what they want
######

#input is questionnaireoutput.json and then will output a json insights to be an input for the next agent (problem discovery)
#  and also will be stored in the database to be used by the future agents in the flow.


import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from db.connection import SessionLocal
from db import crud

#

# -------------------------
# Load environment
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

# -------------------------
# Init client
# -------------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)

# -------------------------
def run_profile_analysis(user_id: str):
    db = SessionLocal()
    print(f"[ProfileAnalysis] Running for user {user_id}")

    try:
        questionnaire = crud.get_questionnaire_output_json(db, user_id)

        if not questionnaire:
            raise ValueError("No questionnaire found")


        prompt = f"""
You are a senior startup advisor and venture builder.

Your task is to deeply analyze a user and determine what kind of business they can realistically build.

You MUST:
- Think critically
- Avoid generic answers
- Base ALL recommendations on user skills, experience, and behavior
- Prefer realistic, executable ideas over fancy ideas

-------------------------
INPUT DATA
-------------------------
{json.dumps(questionnaire, indent=2)}

-------------------------
ANALYSIS INSTRUCTIONS
-------------------------

1. PERSONALITY ANALYSIS
- Identify thinking style (builder, analytical, creative, operator, etc.)
- Identify motivation (money, impact, innovation, independence)
- Identify behavioral traits

2. SKILL-BASED CAPABILITY ANALYSIS (VERY IMPORTANT)
- Analyze skills_json carefully
- Determine what the user can ACTUALLY build
- Highlight skill gaps
- Do NOT suggest industries that require skills the user does not have

3. FOUNDER READINESS
- beginner / intermediate / advanced
- ability to execute alone vs needs team

4. INDUSTRY MATCHING
- Recommend industries that:
  ✔ match skills
  ✔ match interests
  ✔ match region
- Avoid unrealistic industries

5. PROBLEM SPACE SELECTION
- Suggest REALISTIC problem areas the user can work on
- Must align with:
  ✔ skills
  ✔ market
  ✔ business type

6. SEARCH DIRECTION (CRITICAL FOR NEXT AGENT)
- Generate HIGH QUALITY search queries
- Must be specific and useful for discovering real-world problems

Prioritize industries where the user can BUILD or OPERATE using their current skills.

If the user lacks creative or technical skills, avoid suggesting industries that require them directly.

Prefer platform, service, or operational business models over production-based ones.

Focus on platform, service, or system-level opportunities rather than content creation or production.

Avoid suggesting businesses that require artistic or technical execution unless explicitly supported by user skills.

Prefer problems where the user can act as a strategist, operator, or marketplace builder.

-------------------------
OUTPUT RULES (STRICT)
-------------------------
- Return ONLY valid JSON
- No explanations
- No text outside JSON
- Keep answers concise but meaningful
- Avoid generic words like "various", "many", "etc."

-------------------------
OUTPUT FORMAT
-------------------------
{{
  "personality_insights": {{
    "type": "...",
    "motivation": "...",
    "traits": [],
    "strengths": [],
    "weaknesses": []
  }},
  "founder_profile": {{
    "experience_level": "...",
    "execution_style": "...",
    "risk_level": "...",
    "readiness": "...",
    "skill_level_summary": "...",
    "key_skill_gaps": []
  }},
  "recommended_industries": [],
  "recommended_problem_spaces": [],
  "search_direction": {{
    "keywords": []
  }},
  "system_flags": {{
    "needs_guidance": true/false,
    "should_suggest_learning": true/false
  }}
}}
"""


        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content

        # clean
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            result = json.loads(cleaned)

            crud.save_profile(db, user_id, result)

            print(f"[ProfileAnalysis] Profile saved for user {user_id}")

            return result

        except Exception as e:
            print("ProfileAnalysis Error:", str(e))
            raise 

        

    finally:
        db.close()