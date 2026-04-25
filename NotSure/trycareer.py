#!/usr/bin/env python3
"""
CareerPath AI Agent v3 — Groq-powered career discovery chatbot.
Fixes in this version:
  - Two-phase design: conversation phase + separate JSON generation call
  - JSON generation uses 4000 tokens so it never gets cut off
  - Answer fields tracked by explicit AI signal, not blind index
  - Vague answer detection with mandatory follow-up probing
  - Clean terminal display of final career results
Requires: pip install requests python-dotenv
"""

import os
import re
import time
import json
import sys
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
REQUEST_TIMEOUT = 60

if not GROQ_API_KEY:
   raise RuntimeError("GROQ_API_KEY is not set. Add it to Streamlit secrets.")


# ─────────────────────────────────────────────
# Career Data
# ─────────────────────────────────────────────
CAREER_CLUSTERS = [
    {"name": "Technology & Digital Careers",        "jobs": ["Software Development", "AI & Machine Learning", "Data Science & Analytics", "Cybersecurity", "Cloud & DevOps", "Game Development", "UI/UX Design", "Blockchain / Web3", "IT Support", "AR/VR Development", "Embedded Systems", "No-Code / Low-Code"]},
    {"name": "Business & Management",               "jobs": ["Entrepreneurship / Startups", "Project Management", "Business Analysis", "HR & Talent Management", "Digital Marketing", "Sales & Customer Success", "Product Management", "E-commerce", "Operations & Supply Chain", "Business Consulting"]},
    {"name": "Creative & Media",                    "jobs": ["Graphic Design", "Animation (2D/3D)", "Video Editing", "Photography", "Filmmaking", "Content Writing", "Copywriting", "Social Media Management", "Branding & Art Direction", "Interior Design"]},
    {"name": "Science & Engineering",               "jobs": ["Mechanical Engineering", "Civil Engineering", "Electrical & Electronics", "Chemical Engineering", "Robotics & IoT", "Aerospace Engineering", "Environmental Engineering", "Industrial Engineering"]},
    {"name": "Finance & Economics",                 "jobs": ["Accounting", "Financial Analysis", "Banking", "Risk Management", "Investment & Trading", "Tax Consultancy", "Insurance & Actuarial Science", "Corporate Finance", "FinTech"]},
    {"name": "Healthcare & Life Sciences",          "jobs": ["Medicine & Surgery", "Nursing", "Pharmacy", "Physiotherapy", "Psychology & Counseling", "Nutrition & Dietetics", "Medical Research", "Dentistry", "Public Health"]},
    {"name": "Education & Training",                "jobs": ["School Teaching", "University / Academic Education", "Tutoring", "Instructional Design", "Curriculum Development", "E-learning Content Creation", "Special Education"]},
    {"name": "Law, Politics & Government",          "jobs": ["Law & Legal Practice", "Public Policy", "Diplomacy & International Relations", "Government Administration", "Immigration & Human Rights", "Criminal Justice"]},
    {"name": "Humanities & Languages",              "jobs": ["Translation & Interpreting", "Linguistics", "Creative Writing / Publishing", "History", "Anthropology / Sociology", "Philosophy"]},
    {"name": "Trades & Practical Professions",      "jobs": ["Carpentry & Woodwork", "Electrical Installation", "Plumbing", "Automotive & Mechanics", "Welding & Metalwork", "Tailoring & Fashion Craft", "Construction Work"]},
    {"name": "Beauty, Fashion & Wellness",          "jobs": ["Makeup Artistry", "Hair Styling & Barbering", "Skin Care & Aesthetics", "Spa & Wellness Therapy", "Fashion Design", "Jewelry Design", "Modeling"]},
    {"name": "Sports, Fitness & Physical Performance", "jobs": ["Fitness Training / Gym Coaching", "Sports Coaching", "Yoga & Pilates Instruction", "Sports Nutrition", "Physical Therapy", "Dance & Performance Arts"]},
    {"name": "Hospitality, Tourism & Events",       "jobs": ["Hotel Management", "Travel & Tour Operations", "Event Planning", "Restaurant Management", "Airline Cabin Crew", "Catering & Banquets"]},
    {"name": "Agriculture, Environment & Wildlife", "jobs": ["Agriculture & Agribusiness", "Farming & Aquaculture", "Environmental Science", "Forestry & Wildlife Conservation", "Sustainable Development", "Horticulture"]},
    {"name": "Media, Journalism & Communications",  "jobs": ["Journalism & Reporting", "Broadcasting", "Public Relations", "Corporate Communications", "Media Production", "Content Strategy"]},
    {"name": "Performing Arts & Entertainment",     "jobs": ["Acting & Theater Arts", "Music Performance", "Dance & Choreography", "Stage Management", "Entertainment Production", "Talent Management"]},
    {"name": "Mining, Oil, Gas & Energy",           "jobs": ["Mining Engineering", "Oil & Gas Exploration", "Renewable Energy Technologies", "Energy Management", "Geology & Earth Sciences"]},
    {"name": "Real Estate & Property Management",   "jobs": ["Real Estate Brokerage", "Property Management", "Real Estate Development", "Urban Planning", "Facility Management"]},
    {"name": "Culinary, Food & Beverage",           "jobs": ["Chef / Cook", "Baking & Pastry Arts", "Food Styling", "Culinary Instruction", "Food Science & Technology", "Restaurant Entrepreneurship"]},
    {"name": "Logistics, Transport & Supply Chain", "jobs": ["Logistics & Supply Chain Management", "Freight & Cargo Handling", "Aviation Operations", "Maritime & Shipping", "Fleet Management"]},
    {"name": "Military, Security & Public Safety",  "jobs": ["Police & Law Enforcement", "Military Careers", "Fire & Rescue Services", "Disaster Management", "Private Security", "Intelligence & Cyber Defense"]},
    {"name": "Charity, NGOs & Social Impact",       "jobs": ["Non-Profit Management", "Fundraising & Development", "Community Outreach", "Social Work", "Advocacy & Campaigning"]},
    {"name": "Freelance, Self-Employment & Gig Work", "jobs": ["Freelance Writing & Editing", "Graphic Design Freelancing", "Consulting Services", "Handmade Crafts & Artisanal Goods", "Gig Economy Roles"]},
]

FIXED_QUESTIONS = [
    {
        "field": "interests",
        "label": "Interests",
        "question": "First up — what sparks your curiosity? Pick as many areas as you like:",
        "multi": True,
        "choices": [
            "Technology",
            "Art & Design",
            "Social Impact",
            "Business",
            "Healthcare",
            "Education",
            "Food & Bev",
            "Other",
            "Don't know, and want to find out",
        ],
    },
    {
        "field": "stage",
        "label": "Current Stage",
        "question": "What best describes your current stage?",
        "multi": False,
        "choices": [
            "Student",
            "Beginner",
            "Growth-Seeker",
        ],
    },
    {
        "field": "business_type",
        "label": "Business Type",
        "question": "What type(s) of business are you most interested in?",
        "multi": False,
        "choices": [
            "Marketplace",
            "E-commerce",
            "Both",
            "I don't know",
        ],
    },
    {
        "field": "region",
        "label": "Region",
        "question": "What region or country will the business operate in?",
        "multi": False,
        "choices": [
            "North America",
            "Europe",
            "Middle East",
            "Southeast Asia",
            "South Asia",
            "Africa",
            "Latin America",
            "East Asia",
            "Global / Remote",
        ],
    },
    {
        "field": "founder_type",
        "label": "Founder Type",
        "question": "Will you be going solo, or do you have co-founders/partners?",
        "multi": False,
        "choices": [
            "Solo Founder",
            "Co-founders / Partners",
        ],
    },
]

CAREER_DISCOVERY_QUESTIONS = [
    {
        "field": "career_q1_free_day",
        "label": "Career Q1",
        "question": "If you had a completely free day, what would you most likely choose to do?",
        "multi": True,
        "choices": [
            "Build or create something",
            "Learn something new",
            "Solve a problem",
            "Help someone",
            "Design or express ideas",
            "Work with technology",
            "Organize or plan something",
        ],
    },
    {
        "field": "career_q2_work_type",
        "label": "Career Q2",
        "question": "What type of work do you enjoy the most?",
        "multi": True,
        "choices": [
            "Working with people",
            "Working with ideas",
            "Working with technology",
            "Creative work (design, writing, media)",
            "Hands-on physical work",
            "Analyzing data",
            "Leading others",
            "Working independently",
        ],
    },
    {
        "field": "career_q3_problem_type",
        "label": "Career Q3",
        "question": "What kind of problems do you naturally enjoy solving?",
        "multi": True,
        "choices": [
            "Logical or technical problems",
            "Creative challenges",
            "Real-life practical issues",
            "Helping people with personal problems",
            "Organizing or improving systems",
            "Business or money-related challenges",
        ],
    },
    {
        "field": "career_q4_environment",
        "label": "Career Q4",
        "question": "What type of work environment suits you best?",
        "multi": True,
        "choices": [
            "Office",
            "Lab",
            "Creative studio",
            "Outdoors",
            "Clinic or service setting",
            "Workshop or production space",
            "Remote / flexible",
            "Fast-paced environment",
            "Calm and structured environment",
        ],
    },
    {
        "field": "career_q5_impact",
        "label": "Career Q5",
        "question": "What kind of impact do you want to have in your career?",
        "multi": True,
        "choices": [
            "Help people",
            "Build products",
            "Innovate new ideas",
            "Make strong income",
            "Protect the environment",
            "Teach or share knowledge",
            "Lead organizations",
            "Entertain or inspire others",
        ],
    },
    {
        "field": "career_q6_risk",
        "label": "Career Q6",
        "question": "How comfortable are you with uncertainty and risk in your career?",
        "multi": False,
        "choices": [
            "I prefer stability and predictable outcomes",
            "I'm comfortable with moderate risk if there's growth potential",
            "I enjoy taking calculated risks",
            "I thrive in high-risk, high-reward situations",
            "It depends on the situation",
        ],
    },
]

ALL_QUESTIONS = FIXED_QUESTIONS + CAREER_DISCOVERY_QUESTIONS

VAGUE_PATTERN = re.compile(
    r"^\s*(i\s+don'?t\s+(know|really\s+know)|not\s+sure|idk|no\s+idea|maybe|"
    r"i\s+guess|hmm+|um+|uh+|nothing|none|i\s+have\s+no\s+idea|hard\s+to\s+say|"
    r"unsure|no\s+preference|whatever|anything)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

def is_vague(text: str) -> bool:
    text = text.strip()
    if len(text) < 5:
        return True
    return bool(VAGUE_PATTERN.match(text))


def needs_career_discovery(first_answer: str) -> bool:
    answer = first_answer.strip().lower()
    return bool(
        re.search(
            r"(don'?t\s+know|do\s+not\s+know|not\s+sure|unsure|idk|no\s+idea|find\s+out|unknown)",
            answer,
        )
    )


# ─────────────────────────────────────────────
# User Profile Memory
# ─────────────────────────────────────────────
class UserProfileMemory:
    def __init__(self, filename: str = "career_profile.json"):
        self.filename = filename
        self.profile: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        self.profile[key] = value
        self._save()

    def _save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=2, ensure_ascii=False)

    def load(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self.profile = json.load(f)
            print(f"  📂 Loaded existing profile from '{self.filename}'")
        except (FileNotFoundError, json.JSONDecodeError):
            self.profile = {}

    def get(self, key: str, default=None):
        return self.profile.get(key, default)

    def summary(self) -> Dict[str, Any]:
        return self.profile.copy()

    def store_answer(self, field: str, answer: str):
        if "answers" not in self.profile:
            self.profile["answers"] = {}
        self.profile["answers"][field] = answer
        self._save()
        label = next((q["label"] for q in ALL_QUESTIONS if q["field"] == field), field)
        short = answer[:65] + ("…" if len(answer) > 65 else "")
        print(f'  ✔  {label:<22} → "{short}"')

    def get_answers(self) -> Dict[str, str]:
        return self.profile.get("answers", {})

    def answered_count(self) -> int:
        return len(self.get_answers())

    def clear(self):
        self.profile = {}
        self._save()


# ─────────────────────────────────────────────
# Groq Chat Client
# ─────────────────────────────────────────────
class GroqChatClient:
    def __init__(self, api_key: str, base_url: str = GROQ_API_BASE, model: str = GROQ_MODEL):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.endpoint = f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.4,
    ) -> str:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        for attempt in range(1, 4):
            try:
                effective_model = self.model
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
                )
                if resp.status_code == 200:
                    data   = resp.json()
                    choice = data.get("choices", [{}])[0]
                    return choice.get("message", {}).get("content", "").strip()
                if resp.status_code in (429, 502, 503, 504):
                    wait = 2 ** attempt
                    print(f"  [groq] error {resp.status_code}. Retrying in {wait}s…")
                    time.sleep(wait)
                    continue
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text

                # Some models can return tool-call style generations that fail without tools.
                # Retry once with a known-compatible fallback model.
                err_txt = str(err).lower()
                if (
                    resp.status_code == 400
                    and "tool_use_failed" in err_txt
                    and "tool choice is none" in err_txt
                    and self.model != GROQ_FALLBACK_MODEL
                ):
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = GROQ_FALLBACK_MODEL
                    fallback_resp = requests.post(
                        self.endpoint, headers=headers, json=fallback_payload, timeout=REQUEST_TIMEOUT
                    )
                    if fallback_resp.status_code == 200:
                        data = fallback_resp.json()
                        choice = data.get("choices", [{}])[0]
                        return choice.get("message", {}).get("content", "").strip()
                    try:
                        fb_err = fallback_resp.json()
                    except Exception:
                        fb_err = fallback_resp.text
                    raise RuntimeError(
                        f"Groq API error {resp.status_code}: {err}. "
                        f"Fallback model '{GROQ_FALLBACK_MODEL}' also failed: {fb_err}"
                    )
                raise RuntimeError(f"Groq API error {resp.status_code}: {err}")
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"  [groq] network error: {e}. Retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError("Groq API request failed after all retries.")


# ─────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_messages: int = 80):
        self.buffer: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        while len(self.buffer) > self.max_messages:
            if self.buffer[0]["role"] == "system":
                self.buffer.pop(1)
            else:
                self.buffer.pop(0)

    def as_messages(self) -> List[Dict[str, str]]:
        return list(self.buffer)


# ─────────────────────────────────────────────
# System Prompt (conversation phase only — NO JSON here)
# ─────────────────────────────────────────────
CLUSTER_NAMES = [c["name"] for c in CAREER_CLUSTERS]


def _format_fixed_questions_for_prompt() -> str:
    lines = []
    for idx, q in enumerate(FIXED_QUESTIONS, 1):
        choice_text = " | ".join(q["choices"])
        mode = "multi-select" if q.get("multi") else "single-select"
        lines.append(f"{idx}. {q['question']} ({mode}; choices: {choice_text})")
    return "\n".join(lines)


def _format_career_questions_for_prompt() -> str:
    lines = []
    for idx, q in enumerate(CAREER_DISCOVERY_QUESTIONS, 1):
        choice_text = " | ".join(q["choices"])
        mode = "multi-select" if q.get("multi") else "single-select"
        lines.append(f"{idx}. {q['question']} ({mode}; choices: {choice_text})")
    return "\n".join(lines)


CONVERSATION_SYSTEM_PROMPT = f"""
You are CareerPath AI — a warm, empathetic, and encouraging career counselor.

YOUR ONLY JOB IN THIS PHASE:
Ask the user questions one at a time and collect meaningful answers.
After all required questions are done, say exactly this line and nothing else:
  ALL_QUESTIONS_DONE

════════════════════════════════════════════
FIRST 5 FIXED QUESTIONS (ask in this exact order)
════════════════════════════════════════════
{_format_fixed_questions_for_prompt()}

BRANCHING RULE:
- Use ONLY the user's answer to fixed question 1.
- If they indicate they do not know their career path (for example: selecting "Don't know, and want to find out"), then after finishing fixed questions 1-5, ask the extra 6 career discovery questions below.
- If they clearly know their career path, do NOT ask the extra 6 questions.

════════════════════════════════════════════
EXTRA 6 CAREER DISCOVERY QUESTIONS
(ask these only when branch says user does NOT know their career)
════════════════════════════════════════════
{_format_career_questions_for_prompt()}

════════════════════════════════════════════
STRICT RULES
════════════════════════════════════════════
1. Ask ONE question at a time. Never combine two questions.
2. After a real answer → give 1 warm sentence of acknowledgment, then ask the next question.
3. If the user gives a VAGUE answer ("I don't know", "idk", "not sure", "maybe", a single word, etc.):
   → Do NOT move to the next question.
   → Probe deeper. Offer 2-3 concrete examples.
   → Stay on the SAME question until you get a real, specific answer.
4. Never number your questions out loud.
5. Be human, warm, and encouraging — like a trusted friend.
6. Only discuss career discovery. Politely decline anything off-topic.
7. Do NOT generate any JSON. Do NOT analyze careers. That happens separately.
8. For all fixed and career questions, show options and allow either selecting from options or typing their own answer.
9. If question is multi-select, user may choose multiple options in one response (comma-separated is fine).
10. Once all required questions are answered with real answers, say ONLY: ALL_QUESTIONS_DONE
"""

# ─────────────────────────────────────────────
# JSON Generation Prompt (separate call, high token limit)
# ─────────────────────────────────────────────
def build_json_prompt(answers: Dict[str, str]) -> str:
    answers_text = "\n".join(
        f"  {q['label']}: {answers.get(q['field'], 'N/A')}"
        for q in ALL_QUESTIONS
    )
    user_answer_schema = {
        q["field"]: "<copy from user answers or N/A>"
        for q in ALL_QUESTIONS
    }
    cluster_list = json.dumps(CLUSTER_NAMES, indent=2)
    schema_json = json.dumps(user_answer_schema, indent=4, ensure_ascii=False)

    return f"""You are a career analysis AI. Based on the user answers below, generate a complete career profile JSON.

USER ANSWERS:
{answers_text}

APPROVED CAREER CLUSTERS (use ONLY these names, spelled exactly):
{cluster_list}

Output ONLY the JSON. No introduction, no explanation, no markdown prose. Start with {{ and end with }}.

{{
  "user_answers": {schema_json},
  "career_analysis": {{
    "personality_traits": ["trait1", "trait2", "trait3"],
    "core_strengths": ["strength1", "strength2", "strength3"],
    "work_style_summary": "A short paragraph describing how this person works best",
    "ideal_environment": "Short description of their ideal workplace"
  }},
  "career_recommendations": [
    {{
      "rank": 1,
      "cluster": "Exact Cluster Name from approved list",
      "top_roles": ["Role A", "Role B", "Role C"],
      "match_reason": "Specific explanation referencing the user's actual answers",
      "confidence_score": 95
    }},
    {{
      "rank": 2,
      "cluster": "Exact Cluster Name from approved list",
      "top_roles": ["Role A", "Role B", "Role C"],
      "match_reason": "Specific explanation referencing the user's actual answers",
      "confidence_score": 83
    }},
    {{
      "rank": 3,
      "cluster": "Exact Cluster Name from approved list",
      "top_roles": ["Role A", "Role B", "Role C"],
      "match_reason": "Specific explanation referencing the user's actual answers",
      "confidence_score": 72
    }}
  ],
  "next_steps": [
    "Concrete action step 1",
    "Concrete action step 2",
    "Concrete action step 3"
  ],
  "motivational_note": "A personalized encouraging message referencing something specific from the user's actual answers"
}}"""


# ─────────────────────────────────────────────
# Answer Tracker
# ─────────────────────────────────────────────
class AnswerTracker:
    def __init__(self):
        self.current_q_index: int = 0
        self.in_followup: bool = False
        self.discovery_required: Optional[bool] = None

    @property
    def active_questions(self) -> List[Dict[str, Any]]:
        if self.discovery_required:
            return ALL_QUESTIONS
        return FIXED_QUESTIONS

    @property
    def current_field(self) -> Optional[str]:
        if self.current_q_index < len(self.active_questions):
            return self.active_questions[self.current_q_index]["field"]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_q_index >= len(self.active_questions)

    @property
    def total_questions(self) -> int:
        return len(self.active_questions)

    @property
    def current_question(self) -> Optional[Dict[str, Any]]:
        if self.current_q_index < len(self.active_questions):
            return self.active_questions[self.current_q_index]
        return None

    def _matches_option(self, user_text: str) -> bool:
        q = self.current_question
        if not q or not q.get("choices"):
            return False

        raw_parts = [p.strip() for p in re.split(r"[,/|]", user_text) if p.strip()]
        if not raw_parts:
            raw_parts = [user_text.strip()]
        norm_parts = [re.sub(r"\s+", " ", p.lower()) for p in raw_parts]
        option_set = {
            re.sub(r"\s+", " ", opt.lower())
            for opt in q["choices"]
        }

        if q.get("multi"):
            return all(part in option_set for part in norm_parts)
        return len(norm_parts) == 1 and norm_parts[0] in option_set

    def evaluate(self, user_text: str) -> bool:
        """Returns True if answer is real and we should advance."""
        if self._matches_option(user_text):
            self.in_followup = False
            return True
        if is_vague(user_text):
            self.in_followup = True
            return False
        self.in_followup = False
        return True

    def advance(self, user_text: str, user_profile: UserProfileMemory):
        field = self.current_field
        if field:
            user_profile.store_answer(field, user_text)
        if self.current_q_index == 0 and self.discovery_required is None:
            self.discovery_required = needs_career_discovery(user_text)
        self.current_q_index += 1

    def progress_bar(self) -> str:
        done  = self.current_q_index
        total = self.total_questions
        filled = "●" * done + "○" * (total - done)
        pct = int((done / total) * 100)
        return f"[{filled}]  {done}/{total}  ({pct}%)"


# ─────────────────────────────────────────────
# Career Agent
# ─────────────────────────────────────────────
class CareerAdvisorAgent:
    def __init__(self, client: GroqChatClient, memory: ConversationMemory):
        self.client = client
        self.memory = memory
        self.memory.add("system", CONVERSATION_SYSTEM_PROMPT.strip())

    def chat(self, user_text: str) -> str:
        self.memory.add("user", user_text)
        reply = self.client.chat(
            messages=self.memory.as_messages(),
            max_tokens=600,   # conversation replies should be short
            temperature=0.45,
        )
        self.memory.add("assistant", reply)
        return reply

    def generate_career_report(self, answers: Dict[str, str]) -> dict:
        """
        Separate API call with high token budget dedicated to JSON generation.
        Uses a fresh message list — no conversation history contamination.
        """
        prompt = build_json_prompt(answers)
        messages = [{"role": "user", "content": prompt}]

        print("\n  ⚙️  Generating your career profile… (this takes a few seconds)\n")

        raw = self.client.chat(
            messages=messages,
            max_tokens=4000,   # plenty of room for full JSON
            temperature=0.2,   # low temp for consistent structured output
        )

        # Strip any accidental markdown fences
        raw = re.sub(r"^```json\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object if there's surrounding text
            match = re.search(r"\{[\s\S]+\}", raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(
                f"Could not parse career report JSON.\nRaw output:\n{raw[:500]}"
            )


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────
def save_final_report(data: dict, filename: str = "career_report.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Full report saved → '{filename}'")


def display_report(data: dict):
    W = 62
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  🎯  YOUR CAREER PROFILE RESULTS  ".center(W) + "║")
    print("╚" + "═" * W + "╝")

    analysis  = data.get("career_analysis", {})
    traits    = ", ".join(analysis.get("personality_traits", []))
    strengths = ", ".join(analysis.get("core_strengths", []))
    env       = analysis.get("ideal_environment", "")
    style     = analysis.get("work_style_summary", "")

    print()
    if traits:    print(f"  🧠 Personality Traits  : {traits}")
    if strengths: print(f"  💪 Core Strengths      : {strengths}")
    if env:       print(f"  🏢 Ideal Environment   : {env}")
    if style:
        # word-wrap at 55 chars
        words, line, lines = style.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 55:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        prefix = "  ⚙️  Work Style          : "
        for i, l in enumerate(lines):
            print((prefix if i == 0 else " " * len(prefix)) + l)

    recs = data.get("career_recommendations", [])
    if recs:
        print()
        print("  " + "─" * (W - 2))
        print("  📋  TOP CAREER MATCHES")
        print("  " + "─" * (W - 2))
        for rec in recs:
            score = rec.get("confidence_score", 0)
            filled_blocks = score // 10
            bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
            cluster = rec.get("cluster", "")
            roles   = ", ".join(rec.get("top_roles", []))
            reason  = rec.get("match_reason", "")

            print()
            print(f"  #{rec['rank']}  {cluster}")
            print(f"      Match Score : [{bar}] {score}%")
            print(f"      Top Roles   : {roles}")
            # word-wrap reason
            words, line, lines = reason.split(), "", []
            for w in words:
                if len(line) + len(w) + 1 > 52:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            prefix = "      Why Match   : "
            for i, l in enumerate(lines):
                print((prefix if i == 0 else " " * len(prefix)) + l)

    steps = data.get("next_steps", [])
    if steps:
        print()
        print("  " + "─" * (W - 2))
        print("  🚀  RECOMMENDED NEXT STEPS")
        print("  " + "─" * (W - 2))
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

    note = data.get("motivational_note", "")
    if note:
        print()
        print("  " + "─" * (W - 2))
        words, line, lines = note.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 56:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        prefix = "  💬  "
        for i, l in enumerate(lines):
            print((prefix if i == 0 else "      ") + l)

    print()
    print("═" * W)
    print()


def display_report_streamlit(data: dict):
    st.subheader("Your Career Profile Results")

    analysis = data.get("career_analysis", {})
    traits = analysis.get("personality_traits", [])
    strengths = analysis.get("core_strengths", [])
    env = analysis.get("ideal_environment", "")
    style = analysis.get("work_style_summary", "")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Personality Traits**")
        if traits:
            for t in traits:
                st.write(f"- {t}")
        else:
            st.write("- N/A")

        st.markdown("**Core Strengths**")
        if strengths:
            for s in strengths:
                st.write(f"- {s}")
        else:
            st.write("- N/A")
    with c2:
        st.markdown("**Ideal Environment**")
        st.write(env or "N/A")
        st.markdown("**Work Style Summary**")
        st.write(style or "N/A")

    recs = data.get("career_recommendations", [])
    if recs:
        st.markdown("---")
        st.markdown("### Top Career Matches")
        for rec in recs:
            rank = rec.get("rank", "")
            cluster = rec.get("cluster", "Unknown")
            roles = rec.get("top_roles", [])
            reason = rec.get("match_reason", "")
            score = int(rec.get("confidence_score", 0))

            st.markdown(f"**#{rank} {cluster}**")
            st.progress(max(0, min(100, score)) / 100)
            st.caption(f"Match Score: {score}%")
            if roles:
                st.write("Top Roles:", ", ".join(roles))
            if reason:
                st.write("Why Match:", reason)
            st.markdown("")

    steps = data.get("next_steps", [])
    if steps:
        st.markdown("---")
        st.markdown("### Recommended Next Steps")
        for idx, step in enumerate(steps, 1):
            st.write(f"{idx}. {step}")

    note = data.get("motivational_note", "")
    if note:
        st.markdown("---")
        st.info(note)

    with st.expander("Show Full JSON"):
        st.json(data)


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║      🤖  CAREERPATH AI — Discover Your Future        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Be honest — even rough answers help a lot!          ║")
    print("║  Commands:  progress  ·  profile  ·  exit            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────────
# CLI REPL
# ─────────────────────────────────────────────
def repl(agent: CareerAdvisorAgent, user_profile: UserProfileMemory):
    print_banner()

    tracker    = AnswerTracker()
    final_done = False

    # ── Boot ─────────────────────────────────────────────────────
    print("CareerPath AI: (connecting…)")
    try:
        greeting = agent.chat("Hello! I am ready to start.")
        print(f"\nCareerPath AI: {greeting}\n")
    except Exception as e:
        print(f"CareerPath AI: Connection failed — {e}")
        return

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Your progress is saved.")
            break

        if not raw:
            continue

        lower = raw.lower()

        # ── Commands ──────────────────────────────────────────────
        if lower in {"exit", "quit"}:
            print("Goodbye! Come back anytime.")
            break

        if lower == "progress":
            print(f"\n  Progress: {tracker.progress_bar()}")
            answers = user_profile.get_answers()
            if answers:
                print()
                for q in ALL_QUESTIONS:
                    ans = answers.get(q["field"])
                    if ans:
                        short = ans[:60] + ("…" if len(ans) > 60 else "")
                        print(f"  • {q['label']:<22} : {short}")
            print()
            continue

        if lower == "profile":
            print("\n── Saved Profile ──────────────────────────────────────")
            print(json.dumps(user_profile.summary(), indent=2, ensure_ascii=False))
            print("───────────────────────────────────────────────────────\n")
            continue

        if lower == "help":
            print("  Commands: progress · profile · exit\n")
            continue

        # ── After session complete: free chat ─────────────────────
        if final_done:
            print("CareerPath AI: (thinking…)")
            try:
                reply = agent.chat(raw)
                print(f"\nCareerPath AI: {reply}\n")
            except Exception as e:
                print(f"CareerPath AI: Error — {e}\n")
            continue

        # ── Evaluate the answer ───────────────────────────────────
        should_advance = tracker.evaluate(raw)

        # ── Send to AI (conversation mode) ────────────────────────
        print("CareerPath AI: (thinking…)")
        try:
            reply = agent.chat(raw)
        except Exception as e:
            print(f"CareerPath AI: Error — {e}\n")
            continue

        if should_advance and not tracker.is_complete:
            tracker.advance(raw, user_profile)

        # ── Check for completion signal ───────────────────────────
        if tracker.is_complete or "ALL_QUESTIONS_DONE" in reply:
            all_answers = user_profile.get_answers()
            try:
                report = agent.generate_career_report(all_answers)
                user_profile.update("final_report", report)
                save_final_report(report)
                display_report(report)
                final_done = True
                print("  (Type 'profile' to see full JSON · 'exit' to quit · or keep chatting)\n")
            except Exception as e:
                print(f"\n  ❌ Could not generate report: {e}")
                print("  Try typing anything to retry, or 'exit' to quit.\n")
            continue

        # ── Normal conversational reply ───────────────────────────
        print(f"\nCareerPath AI: {reply}\n")
        if should_advance and not tracker.is_complete:
            print(f"  Progress: {tracker.progress_bar()}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    groq_client = GroqChatClient(
        api_key  = GROQ_API_KEY,
        base_url = GROQ_API_BASE,
        model    = GROQ_MODEL,
    )
    memory = ConversationMemory(max_messages=80)
    agent  = CareerAdvisorAgent(client=groq_client, memory=memory)

    user_profile = UserProfileMemory(filename="career_profile.json")
    user_profile.load()

    existing = user_profile.get_answers()
    if existing:
        count = len(existing)
        print(f"\n  Found {count} saved answer(s) from a previous session.")
        choice = input("  (R) Resume   (N) Start fresh  →  ").strip().upper()
        if choice == "N":
            user_profile.clear()
            print("  ✓ Starting fresh!\n")
        else:
            print("  ✓ Resuming — saved answers are loaded.\n")

    repl(agent, user_profile)


def _is_streamlit_run() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def streamlit_main():
    st.set_page_config(page_title="CareerPath AI", page_icon=":robot_face:", layout="wide")
    st.title("CareerPath AI - Career Discovery")

    if "agent" not in st.session_state:
        groq_client = GroqChatClient(
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_BASE,
            model=GROQ_MODEL,
        )
        memory = ConversationMemory(max_messages=80)
        agent = CareerAdvisorAgent(client=groq_client, memory=memory)
        user_profile = UserProfileMemory(filename="career_profile.json")
        user_profile.load()

        st.session_state.agent = agent
        st.session_state.user_profile = user_profile
        st.session_state.tracker = AnswerTracker()
        st.session_state.final_done = False
        st.session_state.report = None
        st.session_state.chat = []

        greeting = agent.chat("Hello! I am ready to start.")
        st.session_state.chat.append({"role": "assistant", "content": greeting})

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Start Fresh"):
            st.session_state.user_profile.clear()
            st.session_state.clear()
            st.rerun()

        tracker = st.session_state.tracker
        st.write("Progress")
        pct = tracker.current_q_index / max(1, tracker.total_questions)
        st.progress(pct)
        st.caption(f"{tracker.current_q_index}/{tracker.total_questions} answered")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.report:
        display_report_streamlit(st.session_state.report)

    agent = st.session_state.agent
    user_profile = st.session_state.user_profile
    tracker = st.session_state.tracker

    def process_answer(user_text: str):
        st.session_state.chat.append({"role": "user", "content": user_text})

        if st.session_state.final_done:
            reply = agent.chat(user_text)
            st.session_state.chat.append({"role": "assistant", "content": reply})
            return

        should_advance = tracker.evaluate(user_text)
        reply = agent.chat(user_text)
        st.session_state.chat.append({"role": "assistant", "content": reply})

        if should_advance and not tracker.is_complete:
            tracker.advance(user_text, user_profile)

        # Generate result once tracker is complete, even if model forgets completion token.
        if tracker.is_complete and not st.session_state.final_done:
            all_answers = user_profile.get_answers()
            try:
                report = agent.generate_career_report(all_answers)
                user_profile.update("final_report", report)
                save_final_report(report, filename=os.path.join("career_report.json"))
                st.session_state.report = report
                st.session_state.final_done = True
                st.session_state.chat.append(
                    {"role": "assistant", "content": "All questions complete. I generated your career report."}
                )
            except Exception as e:
                st.session_state.chat.append({"role": "assistant", "content": f"Could not generate report: {e}"})

    # Choice-style input for current question (like your React flow)
    if not st.session_state.final_done and not st.session_state.report:
        q = tracker.current_question
        if q and q.get("choices"):
            st.markdown("---")
            st.markdown(f"**Current Question:** {q['question']}")
            key_base = f"q_{tracker.current_q_index}"

            if q.get("multi"):
                selected_multi = st.multiselect(
                    "Choose one or more options",
                    q["choices"],
                    key=f"{key_base}_multi",
                )
                if st.button(
                    "Submit selected options",
                    key=f"{key_base}_submit_multi",
                    disabled=len(selected_multi) == 0,
                ):
                    process_answer(", ".join(selected_multi))
                    st.rerun()
            else:
                selected_single = st.radio(
                    "Choose one option",
                    q["choices"],
                    index=None,
                    key=f"{key_base}_single",
                )
                if st.button(
                    "Submit selected option",
                    key=f"{key_base}_submit_single",
                    disabled=selected_single is None,
                ):
                    process_answer(selected_single)
                    st.rerun()

            typed = st.text_input("Or type your own answer", key=f"{key_base}_typed")
            if st.button(
                "Send typed answer",
                key=f"{key_base}_typed_submit",
                disabled=not typed.strip(),
            ):
                process_answer(typed.strip())
                st.rerun()

    user_text = st.chat_input("Type your answer...")
    if user_text:
        process_answer(user_text)
        st.rerun()


if __name__ == "__main__":
    if _is_streamlit_run():
        streamlit_main()
    else:
        main()
