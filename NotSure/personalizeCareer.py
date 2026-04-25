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
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
REQUEST_TIMEOUT = 60

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")


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

STAGE1_QUESTIONS = [
    {"field": "free_day_activities", "label": "Free Day Activities",  "question": "If you had a free day to work on anything you love, what kind of things would you spend time doing?"},
    {"field": "work_style",          "label": "Work Style",           "question": "Do you enjoy working with people, ideas, technology, creativity, or hands-on activities more?"},
    {"field": "problem_solving",     "label": "Problem Solving",      "question": "When someone asks for your help, what type of problems do you naturally enjoy solving?"},
    {"field": "environment",         "label": "Work Environment",     "question": "What kind of environment do you see yourself in — office, lab, studio, outdoors, kitchen, clinic, workshop, or remote work?"},
    {"field": "preference",          "label": "Work Preference",      "question": "Do you prefer creating things, analyzing data, helping people, designing visuals, managing projects, or building with your hands?"},
    {"field": "interests",           "label": "Interests & Topics",   "question": "What subjects or topics have always caught your interest — even if you haven't studied them yet?"},
    {"field": "master_skill",        "label": "Dream Skill",          "question": "If you could master one skill instantly, what would it be and why?"},
    {"field": "team_size",           "label": "Team Preference",      "question": "Do you imagine yourself working alone, with small teams, or leading many people?"},
    {"field": "activity_type",       "label": "Activity Type",        "question": "Do you like solving logical problems, designing things, helping others, moving around, or expressing ideas?"},
    {"field": "impact",              "label": "Desired Impact",       "question": "What kind of impact do you dream of having — helping people, building products, entertaining others, protecting the environment, earning money, innovating, or teaching?"},
]

VAGUE_PATTERN = re.compile(
    r"^\s*(i\s+don'?t\s+(know|really\s+know)|not\s+sure|idk|no\s+idea|maybe|"
    r"i\s+guess|hmm+|um+|uh+|nothing|none|i\s+have\s+no\s+idea|hard\s+to\s+say|"
    r"unsure|no\s+preference|whatever|anything)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

CLARIFICATION_PATTERN = re.compile(
    r"^\s*(like\s+what|what\s+do\s+you\s+mean|can\s+you\s+explain|"
    r"can\s+you\s+give\s+(an?\s+)?example|give\s+me\s+(an?\s+)?example|"
    r"for\s+example|example|which\s+one|"
    r"what\s+kind)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

def is_clarification_request(text: str) -> bool:
    text = text.strip().lower()
    if not text:
        return True
    if CLARIFICATION_PATTERN.match(text):
        return True
    words = text.split()
    if text.endswith("?") and len(words) <= 8:
        return True
    if len(words) <= 3 and words[0] in {"what", "which", "how", "why", "when", "where", "like"}:
        return True
    return False

def is_vague(text: str) -> bool:
    text = text.strip()
    lowered = text.lower()
    if len(text) < 5:
        return True
    if re.match(r"^\s*(idk|i\s+don'?t\s+know)(\s+[a-z']+){0,4}\s*[.!?]?\s*$", lowered):
        return True
    if VAGUE_PATTERN.match(text):
        return True
    if is_clarification_request(text):
        return True
    return False

def is_real_answer(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip()) and not is_vague(text)

def is_answer_relevant(field: Optional[str], text: str) -> bool:
    if not field:
        return True
    t = text.strip().lower()
    if not t:
        return False
    if field == "work_style":
        return any(k in t for k in [
            "people", "person", "team", "ideas", "idea", "technology", "tech",
            "creative", "creativity", "hands-on", "hands on", "practical",
            "coding", "code", "data", "design", "building", "build",
        ])
    if field == "team_size":
        return any(k in t for k in [
            "alone", "solo", "small", "team", "group", "lead", "leader",
            "manage", "many", "independent",
        ])
    if field == "activity_type":
        return any(k in t for k in [
            "logical", "logic", "design", "help", "others", "moving", "express",
            "ideas", "problem", "creative",
        ])
    return True

def next_unanswered_index(answers: Dict[str, Any]) -> int:
    for idx, q in enumerate(STAGE1_QUESTIONS):
        if not is_real_answer(answers.get(q["field"], "")):
            return idx
    return len(STAGE1_QUESTIONS)

def prune_invalid_saved_answers(user_profile: "UserProfileMemory") -> int:
    answers = user_profile.get_answers()
    if not isinstance(answers, dict):
        return 0
    cleaned: Dict[str, str] = {}
    removed = 0
    for q in STAGE1_QUESTIONS:
        field = q["field"]
        raw = answers.get(field)
        if is_real_answer(raw) and is_answer_relevant(field, str(raw)):
            cleaned[field] = raw.strip()
        elif field in answers:
            removed += 1
    if removed:
        user_profile.profile["answers"] = cleaned
        user_profile._save()
    return removed


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
        label = next((q["label"] for q in STAGE1_QUESTIONS if q["field"] == field), field)
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
QUESTION_LIST = [q["question"] for q in STAGE1_QUESTIONS]

CONVERSATION_SYSTEM_PROMPT = f"""
You are CareerPath AI — a warm, empathetic, and encouraging career counselor.

YOUR ONLY JOB IN THIS PHASE:
Ask the user the 10 questions below, one at a time, and collect meaningful answers.
When all 10 are done, say exactly this line and nothing else:
  ALL_QUESTIONS_DONE

════════════════════════════════════════════
THE 10 QUESTIONS (ask in this exact order)
════════════════════════════════════════════
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(QUESTION_LIST))}

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
8. Once question 10 is answered with a real answer, say ONLY: ALL_QUESTIONS_DONE
"""

# ─────────────────────────────────────────────
# JSON Generation Prompt (separate call, high token limit)
# ─────────────────────────────────────────────
def build_json_prompt(answers: Dict[str, str]) -> str:
    answers_text = "\n".join(
        f"  {q['label']}: {answers.get(q['field'], 'N/A')}"
        for q in STAGE1_QUESTIONS
    )
    cluster_list = json.dumps(CLUSTER_NAMES, indent=2)

    return f"""You are a career analysis AI. Based on the user answers below, generate a complete career profile JSON.

USER ANSWERS:
{answers_text}

APPROVED CAREER CLUSTERS (use ONLY these names, spelled exactly):
{cluster_list}

Output ONLY the JSON. No introduction, no explanation, no markdown prose. Start with {{ and end with }}.

{{
  "user_answers": {{
    "free_day_activities": "<copy from user answers>",
    "work_style": "<copy from user answers>",
    "problem_solving": "<copy from user answers>",
    "environment": "<copy from user answers>",
    "preference": "<copy from user answers>",
    "interests": "<copy from user answers>",
    "master_skill": "<copy from user answers>",
    "team_size": "<copy from user answers>",
    "activity_type": "<copy from user answers>",
    "impact": "<copy from user answers>"
  }},
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

    @property
    def current_field(self) -> Optional[str]:
        if self.current_q_index < len(STAGE1_QUESTIONS):
            return STAGE1_QUESTIONS[self.current_q_index]["field"]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_q_index >= len(STAGE1_QUESTIONS)

    def evaluate(self, user_text: str) -> bool:
        """Returns True if answer is real and we should advance."""
        if is_vague(user_text):
            self.in_followup = True
            return False
        if not is_answer_relevant(self.current_field, user_text):
            self.in_followup = True
            return False
        self.in_followup = False
        return True

    def advance(self, user_text: str, user_profile: UserProfileMemory):
        field = self.current_field
        if field:
            user_profile.store_answer(field, user_text)
        self.current_q_index += 1

    def progress_bar(self) -> str:
        done  = self.current_q_index
        total = len(STAGE1_QUESTIONS)
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
    removed = prune_invalid_saved_answers(user_profile)
    if removed:
        print(f"  Cleaned {removed} invalid saved answer(s) from profile.")
    tracker.current_q_index = next_unanswered_index(user_profile.get_answers())

    def finalize_if_ready() -> bool:
        nonlocal final_done
        if final_done:
            return True
        all_answers = user_profile.get_answers()
        tracker.current_q_index = next_unanswered_index(all_answers)
        if not tracker.is_complete:
            return False
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
        return True

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
                for q in STAGE1_QUESTIONS:
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

        # ── Check for completion signal ───────────────────────────
        if "ALL_QUESTIONS_DONE" in reply:
            # Save this last answer first
            if should_advance and not tracker.is_complete:
                tracker.advance(raw, user_profile)

            if finalize_if_ready():
                continue

            # Ignore premature model completion signal.
            q = STAGE1_QUESTIONS[tracker.current_q_index]["question"]
            print(f"\nCareerPath AI: Let's keep going.\nCareerPath AI: {q}\n")
            continue

        # ── Normal conversational reply ───────────────────────────
        print(f"\nCareerPath AI: {reply}\n")

        # ── Save answer and advance tracker ───────────────────────
        if should_advance and not tracker.is_complete:
            tracker.advance(raw, user_profile)
            if not tracker.is_complete:
                print(f"  Progress: {tracker.progress_bar()}\n")
            else:
                finalize_if_ready()


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
        prune_invalid_saved_answers(user_profile)
        tracker = AnswerTracker()
        tracker.current_q_index = next_unanswered_index(user_profile.get_answers())

        st.session_state.agent = agent
        st.session_state.user_profile = user_profile
        st.session_state.tracker = tracker
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
        pct = tracker.current_q_index / len(STAGE1_QUESTIONS)
        st.progress(pct)
        st.caption(f"{tracker.current_q_index}/{len(STAGE1_QUESTIONS)} answered")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.report:
        display_report_streamlit(st.session_state.report)

    user_text = st.chat_input("Type your answer...")
    if not user_text:
        return

    st.session_state.chat.append({"role": "user", "content": user_text})

    agent = st.session_state.agent
    user_profile = st.session_state.user_profile
    tracker = st.session_state.tracker

    if st.session_state.final_done:
        reply = agent.chat(user_text)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    should_advance = tracker.evaluate(user_text)
    reply = agent.chat(user_text)
    st.session_state.chat.append({"role": "assistant", "content": reply})

    if should_advance and not tracker.is_complete:
        tracker.advance(user_text, user_profile)

    # Generate result once tracker is complete, even if model forgets to emit ALL_QUESTIONS_DONE.
    if tracker.is_complete and not st.session_state.final_done:
        all_answers = user_profile.get_answers()
        try:
            report = agent.generate_career_report(all_answers)
            user_profile.update("final_report", report)
            save_final_report(report, filename=os.path.join("data", "career_report.json"))
            st.session_state.report = report
            st.session_state.final_done = True
            st.session_state.chat.append(
                {"role": "assistant", "content": "All questions complete. I generated your career report."}
            )
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Could not generate report: {e}"})

    st.rerun()


if __name__ == "__main__":
    if _is_streamlit_run():
        streamlit_main()
    else:
        main()
