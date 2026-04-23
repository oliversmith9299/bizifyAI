#!/usr/bin/env python3
"""Questionnaire chatbot agent (LLM-based answer understanding with fixed flow)."""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
REQUEST_TIMEOUT = 30

# Fixed 9 base questions (kept as requested).
BASE_QUESTIONS = [
    {
        "id": "q1_curiosity",
        "question": (
            "What sparks your curiosity?\n"
            "Select a few areas you're interested in. This helps personalize your journey.\n"
            "Options: Technology, Art & Design, Social Impact, Business, Healthcare, Education, "
            "Food & Bev, Other, Don't know and want to chat to know."
        ),
    },
    {
        "id": "q2_level",
        "question": (
            "Which option best describes you?\n"
            "Student: Just starting to learn about entrepreneurship.\n"
            "Beginner: I have an idea and I'm ready to build.\n"
            "Growth-Seeker: My venture is launched and I'm focused on scaling."
        ),
    },
    {
        "id": "q3_business_type",
        "question": (
            "What type of business are you most interested in?\n"
            "Options: E-commerce, Marketplace, Both, I don't know."
        ),
    },
    {
        "id": "q4_region",
        "question": "What region or country will the business operate in?",
    },
    {
        "id": "q5_founder_setup",
        "question": (
            "Will you be a solo founder, or have co-founders/partners?\n"
            "Options: Solo Founder, Co-founders / Partners."
        ),
    },
    {
        "id": "q6_risk_style",
        "question": (
            "I am a natural risk taker?\n"
            "Options: chose from 1 to 5 (1 = low risk, 5 = high risk)."
        ),
    },
    {
        "id": "q7_data_decision_style",
        "question": (
            "When making decisions, do you rely more on data and analysis or on intuition and gut feelings?\n"
            "Options: 1 = Data-driven, 2 = Data + Intuition, 3 = Intuition + Data, 4 = Intuition only, 5 = No preference."
        ),
    },
    {
        "id": "q8_creative_brainstorming_style",
        "question": (
            "When brainstorming ideas, do you prefer structured approaches or free-flowing creativity?\n"
            "Options: 1 = Structured, 2 = Structured + Creative, 3 = Creative + Structured, 4 = Free-flowing only, 5 = No preference."
        ),
    },
    {
        "id": "q9_colllabrative_work_style",
        "question": (
            "When working collaboratively, do you prefer structured teamwork or more flexible, informal collaboration?\n"
            "Options: 1 = Structured, 2 = Structured + Flexible, 3 = Flexible + Structured, 4 = Informal only, 5 = No preference."
        ),
    },
]

# Beginner extra questions (kept, extensible).
BEGINNER_EXTRA_QUESTIONS = [
    {
        "id": "b1_idea_stage",
        "question": (
            "Do you already have an idea/problem in mind, or are you starting from zero?\n"
            "Options: I have a specific idea/problem I want to solve, I'm exploring multiple ideas."
        ),
    },
]

DISCOVERY_QUESTIONS = [
    {"id": "d1", "question": "If you had a free day to work on anything you love, what would you do?"},
    {"id": "d2", "question": "Do you enjoy working more with people, ideas, technology, creativity, or hands-on tasks?"},
    {"id": "d3", "question": "What type of problems do you naturally enjoy solving?"},
    {"id": "d4", "question": "What environment fits you best: office, studio, lab, outdoors, workshop, or remote?"},
    {"id": "d5", "question": "Do you prefer creating, analyzing, helping, designing, managing, or building?"},
    {"id": "d6", "question": "What subjects always catch your interest even without formal study?"},
    {"id": "d7", "question": "If you could instantly master one skill, what would it be and why?"},
    {"id": "d8", "question": "Do you prefer working alone, in small teams, or leading large teams?"},
    {"id": "d9", "question": "Do you like logical challenges, creative work, helping others, mobility, or expression?"},
    {"id": "d10", "question": "What impact do you want your career to have in the next years?"},
]

DECISION_PROMPT = (
    "You finished the Beginner follow-up questions.\n"
    "Would you like to proceed to 10 career-discovery questions for better understanding,\n"
    "or finish now and show the results?\n"
    "Options: Proceed / Finish."
)

QUESTION_SPECS: Dict[str, Dict[str, Any]] = {
    "q1_curiosity": {
        "type": "multi_choice",
        "options": [
            "Technology",
            "Art & Design",
            "Social Impact",
            "Business",
            "Healthcare",
            "Education",
            "Food & Bev",
            "Other",
            "Don't know and want to chat to know",
        ],
    },
    "q2_level": {"type": "single_choice", "options": ["Student", "Beginner", "Growth-Seeker"]},
    "q3_business_type": {"type": "single_choice", "options": ["E-commerce", "Marketplace", "Both", "I don't know"]},
    "q4_region": {"type": "free_text"},
    "q5_founder_setup": {"type": "single_choice", "options": ["Solo Founder", "Co-founders / Partners"]},
    "q6_risk_style": {"type": "rating_1_5"},
    "q7_data_decision_style": {"type": "rating_1_5"},
    "q8_creative_brainstorming_style": {"type": "rating_1_5"},
    "q9_colllabrative_work_style": {"type": "rating_1_5"},
    "b1_idea_stage": {
        "type": "single_choice",
        "options": [
            "I have a specific idea/problem I want to solve",
            "I'm exploring multiple ideas",
        ],
    },
    "discovery_decision": {"type": "single_choice", "options": ["Proceed", "Finish"]},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class GroqChatClient:
    api_key: str
    base_url: str = GROQ_API_BASE
    model: str = GROQ_MODEL

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.35) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if resp.status_code in (429, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Groq API error {resp.status_code}: {err}")
            except requests.RequestException:
                time.sleep(2 ** attempt)
        raise RuntimeError("Groq API request failed after retries.")


class QuestionnaireBotAgent:
    def __init__(self, groq_client: Optional[GroqChatClient] = None):
        self.groq_client = groq_client
        self.answers: Dict[str, Any] = {}
        self.phase = "base"
        self.base_index = 0
        self.beginner_index = 0
        self.discovery_index = 0
        self.is_beginner = False
        self.q1_dont_know = False
        self.result: Optional[Dict[str, Any]] = None

    def start(self) -> Dict[str, Any]:
        q = BASE_QUESTIONS[0]
        return self._response(reply=self._compose_reply("start", "", q["question"]), question_id=q["id"], next_question=q["question"])

    def ask(self, user_text: str) -> Dict[str, Any]:
        if self.phase == "completed":
            return self._response(reply="Questionnaire already completed.", complete=True, result=self.result)

        if self.phase == "base":
            return self._handle_base(user_text)
        if self.phase == "beginner":
            return self._handle_beginner(user_text)
        if self.phase == "decision":
            return self._handle_decision(user_text)
        if self.phase == "discovery":
            return self._handle_discovery(user_text)

        self.phase = "completed"
        self.result = self._build_result()
        return self._response(reply="Completed.", complete=True, result=self.result)

    def _handle_base(self, user_text: str) -> Dict[str, Any]:
        q = BASE_QUESTIONS[self.base_index]
        parsed = self._understand_answer(q["id"], q["question"], user_text)
        if not parsed["valid"]:
            return self._invalid(q, user_text, parsed["reason"])

        self.answers[q["id"]] = parsed["canonical"]
        if q["id"] == "q2_level":
            self.is_beginner = parsed["canonical"] == "Beginner"
        if q["id"] == "q1_curiosity":
            canonical = parsed["canonical"] if isinstance(parsed["canonical"], list) else [str(parsed["canonical"])]
            self.q1_dont_know = "Don't know and want to chat to know" in canonical

        self.base_index += 1

        if self.base_index < len(BASE_QUESTIONS):
            next_q = BASE_QUESTIONS[self.base_index]
            note = ""
            if q["id"] == "q1_curiosity" and self.q1_dont_know:
                note = "After we finish the rest of questionnaire questions, I will ask 10 discovery questions to figure your career."
            return self._response(
                reply=self._compose_reply("next", user_text, next_q["question"], note),
                question_id=next_q["id"],
                next_question=next_q["question"],
            )

        if self.is_beginner and BEGINNER_EXTRA_QUESTIONS:
            self.phase = "beginner"
            first = BEGINNER_EXTRA_QUESTIONS[0]
            return self._response(
                reply=self._compose_reply("beginner_start", user_text, first["question"], "Since you selected Beginner, I will ask your Beginner extra questions now."),
                question_id=first["id"],
                next_question=first["question"],
            )

        if self.q1_dont_know:
            self.phase = "discovery"
            first = DISCOVERY_QUESTIONS[0]
            return self._response(
                reply=self._compose_reply("discovery_start", user_text, first["question"]),
                question_id=first["id"],
                next_question=first["question"],
            )

        self.phase = "completed"
        self.result = self._build_result()
        return self._response(reply="Questionnaire completed. Here is your personality result.", complete=True, result=self.result)

    def _handle_beginner(self, user_text: str) -> Dict[str, Any]:
        q = BEGINNER_EXTRA_QUESTIONS[self.beginner_index]
        parsed = self._understand_answer(q["id"], q["question"], user_text)
        if not parsed["valid"]:
            return self._invalid(q, user_text, parsed["reason"])

        self.answers[q["id"]] = parsed["canonical"]
        self.beginner_index += 1

        if self.beginner_index < len(BEGINNER_EXTRA_QUESTIONS):
            next_q = BEGINNER_EXTRA_QUESTIONS[self.beginner_index]
            return self._response(
                reply=self._compose_reply("next", user_text, next_q["question"]),
                question_id=next_q["id"],
                next_question=next_q["question"],
            )

        self.phase = "decision"
        return self._response(
            reply=self._compose_reply("decision", user_text, DECISION_PROMPT),
            question_id="discovery_decision",
            next_question=DECISION_PROMPT,
        )

    def _handle_decision(self, user_text: str) -> Dict[str, Any]:
        parsed = self._understand_answer("discovery_decision", DECISION_PROMPT, user_text)
        if not parsed["valid"]:
            return self._response(
                reply=self._compose_reply("invalid", user_text, DECISION_PROMPT, parsed["reason"]),
                question_id="discovery_decision",
                next_question=DECISION_PROMPT,
            )

        self.answers["discovery_decision"] = parsed["canonical"]
        if parsed["canonical"] == "Proceed":
            self.phase = "discovery"
            first = DISCOVERY_QUESTIONS[0]
            return self._response(
                reply=self._compose_reply("discovery_start", user_text, first["question"]),
                question_id=first["id"],
                next_question=first["question"],
            )

        self.phase = "completed"
        self.result = self._build_result()
        return self._response(reply="Understood. Finishing now and showing your personality result.", complete=True, result=self.result)

    def _handle_discovery(self, user_text: str) -> Dict[str, Any]:
        q = DISCOVERY_QUESTIONS[self.discovery_index]
        if not user_text.strip():
            return self._invalid(q, user_text, "Please answer this question.")

        self.answers[q["id"]] = user_text.strip()
        self.discovery_index += 1

        if self.discovery_index < len(DISCOVERY_QUESTIONS):
            next_q = DISCOVERY_QUESTIONS[self.discovery_index]
            return self._response(
                reply=self._compose_reply("next", user_text, next_q["question"]),
                question_id=next_q["id"],
                next_question=next_q["question"],
            )

        self.phase = "completed"
        self.result = self._build_result()
        return self._response(reply="All done. Here is your personality result.", complete=True, result=self.result)

    def _understand_answer(self, question_id: str, question_text: str, user_text: str) -> Dict[str, Any]:
        spec = QUESTION_SPECS.get(question_id, {"type": "free_text"})
        if not user_text.strip():
            return {"valid": False, "canonical": None, "reason": "Please answer this question."}

        if not self.groq_client:
            return self._fallback_understanding(spec, user_text)

        prompt = (
            "Classify the user's answer for one questionnaire question. Return JSON only with keys: "
            "valid (boolean), canonical (string|number|array), reason (string).\n"
            "Rules:\n"
            "- If valid=false, reason must be short and actionable.\n"
            "- If valid=true, canonical must follow allowed options/format.\n"
            "- For rating_1_5 canonical must be integer 1..5.\n"
            "- For multi_choice canonical must be array of allowed options.\n"
            "- Do semantic matching, not strict keyword matching.\n\n"
            f"question_id: {question_id}\n"
            f"question_text: {question_text}\n"
            f"expected_type: {spec.get('type')}\n"
            f"allowed_options: {json.dumps(spec.get('options', []), ensure_ascii=False)}\n"
            f"user_answer: {user_text}\n"
        )

        try:
            raw = self.groq_client.chat(messages=[{"role": "user", "content": prompt}], max_tokens=220, temperature=0.1)
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "valid" in parsed:
                return {
                    "valid": bool(parsed.get("valid")),
                    "canonical": parsed.get("canonical"),
                    "reason": str(parsed.get("reason") or "Please answer in the expected format."),
                }
        except Exception:
            pass

        return self._fallback_understanding(spec, user_text)

    def _fallback_understanding(self, spec: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        t = spec.get("type")
        raw = user_text.strip()

        if t == "free_text":
            return {"valid": len(raw) >= 2, "canonical": raw, "reason": "Please enter a valid text answer."}

        if t == "rating_1_5":
            m = re.search(r"\b([1-5])\b", raw)
            if m:
                return {"valid": True, "canonical": int(m.group(1)), "reason": ""}
            return {"valid": False, "canonical": None, "reason": "Please answer with a number from 1 to 5."}

        options = spec.get("options", [])
        if t == "single_choice":
            lower = _normalize(raw)
            for opt in options:
                opt_lower = _normalize(opt)
                if opt_lower in lower or lower in opt_lower:
                    return {"valid": True, "canonical": opt, "reason": ""}
            return {"valid": False, "canonical": None, "reason": f"Please choose one valid option: {', '.join(options)}."}

        if t == "multi_choice":
            lower = _normalize(raw)
            picked = []
            for opt in options:
                opt_lower = _normalize(opt)
                if opt_lower in lower:
                    picked.append(opt)
            if picked:
                return {"valid": True, "canonical": picked, "reason": ""}
            return {"valid": False, "canonical": None, "reason": "Please choose at least one valid option."}

        return {"valid": True, "canonical": raw, "reason": ""}

    def _compose_reply(self, mode: str, user_answer: str, next_question: str, note: str = "") -> str:
        fallback = f"{note}\n{next_question}".strip()
        if not self.groq_client:
            return fallback

        system = (
            "You are a concise questionnaire assistant. "
            "Write 1-2 short natural lines. Do not add or alter questions. "
            "The final line must contain next_question exactly."
        )
        user = (
            f"mode: {mode}\n"
            f"user_answer: {user_answer}\n"
            f"note: {note}\n"
            f"next_question: {next_question}"
        )
        try:
            text = self.groq_client.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=180,
                temperature=0.35,
            )
            if next_question not in text:
                first_line = text.strip().splitlines()[0] if text.strip() else ""
                return f"{first_line}\n{next_question}" if first_line else fallback
            return text
        except Exception:
            return fallback

    def _build_result(self) -> Dict[str, Any]:
        report = self._generate_personality_with_groq() or self._local_fallback_report()
        return {"personality_report": report, "questionnaire_answers": self.answers}

    def _generate_personality_with_groq(self) -> Optional[Dict[str, Any]]:
        if not self.groq_client:
            return None

        prompt = (
            "Analyze this entrepreneurship questionnaire and produce only valid JSON with keys: "
            "personality_type, traits, work_style, strengths, growth_areas, recommended_next_step.\n"
            f"Answers: {json.dumps(self.answers, ensure_ascii=False)}"
        )
        try:
            raw = self.groq_client.chat(messages=[{"role": "user", "content": prompt}], max_tokens=700, temperature=0.2)
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _local_fallback_report(self) -> Dict[str, Any]:
        return {
            "personality_type": "Builder-Explorer",
            "traits": ["Curious", "Adaptive", "Goal-oriented"],
            "work_style": "Practical and iterative",
            "strengths": ["initiative", "adaptability", "problem awareness"],
            "growth_areas": ["validation discipline", "strategic focus"],
            "recommended_next_step": "Run 5 customer interviews this week and validate one core assumption.",
        }

    def _invalid(self, question: Dict[str, str], user_answer: str, reason: str) -> Dict[str, Any]:
        return self._response(
            reply=self._compose_reply("invalid", user_answer, question["question"], reason),
            question_id=question["id"],
            next_question=question["question"],
        )

    def _response(
        self,
        reply: str,
        question_id: Optional[str] = None,
        next_question: Optional[str] = None,
        complete: bool = False,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "reply": reply,
            "phase": self.phase,
            "question_id": question_id,
            "next_question": next_question,
            "complete": complete,
            "result": result,
            "progress": {
                "base_answered": self.base_index,
                "beginner_answered": self.beginner_index,
                "discovery_answered": self.discovery_index,
            },
        }


def main() -> None:
    groq_client = None
    if GROQ_API_KEY:
        groq_client = GroqChatClient(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE, model=GROQ_MODEL)

    bot = QuestionnaireBotAgent(groq_client=groq_client)
    out = bot.start()
    print("Bot:", out["reply"])

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break

        out = bot.ask(user)
        print("Bot:", out["reply"])
        if out["complete"]:
            print(json.dumps(out["result"], ensure_ascii=False, indent=2))
            break


if __name__ == "__main__":
    main()
