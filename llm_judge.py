from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


DEFAULT_QUESTION_FIXTURE = Path("tests/fixtures/domain_questions.json")


@dataclass
class JudgeVerdict:
    correct: bool
    relevant: bool
    consistent_with_sources: bool
    reason: str
    score_0_to_1: float


def load_default_questions(path: Path | str = DEFAULT_QUESTION_FIXTURE) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_response(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> JudgeVerdict:
    provider = (provider or os.getenv("JUDGE_PROVIDER", "groq")).lower()
    prompt = _build_judge_prompt(question, answer, sources)

    if provider == "groq":
        raw = _judge_with_groq(prompt, model=model)
    elif provider == "gemini":
        raw = _judge_with_gemini(prompt, model=model)
    else:
        raise ValueError(f"Unsupported judge provider: {provider}")

    return _parse_verdict(raw)


def _build_judge_prompt(question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    return (
        "Evaluate this RAG response.\n"
        "Return JSON only with keys: correct, relevant, consistent_with_sources, reason, score_0_to_1.\n"
        "Use booleans for the first three fields, a short string for reason, and a float from 0 to 1.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Sources:\n{json.dumps(sources, ensure_ascii=True)}\n"
    )


def _judge_with_groq(prompt: str, *, model: str | None = None) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model or os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict RAG evaluator. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _judge_with_gemini(prompt: str, *, model: str | None = None) -> str:
    if not os.getenv("GEMINI_API_KEY", ""):
        raise EnvironmentError("GEMINI_API_KEY not set")

    llm = ChatGoogleGenerativeAI(
        model=model or os.getenv("JUDGE_MODEL", "gemini-2.5-flash"),
        temperature=0.0,
    )
    response = llm.invoke(
        [
            SystemMessage(content="You are a strict RAG evaluator. Return valid JSON only."),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)


def _parse_verdict(raw: str) -> JudgeVerdict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge did not return valid JSON: {raw}") from exc

    required = {
        "correct",
        "relevant",
        "consistent_with_sources",
        "reason",
        "score_0_to_1",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Judge response missing fields: {sorted(missing)}")

    return JudgeVerdict(
        correct=bool(payload["correct"]),
        relevant=bool(payload["relevant"]),
        consistent_with_sources=bool(payload["consistent_with_sources"]),
        reason=str(payload["reason"]),
        score_0_to_1=float(payload["score_0_to_1"]),
    )
