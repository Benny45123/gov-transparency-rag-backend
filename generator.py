"""
generator.py — Prompt construction + LLM generation with retry and streaming.

Key improvements over the original:
  1. Prompt uses a cleaner system/user split (Groq supports it natively)
  2. generate_answer() RETURNS the text instead of printing it
  3. Streaming mode collects chunks → returns the full string
  4. Retry with exponential backoff on transient errors
  5. Token counting so you can track cost
"""
from __future__ import annotations

import time
from typing import Iterator, Sequence

from groq import Groq, APIError, RateLimitError

from config import cfg
from retriever import RetrievedChunk
from observability import get_logger, log_generation

log = get_logger(__name__)

# ── Groq client (one per process) ─────────────────────────────────────────────

def _get_groq_client() -> Groq:
    return Groq(api_key=cfg.groq_api_key)


_groq: Groq | None = None


def groq_client() -> Groq:
    global _groq
    if _groq is None:
        _groq = _get_groq_client()
    return _groq


# ── Prompt construction ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Forensic Document Analyst specialising in U.S. Federal Court records.
Your job is to extract factual information from the provided court document excerpts
and present it as a structured, clinically neutral research report when the user
is asking about the documents.

INTENT HANDLING:
- Greeting or courtesy only (for example: "hi", "hello", "hey", "thanks"):
  reply briefly and naturally, then invite the user to ask about the documents.
  Do not force citations or a report format for pure greetings.
- Clarification, continuation, or expansion requests (for example: "extend
  that", "continue", "explain more", "expand my previous conversation"):
  use conversation history to identify the prior topic, then answer the expanded
  request using the provided excerpts and citations.
- Document research questions: answer directly from the excerpts, with citations.
- Summary, comparison, timeline, list, count, people, date, location, source, and
  procedure requests: adapt the structure to the requested category while citing
  every factual claim.
- Capability or meta questions about this chatbot: answer briefly based only on
  your role and available context; do not invent system details.
- Off-topic requests unrelated to the provided court documents: say that the
  available excerpts do not support the request and ask for a document-related
  question.

CONDUCT RULES:
- Base all claims strictly on the provided excerpts. Do not infer or speculate.
- Use conversation history only to resolve the user's intent and references; do
  not treat chat history as documentary evidence.
- If an excerpt contains graphic or explicit content, describe events in neutral,
  clinical language (e.g. "alleged prohibited contact" rather than explicit terms).
- Replace any unambiguously graphic terms with [REDACTED] if clinical rephrasing
  would lose key factual meaning.
- Focus on: Named parties, Dates, Locations, Procedural actions, Quoted testimony.
- If the context does not contain enough information, state:
  "Information not found in the provided excerpts."

OUTPUT FORMAT:
- For document answers, lead with a one-sentence summary.
- Use bullet points for individual factual findings unless the user asks for a
  different format.
- Cite each document-backed finding as [Source N] matching the numbered excerpts
  below.
- Cite direct urls when available (e.g. [Source 2: https://example.com/doc.pdf]).
- Close document answers with a "Gaps / caveats" section noting what the excerpts
  do NOT address.
- For greetings and brief meta replies, keep the answer short and omit citations.\
"""


def build_prompt(
    chunks: list[RetrievedChunk],
    question: str,
    *,
    retrieval_query: str | None = None,
) -> tuple[str, str]:
    """
    Return (system_prompt, user_message) — two strings for the Groq chat API.

    Separating system from user gives the model clearer role framing and
    produces more reliable, on-format outputs than a single monolithic prompt.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i}: {chunk.filename}]"
        if chunk.source_url:
            header += f" ({chunk.source_url})"
        context_blocks.append(f"{header}\n{chunk.content}")
    context = "\n\n---\n\n".join(context_blocks)

    query_block = f"RESEARCH QUERY:\n{question}\n\n"
    if retrieval_query and retrieval_query.strip() != question.strip():
        query_block = (
            f"USER QUESTION:\n{question}\n\n"
            f"STANDALONE RETRIEVAL QUERY:\n{retrieval_query}\n\n"
        )

    user_message = (
        f"DOCUMENT EXCERPTS:\n\n{context}\n\n"
        f"{query_block}"
        "REPORT:"
    )
    return _SYSTEM_PROMPT, user_message


def build_chat_messages(
    system_prompt: str,
    user_message: str,
    history: Sequence[dict] | None = None,
) -> list[dict[str, str]]:
    """
    Build the provider message payload for a multi-turn request.

    History is expected to contain alternating user/assistant turns, but this
    helper only enforces valid roles and non-empty content.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    for item in history or []:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


_REWRITE_SYSTEM_PROMPT = """\
Rewrite the user's latest question into one standalone search query for a court
document retrieval system.

Rules:
- Resolve pronouns and follow-up references using the conversation history.
- If the user asks to continue, extend, expand, or add more detail to the previous
  conversation, rewrite the query as a request for more detail about the most
  recent substantive document topic.
- If the latest message is only a greeting, thanks, or chatbot meta question,
  return it unchanged.
- Preserve names, dates, locations, and legal terms exactly when possible.
- Do not answer the question.
- Return only the rewritten query text.\
"""


def rewrite_query(question: str, history: Sequence[dict] | None = None) -> str:
    """Return a standalone retrieval query, falling back to the original text."""
    if not history:
        return question

    messages = build_chat_messages(
        _REWRITE_SYSTEM_PROMPT,
        f"Latest user question:\n{question}\n\nStandalone search query:",
        history=history,
    )
    response = groq_client().chat.completions.create(
        model       = cfg.llm.model,
        messages    = messages,
        temperature = 0.0,
        max_tokens  = 128,
        top_p       = 1.0,
        stream      = False,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten or question


# ── LLM generation ─────────────────────────────────────────────────────────────

def generate_answer(
    system_prompt: str,
    user_message: str,
    *,
    history: Sequence[dict] | None = None,
    stream: bool | None = None,
) -> str:
    """
    Call the Groq API and return the full response text.

    Always returns a string — never prints, never returns None.
    Streaming is used internally for lower time-to-first-token but the
    function still returns the complete text for a clean calling interface.

    Raises:
        RuntimeError: after max_attempts exhausted
    """
    stream = cfg.llm.stream if stream is None else stream
    attempt = 0
    last_error: Exception | None = None

    while attempt < cfg.retry.max_attempts:
        try:
            t0 = time.perf_counter()
            text = _call_groq(
                system_prompt,
                user_message,
                history=history,
                stream=stream,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_out = len(text.split())    # rough estimate; use tiktoken for exact count
            log_generation(cfg.llm.model, tokens_out, elapsed_ms)
            return text

        except RateLimitError as e:
            wait = cfg.retry.backoff_base ** attempt
            log.warning("Rate limited — retrying in %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
            last_error = e
            attempt += 1

        except APIError as e:
            if e.status_code and e.status_code < 500:
                # 4xx errors won't be fixed by retrying
                raise RuntimeError(f"Groq API client error: {e}") from e
            wait = cfg.retry.backoff_base ** attempt
            log.error("Groq API server error (5xx) — retrying in %.1fs", wait)
            time.sleep(wait)
            last_error = e
            attempt += 1

        except Exception as e:
            raise RuntimeError(f"Unexpected generation error: {e}") from e

    raise RuntimeError(
        f"LLM generation failed after {cfg.retry.max_attempts} attempts. "
        f"Last error: {last_error}"
    )


def _call_groq(
    system_prompt: str,
    user_message: str,
    *,
    history: Sequence[dict] | None = None,
    stream: bool,
) -> str:
    """Low-level Groq call. Handles both streaming and non-streaming modes."""
    messages = build_chat_messages(system_prompt, user_message, history=history)
    params = dict(
        model       = cfg.llm.model,
        messages    = messages,
        temperature = cfg.llm.temperature,
        max_tokens  = cfg.llm.max_tokens,
        top_p       = cfg.llm.top_p,
        stream      = stream,
    )

    if stream:
        return _collect_stream(groq_client().chat.completions.create(**params))
    else:
        response = groq_client().chat.completions.create(**params)
        return response.choices[0].message.content or ""


def _collect_stream(stream_response) -> str:
    """Consume a streaming response and return the full text."""
    parts: list[str] = []
    for chunk in stream_response:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
    return "".join(parts)


def stream_answer(
    system_prompt: str,
    user_message: str,
    *,
    history: Sequence[dict] | None = None,
) -> Iterator[str]:
    """
    Alternative interface: yield text chunks as they stream.
    Use this when your API layer wants to forward SSE to the client.

    Example (FastAPI):
        return StreamingResponse(stream_answer(sys, usr), media_type="text/plain")
    """
    messages = build_chat_messages(system_prompt, user_message, history=history)
    response = groq_client().chat.completions.create(
        model       = cfg.llm.model,
        messages    = messages,
        temperature = cfg.llm.temperature,
        max_tokens  = cfg.llm.max_tokens,
        top_p       = cfg.llm.top_p,
        stream      = True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
