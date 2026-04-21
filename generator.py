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
from typing import Iterator

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
and present it as a structured, clinically neutral research report.

CONDUCT RULES:
- Base all claims strictly on the provided excerpts. Do not infer or speculate.
- If an excerpt contains graphic or explicit content, describe events in neutral,
  clinical language (e.g. "alleged prohibited contact" rather than explicit terms).
- Replace any unambiguously graphic terms with [REDACTED] if clinical rephrasing
  would lose key factual meaning.
- Focus on: Named parties, Dates, Locations, Procedural actions, Quoted testimony.
- If the context does not contain enough information, state:
  "Information not found in the provided excerpts."

OUTPUT FORMAT:
- Lead with a one-sentence summary.
- Use bullet points for individual factual findings.
- Cite each finding as [Source N] matching the numbered excerpts below.
- Cite direct urls when available (e.g. [Source 2: https://example.com/doc.pdf]).
- Close with a "Gaps / caveats" section noting what the excerpts do NOT address.\
"""


def build_prompt(chunks: list[RetrievedChunk], question: str) -> tuple[str, str]:
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

    user_message = (
        f"DOCUMENT EXCERPTS:\n\n{context}\n\n"
        f"RESEARCH QUERY:\n{question}\n\n"
        "REPORT:"
    )
    return _SYSTEM_PROMPT, user_message


# ── LLM generation ─────────────────────────────────────────────────────────────

def generate_answer(
    system_prompt: str,
    user_message: str,
    *,
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
            text = _call_groq(system_prompt, user_message, stream=stream)
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


def _call_groq(system_prompt: str, user_message: str, *, stream: bool) -> str:
    """Low-level Groq call. Handles both streaming and non-streaming modes."""
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]
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
) -> Iterator[str]:
    """
    Alternative interface: yield text chunks as they stream.
    Use this when your API layer wants to forward SSE to the client.

    Example (FastAPI):
        return StreamingResponse(stream_answer(sys, usr), media_type="text/plain")
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
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