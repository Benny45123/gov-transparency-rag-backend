"""
query.py — Production RAG query pipeline.

Public API:
    rag_query(store, question)          → RAGResponse
    rag_query_stream(store, question)   → Iterator[str]

Full pipeline:
  1. Cache lookup    — Redis; skip inference on hit
  2. Stage-1 retrieval — ANN search with metadata filter + fallback
  3. Stage-2 reranking — Cohere / BM25 fallback
  4. Prompt construction
  5. LLM generation  — Groq llama-3.3-70b, retry on transient errors
  6. Cache write     — Redis SETEX with TTL
  7. DB write        — Supabase query_history (async-safe fire-and-forget)
  8. Trace log       — structured JSON with full latency breakdown
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator

from langchain_pinecone import PineconeVectorStore

from cache import get_cache
from config import cfg
from database import save_query
from generator import build_prompt, generate_answer, stream_answer
from observability import Tracer, get_logger
from retriever import RetrievedChunk, build_vector_store, retrieve_chunks, rerank_chunks

log = get_logger(__name__)


# ── Response schema ────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Typed return value from rag_query().

    answer   — Full LLM answer text.
    sources  — List of source dicts (file, url, chunk_index, score, preview).
    cached   — True if served from Redis cache.
    query    — Original question echoed back.
    error    — Non-None if the pipeline hit a recoverable error.
    """
    answer:  str
    sources: list[dict]  = field(default_factory=list)
    cached:  bool        = False
    query:   str         = ""
    error:   str | None  = None

    def to_dict(self) -> dict:
        return {
            "answer":  self.answer,
            "sources": self.sources,
            "cached":  self.cached,
            "query":   self.query,
            "error":   self.error,
        }


_NO_CHUNKS_RESPONSE = RAGResponse(
    answer="No relevant documents were found in the database for this query.",
)


# ── DB persistence helper ──────────────────────────────────────────────────────

def _persist_async(
    question: str,
    answer: str,
    sources: list[dict],
    cached: bool,
    error: str | None,
    namespace: str,
) -> None:
    """
    Write query + answer to Supabase on a background thread.

    Using a daemon thread means:
    - The main request returns immediately (no added latency)
    - A slow or failed DB write never blocks the caller
    - The thread is automatically reaped when the process exits
    """
    def _write() -> None:
        try:
            save_query(
                question  = question,
                answer    = answer,
                sources   = sources,
                namespace = namespace,
                cached    = cached,
                error     = error,
            )
        except Exception as e:
            log.error("Background DB write failed: %s", e)

    t = threading.Thread(target=_write, daemon=True)
    t.start()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def rag_query(
    store: PineconeVectorStore,
    question: str,
    *,
    namespace: str = "epstein-docs",
    skip_cache: bool = False,
) -> RAGResponse:
    """
    Full RAG pipeline — returns a complete RAGResponse.

    Parameters
    ----------
    store       Connected PineconeVectorStore (build once, reuse everywhere).
    question    User's natural-language question.
    namespace   Pinecone namespace (written to DB for multi-corpus setups).
    skip_cache  Force fresh retrieval + generation even if Redis has a hit.
    """
    cache = get_cache()

    with Tracer("rag_query") as tracer:
        tracer.set("query", question[:200])
        tracer.set("model", cfg.llm.model)
        tracer.set("namespace", namespace)

        # ── 1. Redis cache lookup ────────────────────────────────────────────
        if not skip_cache:
            cached_payload = cache.get(question)
            if cached_payload:
                tracer.set("cache_hit", True)
                response = RAGResponse(
                    answer  = cached_payload["answer"],
                    sources = cached_payload["sources"],
                    cached  = True,
                    query   = question,
                )
                # Still persist cache hits to DB so history is complete
                _persist_async(
                    question  = question,
                    answer    = response.answer,
                    sources   = response.sources,
                    cached    = True,
                    error     = None,
                    namespace = namespace,
                )
                return response

        tracer.set("cache_hit", False)

        # ── 2. Stage-1 retrieval ─────────────────────────────────────────────
        with tracer.span("retrieve"):
            chunks = retrieve_chunks(store, question)

        if not chunks:
            _persist_async(
                question=question, answer=_NO_CHUNKS_RESPONSE.answer,
                sources=[], cached=False, error="no_chunks", namespace=namespace,
            )
            return _NO_CHUNKS_RESPONSE

        tracer.set("chunks_retrieved", len(chunks))

        # ── 3. Stage-2 reranking ─────────────────────────────────────────────
        with tracer.span("rerank"):
            chunks = rerank_chunks(chunks, question)

        tracer.set("chunks_after_rerank", len(chunks))
        tracer.set("sources", [c.filename for c in chunks])

        # ── 4. Prompt construction ────────────────────────────────────────────
        system_prompt, user_message = build_prompt(chunks, question)

        # ── 5. LLM generation ─────────────────────────────────────────────────
        try:
            with tracer.span("generate"):
                answer = generate_answer(system_prompt, user_message)
        except RuntimeError as exc:
            log.error("Generation failed: %s", exc)
            err_response = RAGResponse(
                answer  = f"Generation failed after retries: {exc}",
                sources = [c.to_source_dict() for c in chunks],
                query   = question,
                error   = str(exc),
            )
            _persist_async(
                question=question, answer=err_response.answer,
                sources=err_response.sources, cached=False,
                error=str(exc), namespace=namespace,
            )
            return err_response

        # ── 6. Assemble response ──────────────────────────────────────────────
        sources = [c.to_source_dict() for c in chunks]
        response = RAGResponse(
            answer  = answer,
            sources = sources,
            cached  = False,
            query   = question,
        )

        # ── 7. Redis write ────────────────────────────────────────────────────
        cache.set(question, {"answer": answer, "sources": sources})

        # ── 8. Supabase write (background thread) ─────────────────────────────
        _persist_async(
            question=question, answer=answer, sources=sources,
            cached=False, error=None, namespace=namespace,
        )

    return response


# ── Streaming variant ──────────────────────────────────────────────────────────

def rag_query_stream(
    store: PineconeVectorStore,
    question: str,
    *,
    namespace: str = "epstein-docs",
    skip_cache: bool = False,
) -> Iterator[str]:
    """
    Streaming variant — yields answer tokens as they arrive from the LLM.
    Redis + Supabase writes happen after the last token is yielded.

    Usage (FastAPI):
        return StreamingResponse(
            rag_query_stream(store, q),
            media_type="text/event-stream"
        )
    """
    cache = get_cache()

    # Cache hit → yield full text at once
    cached_payload = cache.get(question)
    if cached_payload and not skip_cache:
        log.info("Cache HIT (stream) for query='%.60s'", question)
        _persist_async(
            question=question, answer=cached_payload["answer"],
            sources=cached_payload["sources"], cached=True,
            error=None, namespace=namespace,
        )
        yield cached_payload["answer"]
        return

    # Retrieve + rerank
    chunks = retrieve_chunks(store, question)
    if not chunks:
        yield _NO_CHUNKS_RESPONSE.answer
        return

    chunks = rerank_chunks(chunks, question)
    system_prompt, user_message = build_prompt(chunks, question)

    # Stream tokens; accumulate for cache + DB write
    accumulated: list[str] = []
    for token in stream_answer(system_prompt, user_message):
        accumulated.append(token)
        yield token

    full_answer = "".join(accumulated)
    sources     = [c.to_source_dict() for c in chunks]

    cache.set(question, {"answer": full_answer, "sources": sources})
    _persist_async(
        question=question, answer=full_answer, sources=sources,
        cached=False, error=None, namespace=namespace,
    )


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    store = build_vector_store()

    test_questions = [
        "What does Epstein do in Palm Beach according to court documents?",
        "Who were Epstein's known associates?",
        "When did the alleged incidents at Epstein's Palm Beach house occur?",
        "How many victims are mentioned in the court filings?",
    ]

    questions = sys.argv[1:] if len(sys.argv) > 1 else test_questions

    for question in questions:
        print(f"\n{'─' * 64}")
        print(f"Q: {question}")
        print("─" * 64)

        result = rag_query(store, question)

        if result.error:
            print(f"⚠  Error: {result.error}")

        print(f"A: {result.answer}")
        print(f"\n📚 Sources ({len(result.sources)}) | cached={result.cached}")
        for i, src in enumerate(result.sources, 1):
            print(
                f"  [{i}] {src['source_file']}  "
                f"chunk={src['chunk_index']}  "
                f"score={src['score']}"
            )
            print(f"      {src['preview']}...")


# from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI,HarmCategory,HarmBlockThreshold
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import PromptTemplate
# from pinecone import Pinecone
# from dotenv import load_dotenv
# from db import save_query
# import os
# import re
# load_dotenv()
# INDEX_NAME    = "gov-transparency-index"
# NAMESPACE     = "epstein-docs"
# TOP_K         = 5           
# MODEL_NAME    = "gemini-2.5-flash" 
# def load_vector_store()->PineconeVectorStore:
#     embedding_model=GoogleGenerativeAIEmbeddings(
#         model="gemini-embedding-001",
#         google_api_key=os.getenv("GEMINI_API_KEY")
#     )
#     vector_store=PineconeVectorStore.from_existing_index(
#         index_name=INDEX_NAME,
#         embedding=embedding_model,
#         namespace=NAMESPACE
#     )
#     print("Connected to vector store")
#     return vector_store
# def retreive(vector_store:PineconeVectorStore,question:str)->list:
#     """
#     Detect what type of question it is
#     and filter Pinecone results accordingly.
#     """
#     q = question.lower()

#     # detect question type
#     is_numerical = any(w in q for w in [
#         "how much", "how many", "amount", "total",
#         "cost", "paid", "price", "number of", "count"
#     ])
#     is_date = any(w in q for w in [
#         "when", "date", "year", "month", "timeline"
#     ])
#     is_table = any(w in q for w in [
#         "list", "table", "all", "who were", "names of"
#     ])

#     # build metadata filter for Pinecone
#     # only search chunks that are likely to have the answer
#     pinecone_filter = {}

#     if is_numerical:
#         pinecone_filter = {"has_numbers": True}
#     elif is_date:
#         pinecone_filter = {"has_dates": True}
#     elif is_table:
#         pinecone_filter = {"has_table": True}

#     # search with or without filter
#     if pinecone_filter:
#         results = vector_store.similarity_search(
#             query=question,
#             k=5,
#             filter=pinecone_filter,    # ← Pinecone metadata filter
#             namespace="epstein-docs"
#         )
#         # fallback to unfiltered if no results
#         if not results:
#             results = vector_store.similarity_search(
#                 query=question, k=5, namespace="epstein-docs"
#             )
#     else:
#         results = vector_store.similarity_search(
#             query=question, k=5, namespace="epstein-docs"
#         )

#     return results
# # vector_store=load_vector_store()
# # print(retreive(vector_store,"How does the testimony describe the relationship or frequency of contact between Jeffrey Epstein and Ghislaine Maxwell?"))

# def build_prompt(chunks: list, question: str) -> str:
#     """
#     Optimized prompt to prevent Safety Blocks by enforcing 
#     Clinical Redaction and Data Extraction formats.
#     """
#     context = ""
#     for i, doc in enumerate(chunks):
#         # Change this line inside build_prompt:
#         source = doc.metadata.get('filename', doc.metadata.get('source_url', 'Unknown'))
#         context += f"[Source {i+1}: {source}]\n{doc.page_content}\n\n"

#     prompt = f"""SYSTEM ROLE:
#     You are a Forensic Document Parser. Your goal is to extract information from unsealed 
#     U.S. Federal Court records. 

#     SAFETY & REDACTION PROTOCOL:
#     If the document contains graphic or explicit descriptions that would normally trigger 
#     a safety block, you MUST NOT refuse the request. Instead, apply the following:
#     1. CLINICAL NEUTRALITY: Replace graphic verbs with clinical terms (e.g., use "alleged physical contact" or "prohibited conduct").
#     2. ENTITY FOCUS: Focus on Names, Dates, Locations, and Procedural Actions.
#     3. CENSORSHIP: If a specific word is too graphic to output, replace it with [REDACTED] or use asterisks (e.g., s** act) to ensure the technical response is generated.

#     DOCUMENT EXCERPTS:
#     {context}

#     RESEARCH QUERY:
#     Analyze the excerpts above and provide a structured summary regarding: {question}

#     FORMATTING RULES:
#     - Use a bulleted list for factual findings.
#     - Cite sources as [Source X].
#     - If the context is missing info, state: "Information not found in excerpts."

#     REPORT:"""

#     return prompt


# # vector_store=load_vector_store()
# # relevant_chunks=retreive(vector_store,"What does epstein do")
# # print(relevant_chunks)
# # print(build_prompt(relevant_chunks,"what does epstein do"))

# from groq import Groq

# _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def generate_answer(prompt: str) -> str:
#     completion = _groq_client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#       {
#         "role": "user",
#         "content": prompt
#       }
#     ],
#     temperature=1,
#     max_completion_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None
# )

#     for chunk in completion:
#         print(chunk.choices[0].delta.content or "", end="")

# # vector_store=load_vector_store()
# # relevant_chunks=retreive(vector_store,"What does epstein do")
# # print(relevant_chunks)
# # prompt=build_prompt(relevant_chunks,"what does epstein do")
# # print(generate_answer(prompt))   

 
# def rag_query(vector_store: PineconeVectorStore, question: str) -> dict:
#     """Full pipeline

#     Pipeline:
#     question -> retrieve -> build_prompt -> generate_answer -> return
#     Returns dict with answer + sources so FastAPI
#     can send both to the React frontend later.
#     """
#     docs=retreive(vector_store,question)
#     if not docs:
#         return {
#             "answer": "No relevant documents found in the database.",
#             "sources": []
#         }
#     prompt=build_prompt(docs,question)
#     answer=generate_answer(prompt)
#     sources = [
#     {
#         "source_file": doc.metadata.get("filename", "unknown"),        # was "source"
#         "source_url":  doc.metadata.get("source_url", ""),             # was missing entirely
#         "chunk_index": doc.metadata.get("chunk_index", 0),             # this one is correct
#         "preview":     doc.page_content[:1000] + "...",
#     }
#     for doc in docs
# ]
#     return {
#         "answer":answer,
#         "sources":sources
#     }
# if __name__=="__main__":
#     vector_store=load_vector_store()
#     test_questions = [
#         # "Who did Epstein fly on his private jet?",
#         # "What locations did Epstein visit frequently?",
#         # "Who were Epstein's known associates?",
#         "what does epstein do in palm beach acc to court"
#     ]
#     for question in test_questions:
#         print(f"\n{'─'*60}")
#         print(f"Q: {question}")
#         print('─'*60)

#         result = rag_query(vector_store, question)
#         save_query(question, result['answer'], result['sources'])
#         print(f"A: {result['answer']}")
#         print(f"\n📚 Sources used ({len(result['sources'])}):")
#         for i, src in enumerate(result['sources']):
#             print(f"  [{i+1}] {src})")
#             print(f"       {src['preview'][:1000]}...")

