import os

import pytest

from llm_judge import evaluate_response, load_default_questions
from query import rag_query
from retriever import build_vector_store


def _require_live_env():
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run live RAG judge tests.")
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY is required for live retrieval tests.")
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("A judge API key is required.")


@pytest.mark.live
def test_default_domain_questions_pass_llm_judge():
    _require_live_env()
    store = build_vector_store()

    for case in load_default_questions():
        response = rag_query(store, case["question"], skip_cache=True)
        assert response.sources or not case.get("expect_context", True)

        verdict = evaluate_response(
            case["question"],
            response.answer,
            response.sources,
        )

        assert verdict.relevant, verdict.reason
        assert verdict.consistent_with_sources, verdict.reason
        if case.get("expect_context", True):
            assert verdict.score_0_to_1 >= 0.6, verdict.reason
        for banned_phrase in case.get("must_not_include", []):
            assert banned_phrase not in response.answer


@pytest.mark.live
def test_children_victim_queries_are_consistent_under_judge():
    _require_live_env()
    store = build_vector_store()
    cases = [
        case for case in load_default_questions()
        if case.get("consistency_group") == "children_victims"
    ]

    responses = [rag_query(store, case["question"], skip_cache=True) for case in cases]
    combined_answer = "\n\n".join(
        f"Q: {case['question']}\nA: {response.answer}"
        for case, response in zip(cases, responses, strict=True)
    )
    combined_sources = []
    for response in responses:
        combined_sources.extend(response.sources)

    verdict = evaluate_response(
        "Are these two answers semantically consistent for the same topic?",
        combined_answer,
        combined_sources,
    )

    assert verdict.correct, verdict.reason
    assert verdict.consistent_with_sources, verdict.reason
