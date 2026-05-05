from langchain_core.documents import Document

from query import _rewrite_for_retrieval, rag_query
from retriever import RetrievedChunk, expand_query, merge_chunks, retrieve_chunks


class FakeStore:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def similarity_search(self, *, query, k, namespace):
        self.calls.append(query)
        return self.responses.get(query, [])


def test_expand_query_adds_domain_synonyms():
    expanded = expand_query("How many children were victimized?")

    assert "children" in expanded
    assert "victimized" in expanded
    assert "minor" in expanded
    assert "abused" in expanded


def test_merge_chunks_deduplicates_on_source_identity():
    first = RetrievedChunk("A", "doc.pdf", "url", 1)
    duplicate = RetrievedChunk("A duplicate", "doc.pdf", "other", 1)
    second = RetrievedChunk("B", "doc.pdf", "url", 2)

    merged = merge_chunks([first, duplicate], [second])

    assert [(chunk.filename, chunk.chunk_index) for chunk in merged] == [
        ("doc.pdf", 1),
        ("doc.pdf", 2),
    ]


def test_retrieve_chunks_uses_expanded_query_for_victimized_children_case():
    question = "How many children were victimized?"
    expanded = expand_query(question)
    store = FakeStore(
        {
            question: [],
            expanded: [
                Document(
                    page_content="Court filings describe child victims and abuse counts.",
                    metadata={"filename": "case.pdf", "chunk_index": 7, "source_url": "https://example.test/case"},
                )
            ],
        }
    )

    chunks = retrieve_chunks(store, question, top_k=3)

    assert len(chunks) == 1
    assert store.calls == [question, expanded]
    assert chunks[0].filename == "case.pdf"


def test_rag_query_does_not_return_no_context_when_chunks_exist(monkeypatch):
    expected_chunk = RetrievedChunk(
        content="The filing states that multiple child victims were identified.",
        filename="case.pdf",
        source_url="https://example.test/case",
        chunk_index=2,
    )

    monkeypatch.setattr("query.retrieve_chunks", lambda store, question: [expected_chunk])
    monkeypatch.setattr("query.rerank_chunks", lambda chunks, question: chunks)
    monkeypatch.setattr(
        "query.generate_answer",
        lambda system_prompt, user_message, history=None: "Answer with cited child-victim context.",
    )

    class DummyCache:
        def get(self, key):
            return None

        def set(self, key, payload):
            return None

    monkeypatch.setattr("query.get_cache", lambda: DummyCache())
    monkeypatch.setattr("query._persist_async", lambda **kwargs: None)

    response = rag_query(object(), "How many children were victimized?")

    assert response.answer != "No relevant documents were found in the database for this query."
    assert response.sources[0]["source_file"] == "case.pdf"


def test_rag_query_rewrites_follow_up_before_retrieval(monkeypatch):
    expected_chunk = RetrievedChunk(
        content="The filing identifies Palm Beach as the location.",
        filename="case.pdf",
        source_url="https://example.test/case",
        chunk_index=3,
    )
    calls = {}

    monkeypatch.setattr(
        "query.rewrite_query",
        lambda question, history: "Where did Epstein operate in Palm Beach?",
    )

    def fake_retrieve(store, question):
        calls["retrieval_query"] = question
        return [expected_chunk]

    monkeypatch.setattr("query.retrieve_chunks", fake_retrieve)
    monkeypatch.setattr("query.rerank_chunks", lambda chunks, question: chunks)
    monkeypatch.setattr(
        "query.generate_answer",
        lambda system_prompt, user_message, history=None: "Answer with rewritten context.",
    )

    class DummyCache:
        def get(self, key):
            return None

        def set(self, key, payload):
            return None

    monkeypatch.setattr("query.get_cache", lambda: DummyCache())
    monkeypatch.setattr("query._persist_async", lambda **kwargs: None)

    response = rag_query(
        object(),
        "Where was that?",
        history=[{"role": "user", "content": "What does Epstein do in Palm Beach?"}],
    )

    assert calls["retrieval_query"] == "Where did Epstein operate in Palm Beach?"
    assert response.answer == "Answer with rewritten context."


def test_continuation_request_uses_last_substantive_user_question(monkeypatch):
    monkeypatch.setattr(
        "query.rewrite_query",
        lambda question, history: "wrong vague model rewrite",
    )

    rewritten = _rewrite_for_retrieval(
        "extend my previous conversation information",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Who were Jeffrey Epstein's companions?"},
            {"role": "assistant", "content": "Companions included Ghislaine Maxwell."},
        ],
    )

    assert rewritten == "Who were Jeffrey Epstein's companions? more details"


def test_greeting_does_not_hit_retrieval(monkeypatch):
    def fail_retrieve(store, question):
        raise AssertionError("pure greeting should not retrieve documents")

    class DummyCache:
        def get(self, key):
            return None

        def set(self, key, payload):
            return None

    monkeypatch.setattr("query.retrieve_chunks", fail_retrieve)
    monkeypatch.setattr("query.get_cache", lambda: DummyCache())
    monkeypatch.setattr("query._persist_async", lambda **kwargs: None)

    response = rag_query(object(), "hello")

    assert response.sources == []
    assert "Ask me a question" in response.answer


def test_continuation_without_history_asks_for_context(monkeypatch):
    def fail_retrieve(store, question):
        raise AssertionError("continuation without memory should not retrieve unrelated documents")

    class DummyCache:
        def get(self, key):
            return None

        def set(self, key, payload):
            return None

    monkeypatch.setattr("query.retrieve_chunks", fail_retrieve)
    monkeypatch.setattr("query.get_cache", lambda: DummyCache())
    monkeypatch.setattr("query._persist_async", lambda **kwargs: None)

    response = rag_query(object(), "extend my previous conversation information")

    assert response.sources == []
    assert response.error == "missing_history"
    assert "conversation_id" in response.answer
