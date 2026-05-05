import pytest
from pydantic import ValidationError

from generator import build_chat_messages, build_prompt
from main import QueryRequest
from query import _cache_key, _resolve_history, _scoped_cache_key


def test_build_chat_messages_preserves_history_order():
    messages = build_chat_messages(
        "system prompt",
        "latest question with context",
        history=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ],
    )

    assert messages == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
        {"role": "user", "content": "latest question with context"},
    ]


def test_cache_key_changes_with_history():
    base = _cache_key("What happened?", history=[])
    with_history = _cache_key(
        "What happened?",
        history=[{"role": "user", "content": "Tell me about the victims"}],
    )

    assert base != with_history


def test_scoped_cache_key_changes_with_conversation_id():
    first = _scoped_cache_key("What happened?", conversation_id="chat-a")
    second = _scoped_cache_key("What happened?", conversation_id="chat-b")

    assert first != second


def test_resolve_history_prefers_explicit_messages():
    class DummyCache:
        def get_conversation_history(self, conversation_id, *, namespace):
            raise AssertionError("cache should not be used when explicit history is present")

    history = _resolve_history(
        DummyCache(),
        explicit_history=[{"role": "user", "content": "explicit question"}],
        conversation_id="chat-a",
        namespace="epstein-docs",
    )

    assert history == [{"role": "user", "content": "explicit question"}]


def test_resolve_history_uses_conversation_cache_before_db(monkeypatch):
    class DummyCache:
        def get_conversation_history(self, conversation_id, *, namespace):
            return [{"role": "user", "content": "cached question"}]

    monkeypatch.setattr(
        "query.fetch_history",
        lambda **kwargs: pytest.fail("db should not be used on cache hit"),
    )

    history = _resolve_history(
        DummyCache(),
        explicit_history=[],
        conversation_id="chat-a",
        namespace="epstein-docs",
    )

    assert history == [{"role": "user", "content": "cached question"}]


def test_resolve_history_falls_back_to_db_and_warms_cache(monkeypatch):
    cached = {}

    class DummyCache:
        def get_conversation_history(self, conversation_id, *, namespace):
            return []

        def set_conversation_history(self, conversation_id, history, *, namespace):
            cached["conversation_id"] = conversation_id
            cached["history"] = history

    monkeypatch.setattr(
        "query.fetch_history",
        lambda **kwargs: [
            {"question": "follow up", "answer": "second answer", "error": None},
            {"question": "first question", "answer": "first answer", "error": None},
        ],
    )

    history = _resolve_history(
        DummyCache(),
        explicit_history=[],
        conversation_id="chat-a",
        namespace="epstein-docs",
    )

    assert history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "second answer"},
    ]
    assert cached["conversation_id"] == "chat-a"
    assert cached["history"] == history


def test_query_request_rejects_invalid_chat_role():
    with pytest.raises(ValidationError):
        QueryRequest(
            question="test question",
            messages=[{"role": "system", "content": "not allowed"}],
        )


def test_system_prompt_includes_intent_handling_guidance():
    system_prompt, _ = build_prompt([], "hello")

    assert "Greeting or courtesy only" in system_prompt
    assert "Clarification, continuation, or expansion requests" in system_prompt
    assert "expand my previous conversation" in system_prompt
    assert "Off-topic requests" in system_prompt
