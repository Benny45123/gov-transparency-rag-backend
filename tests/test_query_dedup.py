from database import build_query_hash, normalize_query_text


def test_normalize_query_text_dedups_case_spacing_and_punctuation():
    assert normalize_query_text("  What happened to Epstein? ") == normalize_query_text(
        "what   happened to epstein"
    )


def test_query_hash_is_namespace_scoped():
    first = build_query_hash("What happened?", namespace="epstein-docs")
    second = build_query_hash("What happened?", namespace="other-docs")

    assert first != second
