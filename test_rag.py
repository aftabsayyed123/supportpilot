import rag


class FakeHit:
    def __init__(self, score, source="x.md", section="y"):
        self.score = score
        self.payload = {"source": source, "section": section, "text": "z"}


def ai_must_not_be_called(**kwargs):
    raise AssertionError("The AI should not be called here")


def test_long_irrelevant_question_hands_off_without_calling_ai(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question: [])
    monkeypatch.setattr(rag.openai_client.chat.completions, "create", ai_must_not_be_called)

    result = rag.answer_question("What is the capital of France?")
    assert result.status == "not_found"
    assert result.reason == "below_threshold"
    assert result.needs_human is True


def test_short_vague_question_asks_to_clarify_without_calling_ai(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question: [])
    monkeypatch.setattr(rag.openai_client.chat.completions, "create", ai_must_not_be_called)

    result = rag.answer_question("money")
    assert result.status == "clarify"
    assert result.reason == "too_vague"
    assert result.needs_human is False


def test_threshold_drops_weak_matches(monkeypatch):
    class FakeResult:
        points = [FakeHit(0.62), FakeHit(0.31), FakeHit(0.12)]

    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0]])
    monkeypatch.setattr(rag.qdrant, "query_points", lambda **kwargs: FakeResult())

    hits = rag.retrieve("anything")
    assert [hit.score for hit in hits] == [0.62, 0.31]


def test_select_sources_keeps_only_used_valid_excerpts():
    hits = [FakeHit(0.7, "a.md"), FakeHit(0.5, "b.md"), FakeHit(0.4, "c.md")]
    sources = rag.select_sources(hits, [2, 2, 9])
    assert [s.source for s in sources] == ["b.md"]


def test_select_sources_falls_back_to_all_hits():
    hits = [FakeHit(0.7, "a.md"), FakeHit(0.5, "b.md")]
    sources = rag.select_sources(hits, [])
    assert [s.source for s in sources] == ["a.md", "b.md"]