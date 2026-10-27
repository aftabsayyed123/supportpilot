import rag


def test_no_relevant_chunks_hands_off_without_calling_ai(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question: [])

    def ai_must_not_be_called(**kwargs):
        raise AssertionError("The AI should not be called when nothing is relevant")

    monkeypatch.setattr(rag.openai_client.chat.completions, "create", ai_must_not_be_called)

    result = rag.answer_question("What is the capital of France?")
    assert result.status == "not_found"
    assert result.needs_human is True
    assert result.sources == []


def test_threshold_drops_weak_matches(monkeypatch):
    class FakeHit:
        def __init__(self, score):
            self.score = score
            self.payload = {"source": "x.md", "section": "y", "text": "z"}

    class FakeResult:
        points = [FakeHit(0.62), FakeHit(0.31), FakeHit(0.12)]

    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0]])
    monkeypatch.setattr(rag.qdrant, "query_points", lambda **kwargs: FakeResult())

    hits = rag.retrieve("anything")
    assert [hit.score for hit in hits] == [0.62, 0.31]