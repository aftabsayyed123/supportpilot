import agent
import rag
import support_graph as sg
from rag import Answer, Source


def ask(text: str):
    return sg.run_turn([{"role": "user", "content": text}])


def test_order_question_goes_to_order_lookup(monkeypatch):
    monkeypatch.setattr(sg, "classify_intent", lambda messages: "order_status")
    monkeypatch.setattr(agent, "chat", lambda messages: "Your order 4521 has shipped.")

    result = ask("Where is order 4521?")
    assert result["trace"] == ["router", "order_lookup"]
    assert result["reply"] == "Your order 4521 has shipped."
    assert result["needs_human"] is False


def test_policy_question_returns_sources(monkeypatch):
    monkeypatch.setattr(sg, "classify_intent", lambda messages: "policy")
    fake_answer = Answer(
        answer="Electronics have a 7-day return window.",
        status="answered",
        needs_human=False,
        reason="answered",
        sources=[Source(source="returns.md", section="Return window", score=0.6)],
    )
    monkeypatch.setattr(rag, "answer_question", lambda question: fake_answer)

    result = ask("Can I return headphones?")
    assert result["trace"] == ["router", "policy_answer"]
    assert result["sources"][0]["source"] == "returns.md"


def test_other_goes_to_handoff(monkeypatch):
    monkeypatch.setattr(sg, "classify_intent", lambda messages: "other")

    result = ask("Write me a poem about cats")
    assert result["trace"] == ["router", "handoff"]
    assert result["needs_human"] is True


def test_parse_intent_falls_back_to_other():
    assert sg.parse_intent('{"intent": "policy"}') == "policy"
    assert sg.parse_intent('{"intent": "refund"}') == "other"
    assert sg.parse_intent("not json") == "other"
    assert sg.parse_intent(None) == "other"