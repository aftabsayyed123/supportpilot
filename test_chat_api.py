import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
HEADERS = {"X-API-Key": "test-key"}

ANSWERED = {
    "reply": "Order 4521 has shipped.",
    "needs_human": False,
    "sources": [],
    "trace": ["planner", "order_lookup", "compose"],
}
REVIEW = {"customer_message": "My order arrived broken", "draft": "We're so sorry."}


@pytest.fixture(autouse=True)
def test_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "test-key")


def test_chat_requires_api_key():
    response = client.post("/chat", json={"customer_id": "ravi", "message": "Hi"})
    assert response.status_code == 401


def test_chat_rejects_unsafe_customer_id():
    response = client.post("/chat", json={"customer_id": "ravi/../admin", "message": "Hi"}, headers=HEADERS)
    assert response.status_code == 422


def test_chat_returns_answer(monkeypatch):
    monkeypatch.setattr(main, "pending_review", lambda customer_id: None)
    monkeypatch.setattr(main, "run_turn", lambda text, customer_id: ANSWERED)

    response = client.post("/chat", json={"customer_id": "ravi", "message": "Where is 4521?"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["reply"] == "Order 4521 has shipped."


def test_chat_pauses_for_review_and_hides_old_reply(monkeypatch):
    reviews = iter([None, REVIEW])
    monkeypatch.setattr(main, "pending_review", lambda customer_id: next(reviews))
    monkeypatch.setattr(
        main, "run_turn",
        lambda text, customer_id: {"reply": "OLD REPLY", "trace": ["planner", "draft_reply"]},
    )

    body = client.post("/chat", json={"customer_id": "ravi", "message": "It arrived broken"}, headers=HEADERS).json()
    assert body["status"] == "pending_review"
    assert body["reply"] == main.REVIEW_PENDING_REPLY
    assert body["draft"] == "We're so sorry."
    assert "OLD REPLY" not in body["reply"]


def test_chat_while_review_waiting_does_not_run_graph(monkeypatch):
    def must_not_run(text, customer_id):
        raise AssertionError("graph must not run while a review is pending")

    monkeypatch.setattr(main, "pending_review", lambda customer_id: REVIEW)
    monkeypatch.setattr(main, "run_turn", must_not_run)

    body = client.post("/chat", json={"customer_id": "ravi", "message": "Hello?"}, headers=HEADERS).json()
    assert body["status"] == "pending_review"
    assert body["reply"] == main.STILL_REVIEWING_REPLY


def test_get_review_404_when_nothing_waiting(monkeypatch):
    monkeypatch.setattr(main, "pending_review", lambda customer_id: None)
    response = client.get("/reviews/ravi", headers=HEADERS)
    assert response.status_code == 404


def test_submit_review_returns_ticket(monkeypatch):
    received = {}

    def fake_resume(customer_id, decision):
        received.update(decision)
        return {
            "reply": "We're so sorry. Your ticket number is T-ABC123.",
            "needs_human": True,
            "ticket_id": "T-ABC123",
            "trace": ["planner", "draft_reply", "human_review", "create_ticket"],
        }

    monkeypatch.setattr(main, "pending_review", lambda customer_id: REVIEW)
    monkeypatch.setattr(main, "resume_review", fake_resume)

    body = client.post("/reviews/ravi", json={"action": "approve"}, headers=HEADERS).json()
    assert body["status"] == "ticket_created"
    assert body["ticket_id"] == "T-ABC123"
    assert received == {"action": "approve", "text": ""}


def test_submit_review_rejects_unknown_action():
    response = client.post("/reviews/ravi", json={"action": "publish"}, headers=HEADERS)
    assert response.status_code == 422