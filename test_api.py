from fastapi.testclient import TestClient

import main
from classify import EmailClassification

client = TestClient(main.app)
VALID_HEADERS = {"X-API-Key": main.API_KEY}

FAKE_RESULT = EmailClassification(
    category="refund",
    urgency="high",
    needs_human=True,
    summary="Customer was charged twice.",
)


def test_health_is_public():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_rejects_missing_key():
    response = client.post("/classify", json={"text": "Where is my order?"})
    assert response.status_code == 401


def test_classify_rejects_wrong_key():
    response = client.post(
        "/classify",
        json={"text": "Where is my order?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_classify_rejects_empty_text():
    response = client.post("/classify", json={"text": ""}, headers=VALID_HEADERS)
    assert response.status_code == 422


def test_classify_success(monkeypatch):
    monkeypatch.setattr(main, "classify_email", lambda text: FAKE_RESULT)
    response = client.post(
        "/classify",
        json={"text": "I was charged twice"},
        headers=VALID_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["category"] == "refund"
    assert response.json()["needs_human"] is True


def test_classify_returns_502_when_ai_fails(monkeypatch):
    monkeypatch.setattr(main, "classify_email", lambda text: None)
    response = client.post(
        "/classify",
        json={"text": "I was charged twice"},
        headers=VALID_HEADERS,
    )
    assert response.status_code == 502