from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

import main
from classify import EmailClassification
from rag import Answer, Source

client = TestClient(main.app)
VALID_HEADERS = {"X-API-Key": main.API_KEY}

FAKE_RESULT = EmailClassification(
    category="refund",
    urgency="high",
    needs_human=True,
    summary="Customer was charged twice.",
)

FAKE_ANSWER = Answer(
    answer="The address can only be changed before the order ships.",
    status="answered",
    needs_human=False,
    reason="answered",
    sources=[Source(source="orders.md", section="Changing the delivery address", score=0.588)],
)

# ---------- /health ----------

def test_health_is_public():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- /classify ----------

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


# ---------- /answer ----------

def test_answer_rejects_missing_key():
    response = client.post("/answer", json={"question": "Can I change my address?"})
    assert response.status_code == 401


def test_answer_rejects_empty_question():
    response = client.post("/answer", json={"question": ""}, headers=VALID_HEADERS)
    assert response.status_code == 422


def test_answer_success(monkeypatch):
    monkeypatch.setattr(main, "answer_question", lambda question: FAKE_ANSWER)
    response = client.post(
        "/answer",
        json={"question": "Can I change my address?"},
        headers=VALID_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["source"] == "orders.md"


def test_answer_returns_503_when_qdrant_is_down(monkeypatch):
    def qdrant_down(question):
        raise ResponseHandlingException(ConnectionError("Qdrant is down"))

    monkeypatch.setattr(main, "answer_question", qdrant_down)
    response = client.post(
        "/answer",
        json={"question": "Can I change my address?"},
        headers=VALID_HEADERS,
    )
    assert response.status_code == 503