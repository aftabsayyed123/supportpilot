from typing import Literal

from fastapi import Depends, HTTPException
from openai import APIError
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from support_graph import pending_review, resume_review, run_turn

import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from classify import EmailClassification, classify_email
from rag import Answer, answer_question

load_dotenv()
API_KEY = os.getenv("SUPPORTPILOT_API_KEY")

app = FastAPI(title="SupportPilot API", version="0.3.0")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key is not configured")
    if key is None or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class EmailRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/")
def root():
    return {"service": "SupportPilot API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/classify",
    response_model=EmailClassification,
    dependencies=[Depends(verify_api_key)],
)
def classify(request: EmailRequest):
    result = classify_email(request.text)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid or incomplete response",
        )
    return result


@app.post(
    "/answer",
    response_model=Answer,
    dependencies=[Depends(verify_api_key)],
)
def answer(request: QuestionRequest):
    try:
        result = answer_question(request.question)
    except (ResponseHandlingException, UnexpectedResponse):
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable")
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid or incomplete response",
        )
    return result
# ---------- SupportPilot chat + human review ----------

REVIEW_PENDING_REPLY = (
    "Thanks for your patience. A support specialist is reviewing your message "
    "and will reply here shortly."
)
STILL_REVIEWING_REPLY = (
    "We're still reviewing your previous message. A support specialist will reply here shortly."
)

CustomerId = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")


class ChatRequest(BaseModel):
    customer_id: str = CustomerId
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    status: Literal["answered", "pending_review", "ticket_created"]
    reply: str
    needs_human: bool
    ticket_id: str | None = None
    draft: str | None = None
    sources: list[dict] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    text: str = Field(default="", max_length=2000)


class PendingReview(BaseModel):
    customer_id: str
    customer_message: str
    draft: str | None


def to_chat_response(result: dict) -> ChatResponse:
    ticket_id = result.get("ticket_id")
    return ChatResponse(
        status="ticket_created" if ticket_id else "answered",
        reply=result["reply"],
        needs_human=result.get("needs_human", False),
        ticket_id=ticket_id,
        sources=result.get("sources", []),
        trace=result.get("trace", []),
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest):
    if pending_review(request.customer_id):
        return ChatResponse(status="pending_review", reply=STILL_REVIEWING_REPLY, needs_human=True)

    try:
        result = run_turn(request.message, request.customer_id)
    except (ResponseHandlingException, UnexpectedResponse):
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable. Try again shortly.")
    except APIError:
        raise HTTPException(status_code=502, detail="The AI service failed. Try again shortly.")

    review = pending_review(request.customer_id)
    if review:
        return ChatResponse(
            status="pending_review",
            reply=REVIEW_PENDING_REPLY,
            needs_human=True,
            draft=review["draft"],
            trace=result.get("trace", []),
        )
    return to_chat_response(result)


@app.get("/reviews/{customer_id}", response_model=PendingReview, dependencies=[Depends(verify_api_key)])
def get_review(customer_id: str):
    review = pending_review(customer_id)
    if review is None:
        raise HTTPException(status_code=404, detail="No review is waiting for this customer.")
    return PendingReview(customer_id=customer_id, **review)


@app.post("/reviews/{customer_id}", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def submit_review(customer_id: str, decision: ReviewRequest):
    if pending_review(customer_id) is None:
        raise HTTPException(status_code=404, detail="No review is waiting for this customer.")
    result = resume_review(customer_id, decision.model_dump())
    return to_chat_response(result)