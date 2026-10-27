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