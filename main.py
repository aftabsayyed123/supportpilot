from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field

from classify import classify_email,EmailClassification
app=FastAPI(title="SupportPilot API",version="0.1.0")
class EmailRequest(BaseModel):
    text: str =Field(min_length=1,max_length=5000)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/classify",response_model=EmailClassification)
def classify(request: EmailRequest):
    result= classify_email(request.text)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid or incomplete response",
        )
    return result