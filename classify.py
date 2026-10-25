import os
import json
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field,ValidationError
load_dotenv()
client=OpenAI()
MODEL=os.getenv("MODEL","gpt-4o-mini")
class EmailClassification(BaseModel):
    category:Literal["order_status","refund","technical","complaint","other"]
    urgency:Literal["low","medium","high"]
    needs_human:bool
    summary:str=Field(max_length=200)

SYSTEM_PROMPT="""You classify customer emails for NovaCart, an online store.
Return ONLY a JSON object with exactly these keys:
- "category": one of "order_status", "refund", "technical", "complaint", "other"
- "urgency": one of "low", "medium", "high"
- "needs_human": true if the customer is angry, threatens legal action, or the request is unclear; otherwise false
- "summary": one short sentence describing the request
Do not add any other keys or text."""

def classify_email(email_text:str) -> EmailClassification | None:
    response=client.chat.completions.create(
        model=MODEL,
        response_format= {"type":"json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
    )

    choice=response.choices[0]
    if choice.finish_reason != "stop":
        print(f"Warning: reply did not finish properly ({choice.finish_reason})")
        return None

    raw_json= choice.message.content
    try:
        data=json.loads(raw_json)
        return EmailClassification.model_validate(data)
    except (json.JSONDecodeError,ValidationError) as error:
            print("AI returned invalid data",raw_json)
            print(error)
            return None

if __name__ == "__main__":
    test_email=[
        "Hi, where is my order #4521? It's been 10 days and no update.",
        "I was charged TWICE for the same order. Fix this today or I'm contacting my lawyer.",
        "The app crashes every time I open my cart on iPhone.",
        "Do you sell gift cards?",
        "asdf jkl ???",
        "मेरा ऑर्डर 10 दिन से नहीं आया, पैसे वापस करो"
    ]
    for email in test_email:
        result=classify_email(email)
        print(f"\nEMAIL:",email)
        if result:
            print(f"RESULT:",result.model_dump())
            print("Route to human?",result.needs_human)
