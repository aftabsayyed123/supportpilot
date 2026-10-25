import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client=OpenAI()
response=client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {"role": "system", "content": "You are a support assistant for an online store. "
                                      "You do NOT have access to order data. Never invent tracking info, dates or emails. "
                                      "If asked about a specific order, ask for the order email and say a human will check it. "
                                      "Reply in 2 sentences."}
        ,
            {"role": "user", "content": "Where is my order #4521? It's been 10 days."},
            ],
    temperature=0.3,
)

print(response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)