import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = os.getenv("MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = "You are a support assistant for an online store. Reply in 1 sentence."


def ask(messages, temperature=0.3, max_tokens=200):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response


# ---------- Experiment 1: Temperature ----------
print("\n########## EXPERIMENT 1: TEMPERATURE ##########")

apology_request = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Write an apology for a late order."},
]

for temp in [0, 1.0]:
    print(f"\n--- Temperature {temp} ---")
    for run in range(1, 4):
        response = ask(apology_request, temperature=temp)
        print(f"Run {run}: {response.choices[0].message.content}")


# ---------- Experiment 2: Memory ----------
print("\n########## EXPERIMENT 2: MEMORY ##########")

conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Where is my order #4521?"},
    {"role": "assistant", "content": "Your order #4521 shipped on Monday and should arrive by Friday."},
    {"role": "user", "content": "Can I change the delivery address for it?"},
]

response = ask(conversation)
print("With history:", response.choices[0].message.content)

no_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Can I change the delivery address for it?"},
]

response = ask(no_history)
print("Without history:", response.choices[0].message.content)


# ---------- Experiment 3: finish_reason and usage ----------
print("\n########## EXPERIMENT 3: FINISH REASON & USAGE ##########")

response = ask(apology_request)
print("Normal reply:", response.choices[0].message.content)
print("finish_reason:", response.choices[0].finish_reason)
print("usage:", response.usage)

response = ask(apology_request, max_tokens=5)
print("\nCut-off reply:", response.choices[0].message.content)
print("finish_reason:", response.choices[0].finish_reason)