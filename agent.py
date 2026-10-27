import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOLS

load_dotenv()
client = OpenAI()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_STEPS = 5

FALLBACK_REPLY = "Sorry, I couldn't complete that. I'm passing your question to our support team."

SYSTEM_PROMPT = """You are NovaCart's customer support assistant.
Rules:
- For questions about a specific order, use the get_order_status tool.
- You need BOTH the order number and the email address used for the order. If either is missing, ask the customer for it. Never guess or invent them.
- Only state order details that the tool returned. Never invent dates, statuses, amounts or tracking information.
- If the tool finds no order, say so politely and ask the customer to double-check the order number and email.
- Keep replies to 2 to 3 friendly sentences."""


def run_tool(name: str, arguments_json: str) -> str:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        arguments = json.loads(arguments_json)
        result = function(**arguments)
    except (json.JSONDecodeError, TypeError) as error:
        result = {"error": f"Invalid arguments: {error}"}
    return json.dumps(result)


def chat(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or FALLBACK_REPLY

        messages.append(message)
        for call in message.tool_calls:
            print(f"  [tool call] {call.function.name}({call.function.arguments})")
            result = run_tool(call.function.name, call.function.arguments)
            print(f"  [tool result] {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return FALLBACK_REPLY


if __name__ == "__main__":
    history = []
    print("NovaCart support. Type 'quit' to exit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        history.append({"role": "user", "content": user_text})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}")