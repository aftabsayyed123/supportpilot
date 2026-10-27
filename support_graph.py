import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ValidationError

import agent
import rag

load_dotenv()
client = OpenAI()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
ROUTER_HISTORY = 6

HANDOFF_REPLY = "I'll pass this to our support team, who will get back to you shortly."

ROUTER_PROMPT = """You route customer messages for NovaCart's support team.
Classify the customer's LATEST message, using the earlier conversation for context.
- "order_status": about one of the customer's own orders: where it is, its status, delivery date or tracking.
  Also use this when the customer is replying with an order number or email so an order can be looked up.
- "policy": a general question about NovaCart's rules: delivery times, returns, refunds, cancellations,
  changing an address, payments or gift cards.
- "other": anything else.
Return ONLY a JSON object: {"intent": "order_status" | "policy" | "other"}"""


class SupportState(TypedDict, total=False):
    messages: list[dict]
    intent: str
    reply: str
    needs_human: bool
    sources: list[dict]
    trace: Annotated[list[str], operator.add]


class Route(BaseModel):
    intent: Literal["order_status", "policy", "other"]


def parse_intent(raw: str | None) -> str:
    try:
        return Route.model_validate_json(raw or "").intent
    except ValidationError:
        return "other"


def classify_intent(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role": "system", "content": ROUTER_PROMPT}, *messages[-ROUTER_HISTORY:]],
    )
    return parse_intent(response.choices[0].message.content)


def latest_question(state: SupportState) -> str:
    return state["messages"][-1]["content"]


# ---------- nodes ----------

def router(state: SupportState) -> dict:
    return {"intent": classify_intent(state["messages"]), "trace": ["router"]}


def order_lookup(state: SupportState) -> dict:
    reply = agent.chat(state["messages"])
    return {"reply": reply, "needs_human": False, "sources": [], "trace": ["order_lookup"]}


def policy_answer(state: SupportState) -> dict:
    result = rag.answer_question(latest_question(state))
    if result is None:
        return {"reply": HANDOFF_REPLY, "needs_human": True, "sources": [], "trace": ["policy_answer"]}
    return {
        "reply": result.answer,
        "needs_human": result.needs_human,
        "sources": [source.model_dump() for source in result.sources],
        "trace": ["policy_answer"],
    }


def handoff(state: SupportState) -> dict:
    return {"reply": HANDOFF_REPLY, "needs_human": True, "sources": [], "trace": ["handoff"]}


# ---------- routing ----------

def pick_next(state: SupportState) -> str:
    return state["intent"]


def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("router", router)
    builder.add_node("order_lookup", order_lookup)
    builder.add_node("policy_answer", policy_answer)
    builder.add_node("handoff", handoff)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        pick_next,
        {"order_status": "order_lookup", "policy": "policy_answer", "other": "handoff"},
    )
    builder.add_edge("order_lookup", END)
    builder.add_edge("policy_answer", END)
    builder.add_edge("handoff", END)

    return builder.compile()


graph = build_graph()


def run_turn(history: list[dict]) -> SupportState:
    return graph.invoke({"messages": history, "trace": []})


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    history = []
    print("NovaCart support (graph). Type 'quit' to exit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        history.append({"role": "user", "content": user_text})
        result = run_turn(history)
        print(f"  [path] {' -> '.join(result['trace'])} | needs_human: {result['needs_human']}")
        print(f"Bot: {result['reply']}")
        for source in result.get("sources", []):
            print(f"  SOURCE: {source['source']} > {source['section']}")
        history.append({"role": "assistant", "content": result["reply"]})