import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import agent
import rag

load_dotenv()
client = OpenAI()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
PLANNER_HISTORY = 6

HANDOFF_REPLY = (
    "I'm sorry about this. I'm passing your message to our support team, "
    "who will get back to you shortly."
)

ESCALATION_WORDS = (
    "broken", "damaged", "defective", "faulty", "missing", "wrong item",
    "lawyer", "legal", "court", "fraud",
)

PLANNER_PROMPT = """You plan how NovaCart's support assistant should answer the customer's LATEST message.
Use the earlier conversation for context.
Skills:
- "order": the customer asks about one of their own orders (status, delivery date, tracking),
  or is replying with an order number or email so an order can be looked up.
- "policy": the message asks about NovaCart's rules: delivery times, returns, refunds,
  cancellations, address changes, payments or gift cards.
A message can need both skills, or neither.
Set "escalate" to true if the item is damaged, broken, faulty, missing or the wrong item,
or if the customer is angry or mentions legal action. These always go to a human.
Return ONLY a JSON object: {"skills": ["order", "policy"], "escalate": false}"""

COMBINE_PROMPT = """You write one reply to a NovaCart customer by combining two prepared answers.
Rules:
- Use ONLY facts from the two answers. Do not add new facts, dates, amounts or policies.
- Keep every specific number, date and action from both answers.
- Write plain text only, with no markdown, in 3 to 5 friendly sentences."""


def merge_trace(existing: list[str], new: list[str] | None) -> list[str]:
    if new is None:
        return []
    return existing + new


class SupportState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    skills: list[str]
    escalate: bool
    order_result: str | None
    policy_result: dict | None
    reply: str
    needs_human: bool
    sources: list[dict]
    trace: Annotated[list[str], merge_trace]


class Plan(BaseModel):
    skills: list[Literal["order", "policy"]] = Field(default_factory=list)
    escalate: bool = False


def parse_plan(raw: str | None) -> Plan:
    try:
        return Plan.model_validate_json(raw or "")
    except ValidationError:
        return Plan(escalate=True)


def plan_turn(messages: list[dict]) -> Plan:
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role": "system", "content": PLANNER_PROMPT}, *messages[-PLANNER_HISTORY:]],
    )
    return parse_plan(response.choices[0].message.content)


def mentions_escalation(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in ESCALATION_WORDS)


def combine_answers(question: str, order_answer: str, policy_answer: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": COMBINE_PROMPT},
            {
                "role": "user",
                "content": f"Customer message: {question}\n\n"
                           f"Order answer: {order_answer}\n\n"
                           f"Policy answer: {policy_answer}",
            },
        ],
    )
    return response.choices[0].message.content or f"{order_answer}\n\n{policy_answer}"


def latest_question(state: SupportState) -> str:
    return state["messages"][-1]["content"]


# ---------- nodes ----------

def planner(state: SupportState) -> dict:
    plan = plan_turn(state["messages"])
    escalate = plan.escalate or mentions_escalation(latest_question(state))
    return {"skills": plan.skills, "escalate": escalate, "trace": ["planner"]}


def order_lookup(state: SupportState) -> dict:
    return {"order_result": agent.chat(state["messages"]), "trace": ["order_lookup"]}


def policy_lookup(state: SupportState) -> dict:
    result = rag.answer_question(latest_question(state))
    if result is None:
        policy = {"answer": HANDOFF_REPLY, "needs_human": True, "sources": []}
    else:
        policy = result.model_dump()
    return {"policy_result": policy, "trace": ["policy_lookup"]}


def compose(state: SupportState) -> dict:
    order = state.get("order_result")
    policy = state.get("policy_result")

    if order and policy:
        reply = combine_answers(latest_question(state), order, policy["answer"])
    elif order:
        reply = order
    elif policy:
        reply = policy["answer"]
    else:
        reply = HANDOFF_REPLY

    return {
        "reply": reply,
        "needs_human": bool(policy and policy["needs_human"]),
        "sources": policy["sources"] if policy else [],
        "messages": [{"role": "assistant", "content": reply}],
        "trace": ["compose"],
    }


def handoff(state: SupportState) -> dict:
    return {
        "reply": HANDOFF_REPLY,
        "needs_human": True,
        "sources": [],
        "messages": [{"role": "assistant", "content": HANDOFF_REPLY}],
        "trace": ["handoff"],
    }


# ---------- routing ----------

def pick_next(state: SupportState) -> list[str] | str:
    if state["escalate"]:
        return "handoff"
    targets = []
    if "order" in state["skills"]:
        targets.append("order_lookup")
    if "policy" in state["skills"]:
        targets.append("policy_lookup")
    return targets or "handoff"


def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("planner", planner)
    builder.add_node("order_lookup", order_lookup)
    builder.add_node("policy_lookup", policy_lookup)
    builder.add_node("compose", compose)
    builder.add_node("handoff", handoff)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", pick_next, ["order_lookup", "policy_lookup", "handoff"])
    builder.add_edge("order_lookup", "compose")
    builder.add_edge("policy_lookup", "compose")
    builder.add_edge("compose", END)
    builder.add_edge("handoff", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()

TURN_RESET = {"trace": None, "order_result": None, "policy_result": None}


def run_turn(text: str, customer_id: str, app=None) -> SupportState:
    app = app or graph
    config = {"configurable": {"thread_id": customer_id}}
    return app.invoke({"messages": [{"role": "user", "content": text}], **TURN_RESET}, config)


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    customer = input("Customer ID: ").strip() or "guest"
    print("Type '/switch <id>' to change customer, 'quit' to exit.")
    while True:
        text = input(f"\n[{customer}] You: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text.startswith("/switch "):
            customer = text.split(maxsplit=1)[1].strip()
            print(f"Now chatting as {customer}")
            continue
        if not text:
            continue
        result = run_turn(text, customer)
        print(f"  [path] {' -> '.join(result['trace'])} | needs_human: {result['needs_human']}")
        print(f"Bot: {result['reply']}")
        for source in result.get("sources", []):
            print(f"  SOURCE: {source['source']} > {source['section']}")