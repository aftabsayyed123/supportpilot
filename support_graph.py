import csv
import operator
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import httpx
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import agent
import rag

load_dotenv()
client = OpenAI()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
PLANNER_HISTORY = 6

HANDOFF_LOG = Path("data/handoffs.csv")
LOG_FIELDS = ["created_at", "ticket_id", "customer_id", "email", "customer_message", "review_action", "reply_sent"]
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

TICKET_WEBHOOK_URL = os.getenv("TICKET_WEBHOOK_URL")
TICKET_WEBHOOK_SECRET = os.getenv("TICKET_WEBHOOK_SECRET", "")
WEBHOOK_TIMEOUT_SECONDS = 5

HANDOFF_REPLY = (
    "I'm sorry about this. I'm passing your message to our support team, "
    "who will get back to you shortly."
)
ASK_FOR_EMAIL = "So our team can reach you, please reply with the email address you used for your order."

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

DRAFT_PROMPT = """You draft a reply to an upset NovaCart customer. A human support agent will review it before it is sent.
Rules:
- Be warm and apologetic, and show you understood the problem.
- Do NOT promise refunds, replacements, compensation, dates or outcomes. Say a support specialist will review the case.
- Do not ask the customer for any information.
- No greeting or sign-off. One paragraph, plain text only, 2 to 3 sentences."""


def merge_trace(existing: list[str], new: list[str] | None) -> list[str]:
    if new is None:
        return []
    return existing + new


class SupportState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    customer_id: str
    skills: list[str]
    escalate: bool
    order_result: str | None
    policy_result: dict | None
    draft: str | None
    review_action: str | None
    ticket_id: str | None
    reply: str
    needs_human: bool
    sources: list[dict]
    trace: Annotated[list[str], merge_trace]


class Plan(BaseModel):
    skills: list[Literal["order", "policy"]] = Field(default_factory=list)
    escalate: bool = False


class ReviewDecision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    text: str = ""


def parse_plan(raw: str | None) -> Plan:
    try:
        return Plan.model_validate_json(raw or "")
    except ValidationError:
        return Plan(escalate=True)


def parse_decision(raw) -> ReviewDecision:
    try:
        decision = ReviewDecision.model_validate(raw)
    except ValidationError:
        return ReviewDecision(action="reject")
    if decision.action == "edit" and not decision.text.strip():
        return ReviewDecision(action="reject")
    return decision


def plan_turn(messages: list[dict]) -> Plan:
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role": "system", "content": PLANNER_PROMPT}, *messages[-PLANNER_HISTORY:]],
    )
    return parse_plan(response.choices[0].message.content)


def draft_text(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[{"role": "system", "content": DRAFT_PROMPT}, *messages[-PLANNER_HISTORY:]],
    )
    return response.choices[0].message.content or HANDOFF_REPLY


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


def mentions_escalation(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in ESCALATION_WORDS)


def find_email(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        match = EMAIL_PATTERN.search(message["content"])
        if match:
            return match.group().lower()
    return None


def join_sentences(first: str, second: str) -> str:
    first = first.strip()
    if first and first[-1] not in ".!?":
        first += "."
    return f"{first} {second}"


def write_csv(row: dict) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not HANDOFF_LOG.exists()
    with HANDOFF_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def send_to_webhook(row: dict) -> bool:
    if not TICKET_WEBHOOK_URL:
        return False
    try:
        response = httpx.post(
            TICKET_WEBHOOK_URL,
            json=row,
            headers={"X-Webhook-Secret": TICKET_WEBHOOK_SECRET},
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError as error:
        print(f"WARNING: ticket {row['ticket_id']} was not sent to n8n: {error}")
        return False


def log_handoff(row: dict) -> None:
    write_csv(row)
    send_to_webhook(row)


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


def draft_reply(state: SupportState) -> dict:
    return {"draft": draft_text(state["messages"]), "trace": ["draft_reply"]}


def human_review(state: SupportState) -> dict:
    raw_decision = interrupt({
        "customer_message": latest_question(state),
        "draft": state["draft"],
    })
    decision = parse_decision(raw_decision)

    if decision.action == "approve":
        reply = state["draft"]
    elif decision.action == "edit":
        reply = decision.text.strip()
    else:
        reply = HANDOFF_REPLY

    return {"reply": reply, "review_action": decision.action, "trace": ["human_review"]}


def handoff(state: SupportState) -> dict:
    return {"reply": HANDOFF_REPLY, "review_action": "none", "trace": ["handoff"]}


def create_ticket(state: SupportState) -> dict:
    email = find_email(state["messages"])
    ticket_id = f"T-{uuid.uuid4().hex[:6].upper()}"

    reply = join_sentences(state["reply"], f"Your ticket number is {ticket_id}.")
    if email is None:
        reply = join_sentences(reply, ASK_FOR_EMAIL)

    log_handoff({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticket_id": ticket_id,
        "customer_id": state.get("customer_id", "unknown"),
        "email": email or "",
        "customer_message": latest_question(state),
        "review_action": state.get("review_action") or "none",
        "reply_sent": reply,
    })

    return {
        "reply": reply,
        "needs_human": True,
        "sources": [],
        "ticket_id": ticket_id,
        "messages": [{"role": "assistant", "content": reply}],
        "trace": ["create_ticket"],
    }


# ---------- routing ----------

def pick_next(state: SupportState) -> list[str] | str:
    if state["escalate"]:
        return "draft_reply"
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
    builder.add_node("draft_reply", draft_reply)
    builder.add_node("human_review", human_review)
    builder.add_node("handoff", handoff)
    builder.add_node("create_ticket", create_ticket)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner", pick_next, ["order_lookup", "policy_lookup", "draft_reply", "handoff"]
    )
    builder.add_edge("order_lookup", "compose")
    builder.add_edge("policy_lookup", "compose")
    builder.add_edge("compose", END)
    builder.add_edge("draft_reply", "human_review")
    builder.add_edge("human_review", "create_ticket")
    builder.add_edge("handoff", "create_ticket")
    builder.add_edge("create_ticket", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()

TURN_RESET = {
    "trace": None,
    "order_result": None,
    "policy_result": None,
    "draft": None,
    "review_action": None,
    "ticket_id": None,
}


def thread_config(customer_id: str) -> dict:
    return {"configurable": {"thread_id": customer_id}}


def run_turn(text: str, customer_id: str, app=None) -> SupportState:
    app = app or graph
    turn_input = {
        "messages": [{"role": "user", "content": text}],
        "customer_id": customer_id,
        **TURN_RESET,
    }
    return app.invoke(turn_input, thread_config(customer_id))


def pending_review(customer_id: str, app=None) -> dict | None:
    app = app or graph
    snapshot = app.get_state(thread_config(customer_id))
    if "human_review" not in snapshot.next:
        return None
    return {
        "customer_message": latest_question(snapshot.values),
        "draft": snapshot.values.get("draft"),
    }


def resume_review(customer_id: str, decision: dict, app=None) -> SupportState:
    app = app or graph
    return app.invoke(Command(resume=decision), thread_config(customer_id))


def ask_reviewer(review: dict) -> dict:
    print("\n=========== HUMAN REVIEW NEEDED ===========")
    print(f"Customer said: {review['customer_message']}")
    print(f"AI draft:      {review['draft']}")
    choice = input("Approve (a), edit (e) or reject (r)? ").strip().lower()
    if choice == "a":
        return {"action": "approve"}
    if choice == "e":
        text = input("Your reply: ").strip()
        print(f"\nCustomer will receive:\n{text}")
        if input("Send it? (y/n) ").strip().lower() == "y":
            return {"action": "edit", "text": text}
        return {"action": "reject"}
    return {"action": "reject"}


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
        review = pending_review(customer)
        if review:
            result = resume_review(customer, ask_reviewer(review))

        print(f"  [path] {' -> '.join(result['trace'])} | needs_human: {result['needs_human']}")
        if result.get("ticket_id"):
            print(f"  [ticket] {result['ticket_id']} logged to {HANDOFF_LOG}")
        print(f"Bot: {result['reply']}")
        for source in result.get("sources", []):
            print(f"  SOURCE: {source['source']} > {source['section']}")