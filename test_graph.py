import csv

import pytest

import agent
import rag
import support_graph as sg
from rag import Answer, Source

FAKE_POLICY = Answer(
    answer="Electronics have a 7-day return window.",
    status="answered",
    needs_human=False,
    reason="answered",
    sources=[Source(source="returns.md", section="Return window", score=0.6)],
)
FAKE_DRAFT = "We're so sorry. A support specialist will review this."


@pytest.fixture(autouse=True)
def temp_log(monkeypatch, tmp_path):
    log_file = tmp_path / "handoffs.csv"
    monkeypatch.setattr(sg, "HANDOFF_LOG", log_file)
    return log_file


def plan(skills, escalate=False):
    return lambda messages: sg.Plan(skills=skills, escalate=escalate)


def read_log(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def start_escalation(monkeypatch, text, customer="ravi"):
    monkeypatch.setattr(sg, "plan_turn", plan([], escalate=True))
    monkeypatch.setattr(sg, "draft_text", lambda messages: FAKE_DRAFT)
    app = sg.build_graph()
    first = sg.run_turn(text, customer, app)
    return app, first


# ---------- routing and memory ----------

def test_order_only(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Order 4521 has shipped.")

    result = sg.run_turn("Where is order 4521?", "c1", sg.build_graph())
    assert result["trace"] == ["planner", "order_lookup", "compose"]
    assert result["reply"] == "Order 4521 has shipped."


def test_order_and_policy_run_both_and_combine(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order", "policy"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Order 4521 was delivered.")
    monkeypatch.setattr(rag, "answer_question", lambda question: FAKE_POLICY)
    monkeypatch.setattr(sg, "combine_answers", lambda q, order, policy: f"{order} {policy}")

    result = sg.run_turn("Can I return order 4521?", "c1", sg.build_graph())
    assert set(result["trace"][1:3]) == {"order_lookup", "policy_lookup"}
    assert result["trace"][-1] == "compose"
    assert result["sources"][0]["source"] == "returns.md"


def test_bad_planner_output_escalates():
    assert sg.parse_plan("not json").escalate is True
    assert sg.parse_plan('{"skills": ["refund"]}').escalate is True


def test_customers_have_separate_memory(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Please share your email.")
    app = sg.build_graph()

    sg.run_turn("Where is order 4521?", "ravi", app)
    sg.run_turn("It's ravi@example.com", "ravi", app)
    sg.run_turn("Where is order 4522?", "priya", app)

    priya = app.get_state(sg.thread_config("priya")).values["messages"]
    assert len(priya) == 2
    assert all("4521" not in message["content"] for message in priya)


def test_previous_turn_results_are_cleared(monkeypatch):
    app = sg.build_graph()
    monkeypatch.setattr(sg, "plan_turn", plan(["policy"]))
    monkeypatch.setattr(rag, "answer_question", lambda question: FAKE_POLICY)
    sg.run_turn("What is the return window?", "c1", app)

    monkeypatch.setattr(sg, "plan_turn", plan(["order"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Order 4521 has shipped.")
    result = sg.run_turn("And order 4521?", "c1", app)
    assert result["sources"] == []


# ---------- human in the loop ----------

def test_escalation_pauses_for_human_review(monkeypatch):
    app, first = start_escalation(monkeypatch, "My order arrived broken")

    assert first["trace"] == ["planner", "draft_reply"]
    review = sg.pending_review("ravi", app)
    assert review["draft"] == FAKE_DRAFT
    assert review["customer_message"] == "My order arrived broken"


def test_approved_draft_is_sent_and_logged(monkeypatch, temp_log):
    app, _ = start_escalation(monkeypatch, "My order 4521 arrived broken. ravi@example.com")

    result = sg.resume_review("ravi", {"action": "approve"}, app)
    assert result["trace"] == ["planner", "draft_reply", "human_review", "create_ticket"]
    assert result["reply"].startswith(FAKE_DRAFT)
    assert result["ticket_id"] in result["reply"]
    assert sg.pending_review("ravi", app) is None

    row = read_log(temp_log)[0]
    assert row["email"] == "ravi@example.com"
    assert row["review_action"] == "approve"
    assert row["customer_id"] == "ravi"


def test_edited_reply_replaces_draft(monkeypatch, temp_log):
    app, _ = start_escalation(monkeypatch, "Wrong item received. ravi@example.com")

    result = sg.resume_review("ravi", {"action": "edit", "text": "Sorry! A replacement is on its way."}, app)
    assert result["reply"].startswith("Sorry! A replacement is on its way.")
    assert read_log(temp_log)[0]["review_action"] == "edit"


def test_invalid_or_empty_decisions_send_safe_message(monkeypatch):
    app, _ = start_escalation(monkeypatch, "It arrived broken. ravi@example.com")
    result = sg.resume_review("ravi", {"action": "publish"}, app)
    assert result["reply"].startswith(sg.HANDOFF_REPLY)

    assert sg.parse_decision({"action": "edit", "text": "   "}).action == "reject"
    assert sg.parse_decision(None).action == "reject"


def test_missing_email_asks_for_it(monkeypatch, temp_log):
    monkeypatch.setattr(sg, "plan_turn", plan([]))

    result = sg.run_turn("Can you write me a poem?", "c1", sg.build_graph())
    assert result["trace"] == ["planner", "handoff", "create_ticket"]
    assert sg.ASK_FOR_EMAIL in result["reply"]
    assert read_log(temp_log)[0]["email"] == ""


def test_find_email_reads_only_customer_messages():
    assert sg.find_email([{"role": "user", "content": "I'm Ravi@Example.com"}]) == "ravi@example.com"
    assert sg.find_email([{"role": "assistant", "content": "Write to help@novacart.example"}]) is None
    assert sg.find_email([]) is None


def test_join_sentences_adds_missing_full_stop():
    assert sg.join_sentences("Sorry", "Next.") == "Sorry. Next."
    assert sg.join_sentences("Sorry!", "Next.") == "Sorry! Next."
    assert sg.join_sentences("Sorry.  ", "Next.") == "Sorry. Next."