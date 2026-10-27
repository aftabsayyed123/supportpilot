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


def plan(skills, escalate=False):
    return lambda messages: sg.Plan(skills=skills, escalate=escalate)


def test_order_only(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Order 4521 has shipped.")

    result = sg.run_turn("Where is order 4521?", "c1", sg.build_graph())
    assert result["trace"] == ["planner", "order_lookup", "compose"]
    assert result["reply"] == "Order 4521 has shipped."
    assert result["sources"] == []


def test_order_and_policy_run_both_and_combine(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order", "policy"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Order 4521 was delivered.")
    monkeypatch.setattr(rag, "answer_question", lambda question: FAKE_POLICY)
    monkeypatch.setattr(sg, "combine_answers", lambda q, order, policy: f"{order} {policy}")

    result = sg.run_turn("Can I return order 4521?", "c1", sg.build_graph())
    assert result["trace"][0] == "planner"
    assert set(result["trace"][1:3]) == {"order_lookup", "policy_lookup"}
    assert result["trace"][-1] == "compose"
    assert result["reply"] == "Order 4521 was delivered. Electronics have a 7-day return window."
    assert result["sources"][0]["source"] == "returns.md"


def test_damaged_item_escalates_even_if_planner_misses_it(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order", "policy"], escalate=False))

    result = sg.run_turn("My order arrived broken, can I return it?", "c1", sg.build_graph())
    assert result["trace"] == ["planner", "handoff"]
    assert result["needs_human"] is True


def test_bad_planner_output_escalates():
    assert sg.parse_plan("not json").escalate is True
    assert sg.parse_plan('{"skills": ["refund"]}').escalate is True
    assert sg.parse_plan('{"skills": ["policy"], "escalate": false}').skills == ["policy"]


def test_customers_have_separate_memory(monkeypatch):
    monkeypatch.setattr(sg, "plan_turn", plan(["order"]))
    monkeypatch.setattr(agent, "chat", lambda messages: "Please share your email.")
    app = sg.build_graph()

    sg.run_turn("Where is order 4521?", "ravi", app)
    sg.run_turn("It's ravi@example.com", "ravi", app)
    sg.run_turn("Where is order 4522?", "priya", app)

    ravi = app.get_state({"configurable": {"thread_id": "ravi"}}).values["messages"]
    priya = app.get_state({"configurable": {"thread_id": "priya"}}).values["messages"]
    assert len(ravi) == 4
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

    assert result["trace"] == ["planner", "order_lookup", "compose"]
    assert result["reply"] == "Order 4521 has shipped."
    assert result["sources"] == []