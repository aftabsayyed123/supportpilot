import json

import agent
import tools


def use_test_orders(monkeypatch, tmp_path):
    orders = [
        {"order_id": "1001", "email": "a@example.com", "status": "shipped", "items": ["Mug"],
         "total_inr": 299, "shipped_on": "2026-10-01", "expected_delivery": "2026-10-04",
         "tracking_url": "https://track.example/1001"},
    ]
    test_file = tmp_path / "orders.json"
    test_file.write_text(json.dumps(orders), encoding="utf-8")
    monkeypatch.setattr(tools, "ORDERS_FILE", test_file)


def test_correct_order_and_email_returns_status(monkeypatch, tmp_path):
    use_test_orders(monkeypatch, tmp_path)
    result = tools.get_order_status("#1001", "A@Example.com ")
    assert result["found"] is True
    assert result["status"] == "shipped"
    assert "email" not in result


def test_wrong_email_looks_exactly_like_missing_order(monkeypatch, tmp_path):
    use_test_orders(monkeypatch, tmp_path)
    wrong_email = tools.get_order_status("1001", "someone@example.com")
    missing_order = tools.get_order_status("9999", "a@example.com")
    assert wrong_email == missing_order == tools.NOT_FOUND


def test_run_tool_rejects_unknown_tool():
    result = json.loads(agent.run_tool("delete_all_orders", "{}"))
    assert "Unknown tool" in result["error"]


def test_run_tool_handles_bad_arguments():
    result = json.loads(agent.run_tool("get_order_status", '{"orderId": "1001"}'))
    assert "Invalid arguments" in result["error"]