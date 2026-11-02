import csv

import httpx

import support_graph as sg

ROW = {field: f"test-{field}" for field in sg.LOG_FIELDS}


def read_rows():
    with sg.HANDOFF_LOG.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_ticket_is_sent_with_secret_and_saved_to_csv(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, json=json, headers=headers, timeout=timeout)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(sg, "TICKET_WEBHOOK_URL", "http://n8n.test/webhook/ticket")
    monkeypatch.setattr(sg, "TICKET_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setattr(sg.httpx, "post", fake_post)

    sg.log_handoff(ROW)

    assert sent["json"] == ROW
    assert sent["headers"]["X-Webhook-Secret"] == "test-secret"
    assert sent["timeout"] == sg.WEBHOOK_TIMEOUT_SECONDS
    assert len(read_rows()) == 1


def test_n8n_down_or_erroring_never_crashes_and_csv_keeps_tickets(monkeypatch):
    monkeypatch.setattr(sg, "TICKET_WEBHOOK_URL", "http://n8n.test/webhook/ticket")

    def n8n_down(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(sg.httpx, "post", n8n_down)
    assert sg.send_to_webhook(ROW) is False

    def n8n_error(url, **kwargs):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(sg.httpx, "post", n8n_error)
    assert sg.send_to_webhook(ROW) is False

    sg.log_handoff(ROW)
    assert len(read_rows()) == 1


def test_no_webhook_url_sends_nothing(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("webhook should not be called")

    monkeypatch.setattr(sg.httpx, "post", must_not_be_called)
    assert sg.send_to_webhook(ROW) is False