import pytest

import support_graph as sg


@pytest.fixture(autouse=True)
def no_real_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(sg, "TICKET_WEBHOOK_URL", None)
    monkeypatch.setattr(sg, "HANDOFF_LOG", tmp_path / "handoffs.csv")