import json
from pathlib import Path

ORDERS_FILE = Path("data/orders.json")

NOT_FOUND = {
    "found": False,
    "message": "No order matches that order number and email address.",
}


def load_orders() -> dict[str, dict]:
    orders = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
    return {order["order_id"]: order for order in orders}


def get_order_status(order_id: str, email: str) -> dict:
    clean_id = order_id.strip().lstrip("#")
    order = load_orders().get(clean_id)

    if order is None or order["email"].lower() != email.strip().lower():
        return NOT_FOUND

    return {
        "found": True,
        "order_id": order["order_id"],
        "status": order["status"],
        "items": order["items"],
        "total_inr": order["total_inr"],
        "shipped_on": order["shipped_on"],
        "expected_delivery": order["expected_delivery"],
        "tracking_url": order["tracking_url"],
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up the current status of one customer order. "
                "Requires both the order number and the email address used to place the order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, for example 4521",
                    },
                    "email": {
                        "type": "string",
                        "description": "The email address the customer used for the order",
                    },
                },
                "required": ["order_id", "email"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_FUNCTIONS = {"get_order_status": get_order_status}