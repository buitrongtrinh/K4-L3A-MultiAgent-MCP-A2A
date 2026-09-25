"""Pure, deterministic analysis used by the specialist agents.

MCP rows for an order can include records that do not belong to the order's own
lifecycle (rows whose timestamps sit weeks away from the purchase, or exact duplicate
rows). Every specialist therefore scopes rows to the order timeline before reasoning:

* payment events: within [approved_at - 1h, approved_at + 24h]
* refund events: within [purchased_at, opened_at]
* item shipping limits: within [purchased_at, estimated_delivery]
* shipment events: within [purchased_at, max(opened_at, delivered_at) + 1 day]

Out-of-scope rows are counted (and traced) but never used as evidence for a claim.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

CENT = Decimal("0.01")

PRIMARY_ISSUES = (
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "refund_pending",
    "refund_failed",
    "unsupported_claim",
    "insufficient_evidence",
)
# Findings that do not require any corrective action on their own.
NON_ACTIONABLE = {"valid_split_payment", "unsupported_claim", "insufficient_evidence"}
PAYMENT_TOPICS = {"duplicate_charge", "payment_mismatch", "valid_split_payment"}


def parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def money(value: Any) -> Decimal:
    try:
        return Decimal(str(value)).quantize(CENT)
    except (ArithmeticError, ValueError):
        return Decimal("0.00")


def _unique(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen.setdefault(json.dumps(row, sort_keys=True), row)
    return list(seen.values()), len(rows) - len(seen)


def _within(moment: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if moment is None:
        return False
    if start is not None and moment < start:
        return False
    return not (end is not None and moment > end)


# --------------------------------------------------------------------------- order


@dataclass
class OrderFacts:
    order_id: str
    status: str
    purchased_at: datetime | None
    approved_at: datetime | None
    carrier_at: datetime | None
    delivered_at: datetime | None
    estimated_at: datetime | None
    items: list[dict[str, Any]]
    excluded_item_rows: int = 0
    duplicate_item_rows: int = 0

    @property
    def item_ids(self) -> list[str]:
        return list(dict.fromkeys(item["order_item_id"] for item in self.items))

    @property
    def seller_ids(self) -> list[str]:
        return list(
            dict.fromkeys(item["seller_id"] for item in self.items if item.get("seller_id"))
        )

    @property
    def order_total(self) -> Decimal:
        return sum(
            (money(i.get("price")) + money(i.get("freight_value")) for i in self.items),
            Decimal("0.00"),
        )

    @property
    def freight_total(self) -> Decimal:
        return sum((money(i.get("freight_value")) for i in self.items), Decimal("0.00"))


def analyze_order(
    order: dict[str, Any], items: list[dict[str, Any]], opened_at: datetime | None
) -> OrderFacts:
    purchased_at = parse_ts(order.get("order_purchase_timestamp"))
    estimated_at = parse_ts(order.get("order_estimated_delivery_date"))
    unique_items, duplicates = _unique(
        [i for i in items if i.get("order_id") in (None, order.get("order_id"))]
    )
    scoped = [
        i
        for i in unique_items
        if _within(parse_ts(i.get("shipping_limit_date")), purchased_at, estimated_at or opened_at)
    ]
    if not scoped and unique_items:
        # Fall back to the row whose limit is closest to the purchase, never to all rows.
        def distance(row: dict[str, Any]) -> float:
            limit = parse_ts(row.get("shipping_limit_date"))
            if limit is None or purchased_at is None:
                return float("inf")
            return abs((limit - purchased_at).total_seconds())

        scoped = [min(unique_items, key=distance)]
    return OrderFacts(
        order_id=order.get("order_id", ""),
        status=str(order.get("order_status") or "unknown"),
        purchased_at=purchased_at,
        approved_at=parse_ts(order.get("order_approved_at")),
        carrier_at=parse_ts(order.get("order_delivered_carrier_date")),
        delivered_at=parse_ts(order.get("order_delivered_customer_date")),
        estimated_at=estimated_at,
        items=scoped,
        excluded_item_rows=len(unique_items) - len(scoped),
        duplicate_item_rows=duplicates,
    )


# --------------------------------------------------------------------------- payment


@dataclass
class PaymentFacts:
    captures: list[Decimal] = field(default_factory=list)
    mismatch_total: Decimal = Decimal("0.00")
    failed_refund_total: Decimal = Decimal("0.00")
    pending_refund_total: Decimal = Decimal("0.00")
    duplicate_amount: Decimal = Decimal("0.00")
    signals: list[str] = field(default_factory=list)
    excluded_rows: int = 0
    duplicate_rows: int = 0
    refund_timeline_found: bool = False

    @property
    def captured_total(self) -> Decimal:
        return sum(self.captures, Decimal("0.00"))

    @property
    def verdict(self) -> str:
        return self.signals[0] if self.signals else "insufficient_evidence"


def analyze_payments(
    timeline: dict[str, Any] | None,
    refunds: dict[str, Any] | None,
    order: OrderFacts,
    opened_at: datetime | None,
) -> PaymentFacts:
    facts = PaymentFacts(refund_timeline_found=refunds is not None)
    anchor = order.approved_at or order.purchased_at
    pay_start = anchor - timedelta(hours=1) if anchor else None
    pay_end = anchor + timedelta(hours=24) if anchor else None

    events, duplicates = _unique(list((timeline or {}).get("events") or []))
    facts.duplicate_rows += duplicates
    for event in events:
        if not _within(parse_ts(event.get("event_at")), pay_start, pay_end):
            facts.excluded_rows += 1
            continue
        amount = money(event.get("amount_brl"))
        kind = event.get("event_type")
        if kind == "captured" and event.get("status") == "confirmed":
            facts.captures.append(amount)
        elif kind == "reconciliation_mismatch" and event.get("status") != "resolved":
            facts.mismatch_total += amount

    refund_events, duplicates = _unique(list((refunds or {}).get("events") or []))
    facts.duplicate_rows += duplicates
    for event in refund_events:
        if not _within(parse_ts(event.get("event_at")), order.purchased_at, opened_at):
            facts.excluded_rows += 1
            continue
        amount = money(event.get("amount_brl"))
        if event.get("status") == "failed":
            facts.failed_refund_total += amount
        elif event.get("status") == "pending":
            facts.pending_refund_total += amount

    counts = Counter(facts.captures)
    facts.duplicate_amount = sum(
        (amount * (count - 1) for amount, count in counts.items() if count > 1),
        Decimal("0.00"),
    )
    total = order.order_total
    reconciled = bool(facts.captures) and abs(facts.captured_total - total) < CENT
    if facts.failed_refund_total > 0:
        facts.signals.append("refund_failed")
    if facts.pending_refund_total > 0:
        facts.signals.append("refund_pending")
    if facts.mismatch_total > 0:
        facts.signals.append("capture_mismatch")
    if len(facts.captures) >= 2 and facts.duplicate_amount > 0 and not reconciled:
        facts.signals.append("duplicate_capture")
    if len(facts.captures) >= 2 and reconciled:
        facts.signals.append("split_reconciled")
    elif reconciled:
        facts.signals.append("reconciled")
    if not facts.captures:
        facts.signals.append("no_capture")
    elif not reconciled and not facts.signals:
        facts.signals.append("captured")
    return facts


# --------------------------------------------------------------------------- shipment

ORDER_IS_SOURCE = "get_order"


@dataclass
class ShipmentFacts:
    verdict: str
    late_seller_ids: list[str] = field(default_factory=list)
    consistent_events: int = 0
    excluded_events: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)


def analyze_shipment(
    summary: dict[str, Any] | None, order: OrderFacts, opened_at: datetime | None
) -> ShipmentFacts:
    if order.status in {"canceled", "unavailable"} or order.delivered_at is None:
        verdict = "not_delivered"
    elif order.estimated_at is None:
        verdict = "insufficient_evidence"
    elif order.delivered_at > order.estimated_at:
        verdict = "late"
    else:
        verdict = "on_time"

    late_sellers: list[str] = []
    if verdict == "late":
        for item in order.items:
            limit = parse_ts(item.get("shipping_limit_date"))
            if order.carrier_at and limit and order.carrier_at > limit:
                late_sellers.append(item["seller_id"])
        verdict = "seller_delay" if late_sellers else "logistics_delay"
    facts = ShipmentFacts(verdict=verdict, late_seller_ids=list(dict.fromkeys(late_sellers)))
    if summary is None:
        return facts

    if summary.get("order_status") not in (None, order.status):
        facts.conflicts.append(
            _conflict("order_status", "get_shipment_summary", "ORDER_RECORD_AUTHORITATIVE")
        )
    if parse_ts(summary.get("delivered_customer_at")) != order.delivered_at:
        facts.conflicts.append(
            _conflict("delivered_customer_at", "get_shipment_summary", "ORDER_RECORD_AUTHORITATIVE")
        )

    horizon = max(filter(None, [opened_at, order.delivered_at]), default=None)
    end = horizon + timedelta(days=1) if horizon else None
    expected_actor = {"seller_delay": "seller", "logistics_delay": "logistics_provider"}
    events, _ = _unique(list(summary.get("events") or []))
    for event in events:
        if not _within(parse_ts(event.get("event_at")), order.purchased_at, end):
            facts.excluded_events += 1
            continue
        if event.get("event_type") != "delivered_late":
            continue
        if expected_actor.get(verdict) == event.get("actor"):
            facts.consistent_events += 1
        elif verdict in expected_actor:
            facts.conflicts.append(
                _conflict(
                    "delay_responsibility", "get_shipment_summary", "HANDOFF_VS_SHIPPING_LIMIT"
                )
            )
        else:
            facts.conflicts.append(
                _conflict(
                    "delivery_timeliness", "get_shipment_summary", "ORDER_TIMESTAMPS_AUTHORITATIVE"
                )
            )
    return facts


def _conflict(field_name: str, other_source: str, resolution: str) -> dict[str, Any]:
    return {
        "field": field_name,
        "sources": [ORDER_IS_SOURCE, other_source],
        "selected_source": ORDER_IS_SOURCE,
        "resolution_code": resolution,
    }


# --------------------------------------------------------------------------- adjudication


def candidate_issues(
    order: OrderFacts | None, payment: PaymentFacts | None, shipment: ShipmentFacts | None
) -> list[str]:
    """All issues the evidence supports, most specific first."""
    if order is None:
        return []
    found: list[str] = []
    paid = payment is not None and payment.captured_total > 0
    if order.status == "canceled" and paid:
        found.append("canceled_order_paid")
    if order.status == "unavailable" and paid:
        found.append("unavailable_order_paid")
    if payment is not None:
        mapping = {
            "refund_failed": "refund_failed",
            "refund_pending": "refund_pending",
            "capture_mismatch": "payment_mismatch",
        }
        found += [mapping[s] for s in payment.signals if s in mapping]
    if shipment is not None and shipment.verdict == "seller_delay":
        found.append("late_delivery_seller")
    if shipment is not None and shipment.verdict == "logistics_delay":
        found.append("late_delivery_logistics")
    if payment is not None and "duplicate_capture" in payment.signals:
        found.append("duplicate_charge")
    if payment is not None and "split_reconciled" in payment.signals:
        found.append("valid_split_payment")
    return found


def select_primary_issue(
    claimed_topic: str | None,
    candidates: list[str],
    *,
    has_core_evidence: bool,
) -> str:
    """Pick the issue that answers the complaint, trusting evidence over the claim."""
    if not has_core_evidence:
        return "insufficient_evidence"
    if claimed_topic in candidates:
        return claimed_topic  # the claim is confirmed by evidence
    actionable = [issue for issue in candidates if issue not in NON_ACTIONABLE]
    if actionable:
        return actionable[0]  # evidence shows a different, real problem
    if "valid_split_payment" in candidates and claimed_topic in PAYMENT_TOPICS:
        return "valid_split_payment"
    return "unsupported_claim"
