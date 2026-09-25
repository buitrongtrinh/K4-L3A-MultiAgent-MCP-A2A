from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.a2a import MAX_HOPS_PER_CASE, A2ABus, A2AError
from student_agent.agents import TOOL_SCOPES
from student_agent.analysis import (
    analyze_order,
    analyze_payments,
    analyze_shipment,
    candidate_issues,
    parse_ts,
    select_primary_issue,
)
from student_agent.contracts import Contracts
from student_agent.mcp_gateway import GatewayUnavailableError, ToolCallError
from student_agent.trace import TraceBuffer
from student_agent.workflow import solve_case

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = Contracts(ROOT / "contracts" / "schemas")
ORDER_ID = "order-abc"
OPENED = "2018-03-13T09:00:00-03:00"
POLICY = {
    "currency": "BRL",
    "policy_version": "EC_POLICY_V1",
    "rules": {
        "canceled_order_paid": {
            "case_status": "action_required",
            "recommended_action": "issue_refund",
            "refund_brl": 79.0,
            "responsible_parties": [{"party_id": None, "party_type": "platform"}],
        },
        "late_delivery_logistics": {
            "case_status": "action_required",
            "recommended_action": "refund_freight",
            "refund_brl": 16.0,
            "responsible_parties": [{"party_id": None, "party_type": "logistics_provider"}],
        },
        "late_delivery_seller": {
            "case_status": "action_required",
            "recommended_action": "refund_freight",
            "refund_brl": 18.0,
            "responsible_parties": [{"party_id": "seller-example", "party_type": "seller"}],
        },
        "unsupported_claim": {
            "case_status": "no_action",
            "recommended_action": "document_no_action",
            "refund_brl": 0.0,
            "responsible_parties": [{"party_id": None, "party_type": "customer"}],
        },
    },
}


def order_row(status: str = "delivered", delivered: str | None = "2018-03-09T09:00:00-03:00"):
    return {
        "order_id": ORDER_ID,
        "customer_id": "customer-row-abc",
        "order_status": status,
        "order_purchase_timestamp": "2018-03-01T09:00:00-03:00",
        "order_approved_at": "2018-03-01T10:00:00-03:00",
        "order_delivered_carrier_date": "2018-03-03T09:00:00-03:00",
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": "2018-03-11T09:00:00-03:00",
    }


def item_row(limit: str = "2018-03-04T09:00:00-03:00", freight: str = "10.00"):
    return {
        "order_id": ORDER_ID,
        "order_item_id": "item-abc",
        "product_id": "product-abc",
        "seller_id": "seller-abc",
        "shipping_limit_date": limit,
        "price": "79.00",
        "freight_value": freight,
    }


def capture(at: str, amount: str) -> dict[str, str]:
    return {
        "order_id": ORDER_ID,
        "event_at": at,
        "event_type": "captured",
        "amount_brl": amount,
        "status": "confirmed",
    }


def ship_event(at: str, actor: str) -> dict[str, str]:
    return {
        "order_id": ORDER_ID,
        "event_at": at,
        "event_type": "delivered_late",
        "actor": actor,
        "status": "confirmed",
    }


# --------------------------------------------------------------------------- analysis


def test_order_scope_drops_rows_from_other_timelines_and_duplicates() -> None:
    rows = [item_row(), item_row(), item_row(limit="2018-07-04T09:00:00-03:00", freight="18.00")]
    facts = analyze_order(order_row(), rows, parse_ts(OPENED))
    assert facts.item_ids == ["item-abc"]
    assert facts.duplicate_item_rows == 1
    assert facts.excluded_item_rows == 1
    assert str(facts.order_total) == "89.00"


def test_split_payment_reconciles_but_repeated_overcharge_is_duplicate() -> None:
    order = analyze_order(order_row(), [item_row()], parse_ts(OPENED))
    split = {
        "events": [
            capture("2018-03-01T10:00:00-03:00", "44.50"),
            capture("2018-03-01T11:00:00-03:00", "44.50"),
            capture("2018-06-01T10:00:00-03:00", "52.00"),  # another timeline: ignored
        ]
    }
    facts = analyze_payments(split, None, order, parse_ts(OPENED))
    assert facts.signals == ["split_reconciled"]
    assert facts.excluded_rows == 1

    duplicate = {
        "events": [
            capture("2018-03-01T10:00:00-03:00", "64.00"),
            capture("2018-03-01T11:00:00-03:00", "64.00"),
        ]
    }
    facts = analyze_payments(duplicate, None, order, parse_ts(OPENED))
    assert "duplicate_capture" in facts.signals
    assert str(facts.duplicate_amount) == "64.00"


def test_refund_events_outside_complaint_window_are_ignored() -> None:
    order = analyze_order(order_row(), [item_row()], parse_ts(OPENED))
    refunds = {
        "events": [
            {"event_at": "2018-03-12T09:00:00-03:00", "status": "failed", "amount_brl": "52.00"},
            {"event_at": "2018-09-12T09:00:00-03:00", "status": "pending", "amount_brl": "89.00"},
        ]
    }
    timeline = {"events": [capture("2018-03-01T10:00:00-03:00", "52.00")]}
    facts = analyze_payments(timeline, refunds, order, parse_ts(OPENED))
    assert facts.signals[0] == "refund_failed"
    assert "refund_pending" not in facts.signals


def test_late_delivery_is_attributed_by_handoff_versus_shipping_limit() -> None:
    late = order_row(delivered="2018-03-12T09:00:00-03:00")
    on_time_handoff = analyze_order(late, [item_row()], parse_ts(OPENED))
    summary = {
        "order_status": "delivered",
        "delivered_customer_at": late["order_delivered_customer_date"],
        "events": [ship_event("2018-03-12T09:00:00-03:00", "logistics_provider")],
    }
    facts = analyze_shipment(summary, on_time_handoff, parse_ts(OPENED))
    assert facts.verdict == "logistics_delay"
    assert facts.consistent_events == 1

    missed_limit = analyze_order(
        late, [item_row(limit="2018-03-02T09:00:00-03:00")], parse_ts(OPENED)
    )
    facts = analyze_shipment(summary, missed_limit, parse_ts(OPENED))
    assert facts.verdict == "seller_delay"
    assert facts.late_seller_ids == ["seller-abc"]
    assert facts.conflicts  # the event blames logistics, timestamps blame the seller


def test_late_event_contradicting_on_time_delivery_is_a_conflict() -> None:
    order = analyze_order(order_row(), [item_row()], parse_ts(OPENED))
    summary = {
        "order_status": "delivered",
        "delivered_customer_at": "2018-03-09T09:00:00-03:00",
        "events": [ship_event("2018-03-07T09:00:00-03:00", "logistics_provider")],
    }
    facts = analyze_shipment(summary, order, parse_ts(OPENED))
    assert facts.verdict == "on_time"
    assert facts.conflicts[0]["resolution_code"] == "ORDER_TIMESTAMPS_AUTHORITATIVE"


def test_primary_issue_trusts_evidence_over_the_claim() -> None:
    assert select_primary_issue("refund_failed", ["refund_failed"], has_core_evidence=True) == (
        "refund_failed"
    )
    assert (
        select_primary_issue("duplicate_charge", ["late_delivery_seller"], has_core_evidence=True)
        == "late_delivery_seller"
    )
    assert (
        select_primary_issue("duplicate_charge", ["valid_split_payment"], has_core_evidence=True)
        == "valid_split_payment"
    )
    assert select_primary_issue("late_delivery_seller", [], has_core_evidence=True) == (
        "unsupported_claim"
    )
    assert select_primary_issue("refund_failed", [], has_core_evidence=False) == (
        "insufficient_evidence"
    )


def test_candidates_require_payment_for_canceled_order() -> None:
    order = analyze_order(order_row(status="canceled", delivered=None), [item_row()], None)
    unpaid = analyze_payments({"events": []}, None, order, None)
    shipment = analyze_shipment(None, order, None)
    assert candidate_issues(order, unpaid, shipment) == []


# --------------------------------------------------------------------------- end-to-end


class FakeGateway:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.issued: set[str] = set()

    async def list_tools(self) -> list[str]:
        return sorted({*self.responses, "get_product_context", "get_customer_history"})

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, {"case_id": case_id, **arguments}))
        data = self.responses.get(tool_name)
        if data is None:
            raise ToolCallError(tool_name, "Error executing tool")
        domain = {
            "get_order": "order",
            "get_order_items": "item",
            "get_payment_timeline": "payment",
            "get_refund_timeline": "refund",
            "get_shipment_summary": "shipment",
            "get_sellers": "seller",
            "get_policy": "policy",
        }[tool_name]
        digest = hashlib.sha256(f"{case_id}:{tool_name}:{len(self.calls)}".encode()).hexdigest()
        ref = f"ev_{digest[:32]}"
        self.issued.add(ref)
        envelope = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": ref,
            "result_hash": f"sha256:{digest}",
            "domain": domain,
            "data": data,
            "warnings": [],
        }
        CONTRACTS.validate_evidence(envelope)
        return envelope


def make_case(topic: str) -> dict[str, Any]:
    return {
        "case_id": "TEST_CASE_001",
        "opened_at": OPENED,
        "customer_request": {
            "language": "vi",
            "message": "test",
            "claimed_order_id": ORDER_ID,
            "claims": [
                {"claim_id": "claim-a", "topic": topic},
                {"claim_id": "claim-b", "topic": "requested_full_refund"},
            ],
        },
        "policy_version": "EC_POLICY_V1",
    }


def run(case: dict[str, Any], gateway: FakeGateway) -> tuple[dict[str, Any], list[dict]]:
    trace = TraceBuffer(CONTRACTS)
    trace.emit(case_id=case["case_id"], event_type="case_received", actor="coordinator")
    output = asyncio.run(solve_case(case, gateway, trace))
    CONTRACTS.validate_output(output, "output")
    return output, trace.events


def base_responses(**overrides: Any) -> dict[str, Any]:
    responses = {
        "get_order": order_row(),
        "get_order_items": [item_row()],
        "get_payment_timeline": {"events": [capture("2018-03-01T10:00:00-03:00", "89.00")]},
        "get_shipment_summary": {
            "order_status": "delivered",
            "delivered_customer_at": "2018-03-09T09:00:00-03:00",
            "events": [],
        },
        "get_sellers": [{"seller_id": "seller-abc"}],
        "get_policy": POLICY,
    }
    responses.update(overrides)
    return responses


def test_canceled_paid_order_is_refunded_with_traced_evidence() -> None:
    gateway = FakeGateway(
        base_responses(
            get_order=order_row(status="canceled", delivered=None),
            get_payment_timeline={"events": [capture("2018-03-01T10:00:00-03:00", "79.00")]},
            get_shipment_summary={"order_status": "canceled", "events": []},
        )
    )
    output, events = run(make_case("canceled_order_paid"), gateway)

    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["financial_resolution"]["recommended_refund_brl"] == 79.0
    assert output["resolution_actions"] == ["issue_refund"]
    assert [c["verdict"] for c in output["claim_assessments"]] == ["supported", "supported"]
    assert set(output["evidence_refs"]) <= gateway.issued
    consumed = {
        ref
        for e in events
        if e["event_type"] == "tool_result_consumed"
        for ref in e.get("evidence_refs", [])
    }
    assert set(output["evidence_refs"]) <= consumed
    types = {e["event_type"] for e in events}
    assert {"task_assigned", "handoff", "policy_decided", "verification_completed"} <= types
    verification = [e for e in events if e["event_type"] == "verification_completed"]
    assert verification[-1]["decision_code"] == "VERIFICATION_PASSED"
    assert all(args["case_id"] == "TEST_CASE_001" for _, args in gateway.calls)
    assert len({e["actor"] for e in events}) >= 5


def test_seller_delay_blames_this_orders_seller_not_policy_example() -> None:
    gateway = FakeGateway(
        base_responses(
            get_order=order_row(delivered="2018-03-12T09:00:00-03:00"),
            get_order_items=[item_row(limit="2018-03-02T09:00:00-03:00", freight="18.00")],
            get_payment_timeline={"events": [capture("2018-03-01T10:00:00-03:00", "18.00")]},
            get_shipment_summary={
                "order_status": "delivered",
                "delivered_customer_at": "2018-03-12T09:00:00-03:00",
                "events": [ship_event("2018-03-12T09:00:00-03:00", "seller")],
            },
        )
    )
    output, _ = run(make_case("late_delivery_seller"), gateway)
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["root_cause_analysis"]["responsible_parties"] == [
        {"party_type": "seller", "party_id": "seller-abc"}
    ]
    assert output["affected_entities"]["seller_ids"] == ["seller-abc"]
    assert output["financial_resolution"]["refund_lines"][0]["entity_id"] == "item-abc"
    assert [c["verdict"] for c in output["claim_assessments"]] == [
        "supported",
        "partially_supported",
    ]


def test_claim_without_evidence_is_unsupported_and_no_action() -> None:
    gateway = FakeGateway(base_responses())
    output, _ = run(make_case("late_delivery_logistics"), gateway)
    assert output["assessment"]["primary_issue"] == "unsupported_claim"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert output["financial_resolution"]["refund_lines"] == []
    assert output["claim_assessments"][0]["verdict"] == "unsupported"


def test_missing_order_yields_insufficient_evidence() -> None:
    responses = base_responses()
    del responses["get_order"]
    output, _ = run(make_case("refund_failed"), FakeGateway(responses))
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["assessment"]["confidence"] <= 0.5
    assert output["financial_resolution"]["recommended_refund_brl"] == 0


def test_agents_have_least_privilege_tool_scopes() -> None:
    assert TOOL_SCOPES["coordinator"] == frozenset()
    assert TOOL_SCOPES["verifier"] == frozenset()
    scoped = [tool for tools in TOOL_SCOPES.values() for tool in tools]
    assert len(scoped) == len(set(scoped)), "each tool has exactly one owning agent"


def test_a2a_bus_rejects_foreign_case_and_runaway_loops() -> None:
    trace = TraceBuffer(CONTRACTS)
    bus = A2ABus("TEST_CASE_001", trace)

    async def echo(message):
        return message

    bus.register("echo", echo)

    async def loop_forever() -> None:
        for _ in range(MAX_HOPS_PER_CASE):
            await bus.request("coordinator", "echo", "ping")

    with pytest.raises(A2AError, match="hop budget"):
        asyncio.run(loop_forever())
    assert all(json.dumps(e) for e in trace.events)


def test_policy_outage_is_not_turned_into_an_answer() -> None:
    responses = base_responses()
    responses["get_policy"] = None  # discovered, but the gateway rejects the call
    with pytest.raises(GatewayUnavailableError):
        run(make_case("refund_failed"), FakeGateway(responses))
