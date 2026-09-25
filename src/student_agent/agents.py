"""Specialist agents. Each one owns a narrow tool scope and answers A2A task messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from .a2a import A2ABus, A2AMessage, TraceSink, reply_to
from .analysis import (
    CENT,
    OrderFacts,
    PaymentFacts,
    analyze_order,
    analyze_payments,
    analyze_shipment,
)
from .contracts import ContractError, Contracts
from .evidence import Evidence, EvidenceRegistry
from .mcp_gateway import EvidenceGateway, GatewayUnavailableError, ToolCallError

# Which MCP tools each actor may call (least privilege). Coordinator/verifier call none.
TOOL_SCOPES: dict[str, frozenset[str]] = {
    "order-agent": frozenset(
        {"get_order", "get_order_items", "get_sellers", "get_product_context"}
    ),
    "payment-agent": frozenset(
        {"get_payment_timeline", "get_refund_timeline", "get_order_payments"}
    ),
    "shipment-agent": frozenset({"get_shipment_summary"}),
    "policy-agent": frozenset({"get_policy"}),
    "coordinator": frozenset(),
    "verifier": frozenset(),
}

# Evidence that supports each conclusion. Only these refs are cited in the output.
FOCUSED_EVIDENCE_TOOLS: dict[str, tuple[str, ...]] = {
    "canceled_order_paid": ("get_order", "get_payment_timeline", "get_policy"),
    # Seller records back the liable seller (v5 showed dropping them costs evidence score).
    "unavailable_order_paid": (
        "get_order",
        "get_order_items",
        "get_sellers",
        "get_payment_timeline",
        "get_policy",
    ),
    "late_delivery_seller": (
        "get_order",
        "get_order_items",
        "get_shipment_summary",
        "get_sellers",
        "get_policy",
    ),
    "late_delivery_logistics": (
        "get_order",
        "get_order_items",
        "get_shipment_summary",
        "get_payment_timeline",
        "get_policy",
    ),
    "valid_split_payment": ("get_order", "get_order_items", "get_payment_timeline", "get_policy"),
    "payment_mismatch": ("get_order", "get_payment_timeline", "get_policy"),
    "duplicate_charge": ("get_order", "get_order_items", "get_payment_timeline", "get_policy"),
    "refund_pending": ("get_order", "get_payment_timeline", "get_refund_timeline", "get_policy"),
    "refund_failed": ("get_order", "get_payment_timeline", "get_refund_timeline", "get_policy"),
    "unsupported_claim": (
        "get_order",
        "get_order_items",
        "get_payment_timeline",
        "get_shipment_summary",
        "get_policy",
    ),
    "insufficient_evidence": ("get_order", "get_policy"),
}
# Every order-scoped tool the workflow consumes (product/customer context is never cited).
ALL_CONSUMED_TOOLS: tuple[str, ...] = (
    "get_order",
    "get_order_items",
    "get_payment_timeline",
    "get_refund_timeline",
    "get_shipment_summary",
    "get_sellers",
    "get_policy",
)
# Payment rows and product context: fetched and cited only in "extended" mode.
EXTENDED_TOOLS: tuple[str, ...] = ("get_order_payments", "get_product_context")
# DAY09_EVIDENCE_MODE (diagnostics, see todo.md submission log):
#   focused  - per-issue evidence (default)
#   all      - every consumed ref from the default tool set
#   extended - focused + payment rows + product context
EVIDENCE_MODE = os.getenv("DAY09_EVIDENCE_MODE", "focused")
if EVIDENCE_MODE == "all":
    EVIDENCE_TOOLS = {issue: ALL_CONSUMED_TOOLS for issue in FOCUSED_EVIDENCE_TOOLS}
elif EVIDENCE_MODE == "extended":
    EVIDENCE_TOOLS = {i: t + EXTENDED_TOOLS for i, t in FOCUSED_EVIDENCE_TOOLS.items()}
else:
    EVIDENCE_TOOLS = FOCUSED_EVIDENCE_TOOLS
# Domains that must be covered for each conclusion (checked by the verifier).
REQUIRED_DOMAINS: dict[str, frozenset[str]] = {
    "canceled_order_paid": frozenset({"order", "payment", "policy"}),
    "unavailable_order_paid": frozenset({"order", "payment", "item", "policy"}),
    "late_delivery_seller": frozenset({"order", "item", "shipment", "policy"}),
    "late_delivery_logistics": frozenset({"order", "shipment", "policy"}),
    "valid_split_payment": frozenset({"payment", "item", "policy"}),
    "payment_mismatch": frozenset({"payment", "policy"}),
    "duplicate_charge": frozenset({"payment", "item", "policy"}),
    "refund_pending": frozenset({"refund", "policy"}),
    "refund_failed": frozenset({"refund", "policy"}),
    "unsupported_claim": frozenset({"order", "policy"}),
    "insufficient_evidence": frozenset(),
}
FALLBACK_ACTION = "escalate_manual_review"
ALWAYS_AVAILABLE_TOOLS = frozenset({"get_policy"})


@dataclass
class CaseContext:
    case: dict[str, Any]
    gateway: EvidenceGateway
    trace: TraceSink
    registry: EvidenceRegistry
    discovered_tools: set[str]
    opened_at: datetime | None
    consumed_refs: set[str] = field(default_factory=set)

    @property
    def case_id(self) -> str:
        return self.case["case_id"]


class Agent:
    name = "agent"

    def __init__(self, ctx: CaseContext, bus: A2ABus) -> None:
        self.ctx = ctx
        self.bus = bus
        bus.register(self.name, self.handle)

    async def handle(self, message: A2AMessage) -> A2AMessage:
        raise NotImplementedError

    async def fetch(self, tool: str, **arguments: str) -> Evidence | None:
        if tool not in TOOL_SCOPES[self.name]:
            raise PermissionError(f"{self.name} is not allowed to call {tool}")
        if tool not in self.ctx.discovered_tools:
            self._trace_no_result(tool, "TOOL_NOT_DISCOVERED")
            return None
        try:
            envelope = await self.ctx.gateway.call(tool, case_id=self.ctx.case_id, **arguments)
        except ToolCallError as exc:
            if tool in ALWAYS_AVAILABLE_TOOLS:
                # The policy document exists for every case: failing here means the gateway
                # is down or rejecting us, not "no records". Never turn that into an answer.
                raise GatewayUnavailableError(str(exc)) from exc
            self._trace_no_result(tool, "TOOL_NO_RECORDS")
            return None
        return self.ctx.registry.register(tool, envelope)

    def consumed(self, evidence: Evidence, decision_code: str, **attributes: Any) -> None:
        self.ctx.consumed_refs.add(evidence.ref)
        self.ctx.trace.emit(
            case_id=self.ctx.case_id,
            event_type="tool_result_consumed",
            actor=self.name,
            tool_name=evidence.tool,
            decision_code=decision_code,
            evidence_refs=[evidence.ref],
            attributes={"domain": evidence.domain, **attributes},
        )

    def _trace_no_result(self, tool: str, code: str) -> None:
        self.ctx.trace.emit(
            case_id=self.ctx.case_id,
            event_type="tool_result_consumed",
            actor=self.name,
            tool_name=tool,
            decision_code=code,
        )


def _rows(evidence: Evidence | None) -> list[dict[str, Any]]:
    data = evidence.data if evidence else None
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _obj(evidence: Evidence | None) -> dict[str, Any] | None:
    return evidence.data if evidence and isinstance(evidence.data, dict) else None


# --------------------------------------------------------------------------- order/item


class OrderAgent(Agent):
    name = "order-agent"

    async def handle(self, message: A2AMessage) -> A2AMessage:
        if message.task == "confirm_sellers":
            return await self._confirm_sellers(message)
        order_id = message.payload["order_id"]
        order_ev = await self.fetch("get_order", order_id=order_id)
        order = _obj(order_ev)
        if order_ev is None or order is None or order.get("order_id") != order_id:
            if order_ev is not None:
                self.consumed(order_ev, "ORDER_ID_MISMATCH")
            return reply_to(message, payload={"order": None}, decision_code="ORDER_NOT_FOUND")
        items_ev = await self.fetch("get_order_items", order_id=order_id)
        facts = analyze_order(order, _rows(items_ev), self.ctx.opened_at)
        self.consumed(order_ev, f"ORDER_STATUS_{facts.status.upper()}")
        if items_ev is not None:
            self.consumed(
                items_ev,
                "ITEMS_SCOPED",
                items_in_scope=len(facts.items),
                rows_out_of_scope=facts.excluded_item_rows,
                duplicate_rows=facts.duplicate_item_rows,
            )
        product_ev = None
        if EVIDENCE_MODE == "extended":
            product_ev = await self.fetch("get_product_context", order_id=order_id)
            if product_ev is not None:
                known = {row.get("order_item_id") for row in _rows(product_ev)}
                covered = sum(item in known for item in facts.item_ids)
                self.consumed(
                    product_ev,
                    "PRODUCTS_MATCH_ITEMS"
                    if covered == len(facts.item_ids)
                    else "PRODUCTS_PARTIAL",
                    items_with_product=covered,
                )
        refs = [ev.ref for ev in (order_ev, items_ev, product_ev) if ev is not None]
        return reply_to(
            message, payload={"order": facts}, decision_code="ORDER_FACTS_READY", evidence_refs=refs
        )

    async def _confirm_sellers(self, message: A2AMessage) -> A2AMessage:
        wanted = list(message.payload["seller_ids"])
        sellers_ev = await self.fetch("get_sellers", order_id=message.payload["order_id"])
        known = {row.get("seller_id") for row in _rows(sellers_ev)}
        confirmed = [seller for seller in wanted if seller in known]
        code = "SELLERS_CONFIRMED" if confirmed == wanted and wanted else "SELLERS_UNCONFIRMED"
        if sellers_ev is not None:
            self.consumed(sellers_ev, code, sellers_confirmed=len(confirmed))
        refs = [sellers_ev.ref] if sellers_ev is not None else []
        return reply_to(
            message, payload={"confirmed": confirmed}, decision_code=code, evidence_refs=refs
        )


# --------------------------------------------------------------------------- payment


class PaymentAgent(Agent):
    name = "payment-agent"

    async def handle(self, message: A2AMessage) -> A2AMessage:
        order: OrderFacts = message.payload["order"]
        timeline_ev = await self.fetch("get_payment_timeline", order_id=order.order_id)
        refund_ev = await self.fetch("get_refund_timeline", order_id=order.order_id)
        facts = analyze_payments(_obj(timeline_ev), _obj(refund_ev), order, self.ctx.opened_at)
        if timeline_ev is not None:
            self.consumed(
                timeline_ev,
                f"PAYMENT_{facts.verdict.upper()}",
                captures_in_scope=len(facts.captures),
                captured_total_brl=float(facts.captured_total),
                rows_out_of_scope=facts.excluded_rows,
                duplicate_rows=facts.duplicate_rows,
            )
        if refund_ev is not None:
            refund_code = (
                "REFUND_FAILED"
                if facts.failed_refund_total
                else ("REFUND_PENDING" if facts.pending_refund_total else "REFUND_NONE_IN_SCOPE")
            )
            self.consumed(refund_ev, refund_code)
        rows_ev = None
        if EVIDENCE_MODE == "extended":
            rows_ev = await self.fetch("get_order_payments", order_id=order.order_id)
            if rows_ev is not None:
                timeline_rows = (_obj(timeline_ev) or {}).get("payments") or []
                same = sorted(map(str, _rows(rows_ev))) == sorted(map(str, timeline_rows))
                self.consumed(
                    rows_ev,
                    "PAYMENT_ROWS_MATCH_TIMELINE" if same else "PAYMENT_ROWS_DIFFER",
                    payment_rows=len(_rows(rows_ev)),
                )
        refs = [ev.ref for ev in (timeline_ev, refund_ev, rows_ev) if ev is not None]
        return reply_to(
            message,
            payload={"payment": facts if timeline_ev is not None else None},
            decision_code=f"PAYMENT_{facts.verdict.upper()}",
            evidence_refs=refs,
        )


# --------------------------------------------------------------------------- shipment


class ShipmentAgent(Agent):
    name = "shipment-agent"

    async def handle(self, message: A2AMessage) -> A2AMessage:
        order: OrderFacts = message.payload["order"]
        summary_ev = await self.fetch("get_shipment_summary", order_id=order.order_id)
        facts = analyze_shipment(_obj(summary_ev), order, self.ctx.opened_at)
        if summary_ev is not None:
            self.consumed(
                summary_ev,
                f"DELIVERY_{facts.verdict.upper()}",
                events_consistent=facts.consistent_events,
                events_out_of_scope=facts.excluded_events,
                conflicts=len(facts.conflicts),
            )
        refs = [summary_ev.ref] if summary_ev is not None else []
        return reply_to(
            message,
            payload={"shipment": facts if summary_ev is not None else None},
            decision_code=f"DELIVERY_{facts.verdict.upper()}",
            evidence_refs=refs,
        )


# --------------------------------------------------------------------------- policy


@dataclass
class Decision:
    issue: str
    case_status: str
    action: str
    refund: Decimal
    refund_lines: list[dict[str, Any]]
    parties: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    policy_reference_brl: float | None


def _line(reason: str, amount: Decimal, entity: str | None) -> dict[str, Any]:
    return {"reason_code": reason, "amount_brl": float(amount), "entity_id": entity}


def compute_refund(
    issue: str, action: str, order: OrderFacts | None, payment: PaymentFacts | None
) -> tuple[Decimal, list[dict[str, Any]], list[dict[str, Any]]]:
    """Refund amount derived from this order's own in-scope evidence."""
    if order is None or payment is None:
        return Decimal("0.00"), [], []
    zero = Decimal("0.00")
    conflicts: list[dict[str, Any]] = []
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        amount = payment.captured_total
        lines = [_line(action, amount, order.order_id)] if amount > 0 else []
    elif issue in {"late_delivery_seller", "late_delivery_logistics"}:
        freight = order.freight_total
        amount = freight
        if 0 < payment.captured_total < freight:
            # Never refund more freight than the customer was actually charged.
            amount = payment.captured_total
            conflicts.append(
                {
                    "field": "freight_value",
                    "sources": ["get_order_items", "get_payment_timeline"],
                    "selected_source": "get_payment_timeline",
                    "resolution_code": "REFUND_CAPPED_AT_CAPTURED_AMOUNT",
                }
            )
        entity = order.item_ids[0] if len(order.item_ids) == 1 else order.order_id
        lines = [_line(action, amount, entity)] if amount > 0 else []
    elif issue == "duplicate_charge":
        amount = payment.duplicate_amount
        lines = [_line(action, amount, order.order_id)] if amount > 0 else []
    elif issue == "payment_mismatch":
        amount = payment.mismatch_total
        lines = [_line(action, amount, order.order_id)] if amount > 0 else []
    elif issue == "refund_failed":
        amount = payment.failed_refund_total
        lines = [_line(action, amount, order.order_id)] if amount > 0 else []
    else:
        amount, lines = zero, []
    return amount.quantize(CENT), lines, conflicts


class PolicyAgent(Agent):
    name = "policy-agent"

    async def handle(self, message: A2AMessage) -> A2AMessage:
        issue: str = message.payload["issue"]
        order: OrderFacts | None = message.payload.get("order")
        payment: PaymentFacts | None = message.payload.get("payment")
        late_sellers: list[str] = message.payload.get("late_seller_ids", [])
        policy_ev = await self.fetch("get_policy", policy_version=self.ctx.case["policy_version"])
        rules = (_obj(policy_ev) or {}).get("rules") or {}
        rule = rules.get(issue)
        if rule is None:
            decision = Decision(
                issue=issue,
                case_status="needs_investigation",
                action=FALLBACK_ACTION,
                refund=Decimal("0.00"),
                refund_lines=[],
                parties=[{"party_type": "unknown", "party_id": None}],
                conflicts=[],
                policy_reference_brl=None,
            )
            code = "POLICY_RULE_MISSING"
        else:
            action = rule["recommended_action"]
            refund, lines, conflicts = compute_refund(issue, action, order, payment)
            decision = Decision(
                issue=issue,
                case_status=rule["case_status"],
                action=action,
                refund=refund,
                refund_lines=lines,
                parties=self._parties(rule, issue, order, late_sellers),
                conflicts=conflicts,
                policy_reference_brl=rule.get("refund_brl"),
            )
            code = action.upper()
        if policy_ev is not None:
            self.consumed(policy_ev, f"RULE_{issue.upper()}")
        refs = [policy_ev.ref] if policy_ev is not None else []
        self.ctx.trace.emit(
            case_id=self.ctx.case_id,
            event_type="policy_decided",
            actor=self.name,
            decision_code=code,
            evidence_refs=refs or None,
            attributes={
                "primary_issue": issue,
                "case_status": decision.case_status,
                "refund_brl": float(decision.refund),
            },
        )
        return reply_to(
            message, payload={"decision": decision}, decision_code=code, evidence_refs=refs
        )

    @staticmethod
    def _parties(
        rule: dict[str, Any], issue: str, order: OrderFacts | None, late_sellers: list[str]
    ) -> list[dict[str, Any]]:
        parties: list[dict[str, Any]] = []
        for party in rule.get("responsible_parties") or []:
            party_type = party.get("party_type", "unknown")
            if party_type == "seller":
                # The policy's example seller id is not this order's seller: resolve from evidence.
                sellers = late_sellers if issue == "late_delivery_seller" else []
                sellers = sellers or (order.seller_ids if order else [])
                parties += [{"party_type": "seller", "party_id": s} for s in sellers] or [
                    {"party_type": "seller", "party_id": None}
                ]
            else:
                parties.append({"party_type": party_type, "party_id": None})
        return parties[:5] or [{"party_type": "unknown", "party_id": None}]


# --------------------------------------------------------------------------- verifier

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "schemas"


@lru_cache(maxsize=1)
def _verifier_contracts() -> Contracts | None:
    return Contracts(CONTRACTS_ROOT) if CONTRACTS_ROOT.is_dir() else None


FULL_REFUND_ACTIONS = {"issue_refund", "retry_refund"}


class VerifierAgent(Agent):
    name = "verifier"

    def __init__(self, ctx: CaseContext, bus: A2ABus) -> None:
        super().__init__(ctx, bus)
        self._contracts = _verifier_contracts()

    async def handle(self, message: A2AMessage) -> A2AMessage:
        output: dict[str, Any] = message.payload["output"]
        problems = self.check(output)
        code = "VERIFICATION_PASSED" if not problems else "VERIFICATION_FAILED"
        self.ctx.trace.emit(
            case_id=self.ctx.case_id,
            event_type="verification_completed",
            actor=self.name,
            decision_code=code,
            evidence_refs=output.get("evidence_refs") or None,
            attributes={
                "problems": len(problems),
                "first_problem": problems[0] if problems else None,
            },
        )
        return reply_to(message, payload={"problems": problems}, decision_code=code)

    def check(self, output: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        registry = self.ctx.registry
        if output.get("case_id") != self.ctx.case_id:
            problems.append("CASE_ID_MISMATCH")
        if self._contracts is not None:
            try:
                self._contracts.validate_output(output, "draft")
            except ContractError:
                problems.append("SCHEMA_INVALID")
        refs = list(output.get("evidence_refs", []))
        claim_refs = [r for c in output.get("claim_assessments", []) for r in c["evidence_refs"]]
        if any(ref not in registry for ref in refs + claim_refs):
            problems.append("UNKNOWN_EVIDENCE_REF")
        if any(ref not in self.ctx.consumed_refs for ref in refs + claim_refs):
            problems.append("EVIDENCE_NOT_TRACED")
        if not set(claim_refs) <= set(refs):
            problems.append("CLAIM_REF_NOT_IN_OUTPUT")
        issue = output["assessment"]["primary_issue"]
        domains = {registry.domain_of(ref) for ref in refs}
        if not REQUIRED_DOMAINS.get(issue, frozenset()) <= domains:
            problems.append("MISSING_REQUIRED_EVIDENCE")

        entities = output["affected_entities"]
        claimed = self.ctx.case["customer_request"]["claimed_order_id"]
        if not set(entities["order_ids"]) <= {claimed}:
            problems.append("ENTITY_OUT_OF_SCOPE")

        money = output["financial_resolution"]
        lines_total = sum(
            (Decimal(str(line["amount_brl"])) for line in money["refund_lines"]), Decimal("0")
        )
        refund = Decimal(str(money["recommended_refund_brl"]))
        if lines_total.quantize(CENT) != refund.quantize(CENT):
            problems.append("REFUND_LINES_TOTAL_MISMATCH")

        status = output["assessment"]["case_status"]
        actions = output["resolution_actions"]
        if not actions or len(actions) != len(set(actions)):
            problems.append("ACTIONS_EMPTY_OR_DUPLICATE")
        if status == "no_action" and (refund > 0 or money["refund_lines"]):
            problems.append("NO_ACTION_WITH_REFUND")
        if refund > 0 and status != "action_required":
            problems.append("REFUND_WITHOUT_ACTION_STATUS")

        parties = output["root_cause_analysis"]["responsible_parties"]
        party_types = {p["party_type"] for p in parties}
        if issue == "late_delivery_seller" and "seller" not in party_types:
            problems.append("SELLER_RESPONSIBILITY_MISSING")
        if issue == "late_delivery_logistics" and "seller" in party_types:
            problems.append("SELLER_BLAMED_FOR_LOGISTICS")
        seller_ids = set(entities["seller_ids"])
        if any(p["party_type"] == "seller" and p["party_id"] not in seller_ids for p in parties):
            problems.append("RESPONSIBLE_SELLER_NOT_IN_ENTITIES")
        confidence = output["assessment"]["confidence"]
        if not 0.0 <= confidence <= 1.0:
            problems.append("CONFIDENCE_OUT_OF_BOUNDS")
        if issue == "insufficient_evidence" and confidence > 0.6:
            problems.append("OVERCONFIDENT_WITHOUT_EVIDENCE")
        return problems
