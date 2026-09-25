"""Coordinator for the L3A multi-agent workflow.

Flow per case (all hops are A2A messages recorded in the trace):

    coordinator -> order-agent      collect_order_facts   (get_order, get_order_items)
    coordinator -> payment-agent    collect_payment_facts (payment + refund timelines)
    coordinator -> shipment-agent   assess_delivery       (get_shipment_summary)
    coordinator adjudicates the primary issue from specialist findings
    coordinator -> policy-agent     decide_resolution     (get_policy)
    coordinator -> order-agent      confirm_sellers       (get_sellers, seller-liable issues only)
    coordinator -> verifier         verify_output         (no tools)
"""

from __future__ import annotations

from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .a2a import A2ABus
from .agents import (
    EVIDENCE_MODE,
    EVIDENCE_TOOLS,
    FULL_REFUND_ACTIONS,
    CaseContext,
    Decision,
    OrderAgent,
    PaymentAgent,
    PolicyAgent,
    ShipmentAgent,
    VerifierAgent,
)
from .analysis import (
    NON_ACTIONABLE,
    PRIMARY_ISSUES,
    OrderFacts,
    PaymentFacts,
    ShipmentFacts,
    candidate_issues,
    parse_ts,
    select_primary_issue,
)
from .evidence import EvidenceRegistry
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

COORDINATOR = "coordinator"
REFUND_CLAIM = "requested_full_refund"
PARTIAL_REFUND_ACTIONS = {"refund_freight", "refund_duplicate_charge", "reconcile_payment"}
SELLER_LIABLE = {"late_delivery_seller", "unavailable_order_paid"}
ITEM_LEVEL = {
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
}
ISSUE_TOOLS = {"get_order", "get_order_items", "get_shipment_summary", "get_sellers"}
MONEY_TOOLS = {"get_payment_timeline", "get_refund_timeline", "get_policy"}


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    ctx = CaseContext(
        case=case,
        gateway=gateway,
        trace=trace,
        registry=EvidenceRegistry(case["case_id"]),
        discovered_tools=set(await gateway.list_tools()),
        opened_at=parse_ts(case.get("opened_at")),
    )
    bus = A2ABus(ctx.case_id, trace)
    OrderAgent(ctx, bus)
    PaymentAgent(ctx, bus)
    ShipmentAgent(ctx, bus)
    PolicyAgent(ctx, bus)
    VerifierAgent(ctx, bus)
    return await Coordinator(ctx, bus).run()


class Coordinator:
    def __init__(self, ctx: CaseContext, bus: A2ABus) -> None:
        self.ctx = ctx
        self.bus = bus

    async def run(self) -> dict[str, Any]:
        request = self.ctx.case["customer_request"]
        claims = list(request.get("claims") or [])
        claimed_topic = next(
            (c.get("topic") for c in claims if c.get("topic") in PRIMARY_ISSUES), None
        )

        order_reply = await self.bus.request(
            COORDINATOR,
            "order-agent",
            "collect_order_facts",
            {"order_id": request["claimed_order_id"]},
        )
        order: OrderFacts | None = order_reply.payload.get("order")
        payment: PaymentFacts | None = None
        shipment: ShipmentFacts | None = None
        if order is not None:
            # Sequential on purpose: one in-flight MCP call per session keeps the server-side
            # audit trail unambiguous (see ARCHITECTURE.md, Reproducibility).
            payment_reply = await self.bus.request(
                COORDINATOR, "payment-agent", "collect_payment_facts", {"order": order}
            )
            shipment_reply = await self.bus.request(
                COORDINATOR, "shipment-agent", "assess_delivery", {"order": order}
            )
            payment = payment_reply.payload.get("payment")
            shipment = shipment_reply.payload.get("shipment")

        candidates = candidate_issues(order, payment, shipment)
        issue = select_primary_issue(
            claimed_topic,
            candidates,
            has_core_evidence=order is not None and payment is not None and shipment is not None,
        )

        policy_reply = await self.bus.request(
            COORDINATOR,
            "policy-agent",
            "decide_resolution",
            {
                "issue": issue,
                "order": order,
                "payment": payment,
                "late_seller_ids": shipment.late_seller_ids if shipment else [],
            },
            notes={
                "primary_issue": issue,
                "claimed_topic": claimed_topic,
                "candidates": ",".join(candidates) or None,
            },
        )
        decision: Decision = policy_reply.payload["decision"]

        # Liable sellers are resolved from item evidence by the policy agent, then confirmed
        # against seller records by the order agent (the "all" diagnostic confirms always).
        liable = issue in SELLER_LIABLE
        seller_ids: list[str] = []
        if order is not None and (liable or EVIDENCE_MODE == "all"):
            wanted = [p["party_id"] for p in decision.parties if p["party_type"] == "seller"]
            wanted = [s for s in wanted if s] if liable else order.seller_ids
            if wanted:
                seller_reply = await self.bus.request(
                    COORDINATOR,
                    "order-agent",
                    "confirm_sellers",
                    {"order_id": order.order_id, "seller_ids": wanted},
                )
                if liable:
                    seller_ids = seller_reply.payload["confirmed"] or wanted

        conflicts = (shipment.conflicts if shipment else []) + decision.conflicts
        confidence = self._confidence(issue, claimed_topic, candidates, conflicts)
        output = self._assemble(
            claims, order, decision, seller_ids, conflicts, candidates, confidence
        )

        verdict = await self.bus.request(
            COORDINATOR, "verifier", "verify_output", {"output": output}
        )
        problems: list[str] = verdict.payload["problems"]
        if problems:
            # Do not ship a conclusion the verifier could not back: lower confidence sharply.
            output["assessment"]["confidence"] = min(confidence, 0.4)
            for claim in output.get("claim_assessments", []):
                claim["confidence"] = min(claim["confidence"], 0.4)
        return output

    @staticmethod
    def _confidence(
        issue: str, claimed: str | None, candidates: list[str], conflicts: list[dict[str, Any]]
    ) -> float:
        if issue == "insufficient_evidence":
            return 0.35
        confidence = 0.93
        if claimed is not None and claimed != issue:
            confidence -= 0.1  # evidence contradicts the customer's framing
        rivals = [c for c in candidates if c != issue and c not in NON_ACTIONABLE]
        confidence -= 0.08 * len(rivals)
        confidence -= 0.03 * len(conflicts)
        return round(max(0.3, min(0.97, confidence)), 2)

    def _assemble(
        self,
        claims: list[dict[str, Any]],
        order: OrderFacts | None,
        decision: Decision,
        seller_ids: list[str],
        conflicts: list[dict[str, Any]],
        candidates: list[str],
        confidence: float,
    ) -> dict[str, Any]:
        registry = self.ctx.registry
        issue = decision.issue
        evidence_refs = [
            ref
            for ref in registry.refs_for_tools(EVIDENCE_TOOLS[issue])
            if ref in self.ctx.consumed_refs
        ]
        issue_refs = [
            ref for ref in evidence_refs if registry.domain_of(ref) not in {"policy"}
        ] or evidence_refs
        money_tools = [t for t in EVIDENCE_TOOLS[issue] if t in MONEY_TOOLS]
        money_refs = [r for r in registry.refs_for_tools(money_tools) if r in evidence_refs]

        entities = {
            "order_ids": [order.order_id] if order else [],
            "item_ids": order.item_ids if order and issue in ITEM_LEVEL else [],
            "seller_ids": seller_ids,
            "payment_references": [],
            "shipment_ids": [],
        }
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "case_id": self.ctx.case_id,
            "assessment": {
                "primary_issue": issue,
                "case_status": decision.case_status,
                "confidence": confidence,
            },
            "affected_entities": entities,
            "claim_assessments": [
                self._assess_claim(
                    c, issue, decision, candidates, issue_refs, money_refs, confidence
                )
                for c in claims[:5]
            ],
            "root_cause_analysis": {
                "ranked_causes": [{"cause_code": issue.upper(), "rank": 1}],
                "responsible_parties": decision.parties,
            },
            "evidence_refs": evidence_refs,
            "data_conflicts": _dedupe(conflicts)[:5],
            "financial_resolution": {
                "currency": "BRL",
                "recommended_refund_brl": float(decision.refund),
                "refund_lines": decision.refund_lines,
            },
            "resolution_actions": [decision.action],
        }

    @staticmethod
    def _assess_claim(
        claim: dict[str, Any],
        issue: str,
        decision: Decision,
        candidates: list[str],
        issue_refs: list[str],
        money_refs: list[str],
        confidence: float,
    ) -> dict[str, Any]:
        topic = claim.get("topic")
        refs = issue_refs
        if topic == REFUND_CLAIM:
            refs = money_refs or issue_refs
            if issue == "insufficient_evidence" or decision.case_status == "needs_investigation":
                verdict = "insufficient_evidence"
            elif decision.refund <= 0:
                verdict = "unsupported"
            elif decision.action in FULL_REFUND_ACTIONS:
                verdict = "supported"
            else:
                verdict = "partially_supported"
        elif topic not in PRIMARY_ISSUES or issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif topic == issue:
            verdict = "unsupported" if issue in NON_ACTIONABLE else "supported"
        elif topic in candidates and topic not in NON_ACTIONABLE:
            verdict = "partially_supported"
        else:
            verdict = "unsupported"
        claim_confidence = (
            confidence if verdict != "insufficient_evidence" else min(confidence, 0.4)
        )
        if topic == REFUND_CLAIM:
            claim_confidence = round(max(0.3, claim_confidence - 0.05), 2)
        return {
            "claim_id": str(claim.get("claim_id", "claim"))[:64],
            "verdict": verdict,
            "confidence": claim_confidence,
            "evidence_refs": refs,
        }


def _dedupe(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for conflict in conflicts:
        seen.setdefault((conflict["field"], conflict["resolution_code"]), conflict)
    return list(seen.values())
