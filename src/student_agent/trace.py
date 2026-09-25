from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Contracts


def _build_event(
    contracts: Contracts,
    *,
    case_id: str,
    event_type: str,
    actor: str,
    target: str | None,
    decision_code: str | None,
    tool_name: str | None,
    evidence_refs: list[str] | None,
    attributes: dict[str, str | int | float | bool | None] | None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": "day09-trace-event-v1",
        "event_id": f"evt_{secrets.token_urlsafe(18)}",
        "case_id": case_id,
        "event_type": event_type,
        "occurred_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "actor": actor,
    }
    optional = {
        "target": target,
        "decision_code": decision_code,
        "tool_name": tool_name,
        "evidence_refs": evidence_refs,
        "attributes": attributes,
    }
    event.update({key: value for key, value in optional.items() if value is not None})
    contracts.validate_trace(event, "trace event")
    return event


class TraceBuffer:
    """Collect one case's events in memory so a failed attempt never reaches the trace file."""

    def __init__(self, contracts: Contracts) -> None:
        self.contracts = contracts
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        *,
        case_id: str,
        event_type: str,
        actor: str,
        target: str | None = None,
        decision_code: str | None = None,
        tool_name: str | None = None,
        evidence_refs: list[str] | None = None,
        attributes: dict[str, str | int | float | bool | None] | None = None,
    ) -> dict[str, Any]:
        event = _build_event(
            self.contracts,
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            target=target,
            decision_code=decision_code,
            tool_name=tool_name,
            evidence_refs=evidence_refs,
            attributes=attributes,
        )
        self.events.append(event)
        return event


class TraceWriter:
    """Append observable workflow events. Never put prompts or chain-of-thought here."""

    def __init__(self, path: Path, contracts: Contracts) -> None:
        self.path = path
        self.contracts = contracts
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_events(self, events: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for event in events:
                self.contracts.validate_trace(event, "trace event")
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def emit(
        self,
        *,
        case_id: str,
        event_type: str,
        actor: str,
        target: str | None = None,
        decision_code: str | None = None,
        tool_name: str | None = None,
        evidence_refs: list[str] | None = None,
        attributes: dict[str, str | int | float | bool | None] | None = None,
    ) -> dict[str, Any]:
        event = _build_event(
            self.contracts,
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            target=target,
            decision_code=decision_code,
            tool_name=tool_name,
            evidence_refs=evidence_refs,
            attributes=attributes,
        )
        self.write_events([event])
        return event
