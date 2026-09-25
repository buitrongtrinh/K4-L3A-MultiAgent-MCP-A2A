"""Minimal in-process agent-to-agent (A2A) protocol.

Every message is correlated by ``case_id`` and carries the evidence refs it relies on.
The bus records each hop as an observable trace event (``task_assigned`` for requests,
``handoff`` for results), enforces a per-case hop budget to prevent loops, and bounds
each agent turn with a timeout. Only decision codes are traced, never reasoning text.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_HOPS_PER_CASE = 40
AGENT_TURN_TIMEOUT_S = 240.0


class TraceSink(Protocol):
    def emit(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class A2AMessage:
    case_id: str
    sender: str
    recipient: str
    kind: str  # "task" (request) or "result" (reply)
    task: str
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    decision_code: str | None = None
    message_id: str = ""
    reply_to: str | None = None


class A2AError(RuntimeError):
    pass


Handler = Callable[[A2AMessage], Awaitable[A2AMessage]]


class A2ABus:
    def __init__(self, case_id: str, trace: TraceSink) -> None:
        self.case_id = case_id
        self._trace = trace
        self._handlers: dict[str, Handler] = {}
        self._counter = itertools.count(1)
        self.hops = 0

    def register(self, name: str, handler: Handler) -> None:
        self._handlers[name] = handler

    def _record(self, message: A2AMessage, notes: dict[str, Any] | None = None) -> None:
        if message.case_id != self.case_id:
            raise A2AError(f"message for {message.case_id} routed on bus for {self.case_id}")
        self.hops += 1
        if self.hops > MAX_HOPS_PER_CASE:
            raise A2AError(f"hop budget exceeded for {self.case_id}")
        self._trace.emit(
            case_id=self.case_id,
            event_type="task_assigned" if message.kind == "task" else "handoff",
            actor=message.sender,
            target=message.recipient,
            decision_code=message.decision_code or message.task.upper(),
            evidence_refs=list(message.evidence_refs) or None,
            attributes={
                "message_id": message.message_id,
                "reply_to": message.reply_to,
                **(notes or {}),
            },
        )

    async def request(
        self,
        sender: str,
        recipient: str,
        task: str,
        payload: dict[str, Any] | None = None,
        *,
        notes: dict[str, str | int | float | bool | None] | None = None,
    ) -> A2AMessage:
        """Send a task to ``recipient`` and wait for its result message."""
        handler = self._handlers.get(recipient)
        if handler is None:
            raise A2AError(f"no agent registered as {recipient}")
        message = A2AMessage(
            case_id=self.case_id,
            sender=sender,
            recipient=recipient,
            kind="task",
            task=task,
            payload=payload or {},
            message_id=f"msg-{next(self._counter):03d}",
        )
        self._record(message, notes)
        reply = await asyncio.wait_for(handler(message), timeout=AGENT_TURN_TIMEOUT_S)
        reply = A2AMessage(
            case_id=reply.case_id,
            sender=reply.sender,
            recipient=sender,
            kind="result",
            task=reply.task,
            payload=reply.payload,
            evidence_refs=reply.evidence_refs,
            decision_code=reply.decision_code,
            message_id=f"msg-{next(self._counter):03d}",
            reply_to=message.message_id,
        )
        self._record(reply)
        return reply


def reply_to(
    message: A2AMessage,
    *,
    payload: dict[str, Any],
    decision_code: str,
    evidence_refs: list[str] | tuple[str, ...] = (),
) -> A2AMessage:
    return A2AMessage(
        case_id=message.case_id,
        sender=message.recipient,
        recipient=message.sender,
        kind="result",
        task=message.task,
        payload=payload,
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        decision_code=decision_code,
    )
