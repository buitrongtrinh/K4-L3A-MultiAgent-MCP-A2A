from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Evidence:
    ref: str
    tool: str
    domain: str
    data: Any
    result_hash: str


class EvidenceRegistry:
    """Evidence fetched for exactly one case.

    A fresh registry is created per case attempt, so a ref can only be cited by the case
    whose MCP call produced it. Refs are copied verbatim from the gateway, never built.
    """

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._by_ref: dict[str, Evidence] = {}
        self._by_tool: dict[str, Evidence] = {}

    def register(self, tool: str, envelope: dict[str, Any]) -> Evidence:
        evidence = Evidence(
            ref=envelope["evidence_ref"],
            tool=tool,
            domain=envelope["domain"],
            data=envelope["data"],
            result_hash=envelope["result_hash"],
        )
        self._by_ref[evidence.ref] = evidence
        self._by_tool[tool] = evidence
        return evidence

    def __contains__(self, ref: object) -> bool:
        return ref in self._by_ref

    def for_tool(self, tool: str) -> Evidence | None:
        return self._by_tool.get(tool)

    def refs_for_tools(self, tools: tuple[str, ...] | list[str]) -> list[str]:
        refs = [self._by_tool[tool].ref for tool in tools if tool in self._by_tool]
        return list(dict.fromkeys(refs))

    def domain_of(self, ref: str) -> str | None:
        evidence = self._by_ref.get(ref)
        return evidence.domain if evidence else None

    def all_refs(self) -> list[str]:
        return list(self._by_ref)
