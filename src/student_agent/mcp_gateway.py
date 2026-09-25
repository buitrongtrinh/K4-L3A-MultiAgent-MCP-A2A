from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from .contracts import Contracts


# Session-level failures: the connection is unusable and the whole case must be retried
# on a fresh session. Tool-level failures (ToolCallError) are answers, not outages.
class GatewayUnavailableError(RuntimeError):
    """A tool that must always answer (e.g. the policy document) failed: treat as an outage."""


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    GatewayUnavailableError,
    httpx2.TransportError,
    httpx2.HTTPStatusError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    MCPError,
    TimeoutError,
    OSError,
)


class ToolCallError(RuntimeError):
    """The MCP server executed the tool and reported an error (e.g. no matching records)."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"MCP tool {tool_name} failed: {message or 'unknown error'}")
        self.tool_name = tool_name


class EvidenceGateway:
    def __init__(self, session: ClientSession, contracts: Contracts) -> None:
        self._session = session
        self._contracts = contracts

    async def list_tools(self) -> list[str]:
        response = await self._session.list_tools()
        return sorted(tool.name for tool in response.tools)

    async def describe_tools(self) -> list[dict[str, Any]]:
        response = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
            for tool in sorted(response.tools, key=lambda item: item.name)
        ]

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        payload = {"case_id": case_id, **arguments}
        result = await self._session.call_tool(tool_name, arguments=payload)
        # mcp>=2 renamed isError -> is_error; accept both spellings.
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            message = " ".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            raise ToolCallError(tool_name, message)
        evidence = getattr(result, "structuredContent", None)
        if evidence is None:
            evidence = getattr(result, "structured_content", None)
        if evidence is None:
            text_blocks = [block.text for block in result.content if getattr(block, "text", None)]
            if len(text_blocks) != 1:
                raise ValueError(f"MCP tool {tool_name} did not return one evidence object")
            evidence = json.loads(text_blocks[0])
        self._contracts.validate_evidence(evidence, f"MCP tool {tool_name}")
        return evidence


@asynccontextmanager
async def connect_gateway(
    endpoint: str, team_api_key: str, contracts: Contracts
) -> AsyncIterator[EvidenceGateway]:
    headers = {"Authorization": f"Bearer {team_api_key}"}
    timeout = httpx2.Timeout(300.0, connect=30.0, write=30.0, pool=30.0)
    async with (
        httpx2.AsyncClient(headers=headers, timeout=timeout) as http_client,
        streamable_http_client(endpoint, http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield EvidenceGateway(session, contracts)
