from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .cases import load_case_set
from .config import Settings
from .contracts import Contracts
from .mcp_gateway import TRANSIENT_ERRORS, connect_gateway
from .submission import package_submission, validate_artifacts
from .trace import TraceBuffer, TraceWriter
from .workflow import solve_case

MAX_CASE_ATTEMPTS = 3
DEFAULT_CONCURRENCY = 1


def _root(value: str) -> Path:
    return Path(value).resolve()


def _format_params(input_schema: dict[str, Any] | None) -> list[str]:
    schema = input_schema or {}
    required = set(schema.get("required", []))
    lines = []
    for name, spec in schema.get("properties", {}).items():
        kind = spec.get("type", "any")
        marker = "*" if name in required else " "
        detail = spec.get("description") or ""
        if "enum" in spec:
            detail = f"{detail} enum={spec['enum']}".strip()
        lines.append(f"    {marker} {name}: {kind}  {detail}".rstrip())
    return lines


async def _show_tools(root: Path, *, details: bool, as_json: bool) -> None:
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        if not (details or as_json):
            for tool in await gateway.list_tools():
                print(tool)
            return
        tools = await gateway.describe_tools()
    if as_json:
        print(json.dumps(tools, ensure_ascii=False, indent=2))
        return
    for tool in tools:
        print(tool["name"])
        if tool["description"]:
            print(f"  {tool['description']}")
        print("  params (* = required):")
        print("\n".join(_format_params(tool["input_schema"])) or "    (none)")
        print()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_transient(inner) for inner in exc.exceptions)
    return isinstance(exc, TRANSIENT_ERRORS)


async def _solve_with_retry(
    case: dict[str, Any], settings: Settings, contracts: Contracts
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Solve one case on its own MCP session; retry the whole case on session failures.

    Events are buffered per attempt, so only the successful attempt reaches the trace and
    its output cites only refs produced by that attempt's calls.
    """
    case_id = case["case_id"]
    for attempt in range(1, MAX_CASE_ATTEMPTS + 1):
        buffer = TraceBuffer(contracts)
        buffer.emit(
            case_id=case_id,
            event_type="case_received",
            actor="coordinator",
            attributes={"attempt": attempt, "policy_version": case.get("policy_version")},
        )
        try:
            async with connect_gateway(
                settings.mcp_endpoint, settings.team_api_key, contracts
            ) as gateway:
                output = await solve_case(case, gateway, buffer)
        except Exception as exc:
            if not _is_transient(exc) or attempt == MAX_CASE_ATTEMPTS:
                raise RuntimeError(f"{case_id}: attempt {attempt} failed: {exc!r}") from exc
            print(f"[retry] {case_id} attempt {attempt}: {type(exc).__name__}", file=sys.stderr)
            await asyncio.sleep(2**attempt)
            continue
        contracts.validate_output(output, f"outputs/{case_id}.json")
        if output.get("case_id") != case_id:
            raise ValueError(f"solver returned a mismatched case_id for {case_id}")
        buffer.emit(
            case_id=case_id,
            event_type="case_finalized",
            actor="coordinator",
            decision_code=output["assessment"]["primary_issue"].upper(),
            evidence_refs=output["evidence_refs"] or None,
        )
        return output, buffer.events
    raise RuntimeError(f"{case_id}: no attempt succeeded")


async def _discover_tools(settings: Settings, contracts: Contracts) -> list[str]:
    for attempt in range(1, MAX_CASE_ATTEMPTS + 1):
        try:
            async with connect_gateway(
                settings.mcp_endpoint, settings.team_api_key, contracts
            ) as gateway:
                return await gateway.list_tools()
        except Exception as exc:
            if not _is_transient(exc) or attempt == MAX_CASE_ATTEMPTS:
                raise RuntimeError(f"MCP Gateway unreachable: {exc!r}") from exc
            print(
                f"[retry] tool discovery attempt {attempt}: {type(exc).__name__}", file=sys.stderr
            )
            await asyncio.sleep(2**attempt)
    return []


async def _run(root: Path) -> None:
    settings = Settings.load(root)
    case_set = load_case_set(root)
    contracts = Contracts(root / "contracts" / "schemas")
    output_root = root / "outputs"
    trace_path = root / "traces" / "trace.jsonl"
    # Probe the gateway before touching previous artifacts, so an outage never wipes them.
    discovered_tools = await _discover_tools(settings, contracts)
    if not discovered_tools:
        raise RuntimeError("MCP Gateway returned no tools")

    output_root.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("*.json"):
        stale.unlink()
    trace_path.unlink(missing_ok=True)
    trace = TraceWriter(trace_path, contracts)

    limit = asyncio.Semaphore(int(os.getenv("DAY09_CONCURRENCY", DEFAULT_CONCURRENCY)))
    done = 0

    async def run_case(case_id: str) -> None:
        nonlocal done
        async with limit:
            output, events = await _solve_with_retry(case_set.cases[case_id], settings, contracts)
        target = output_root / f"{case_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        trace.write_events(events)
        done += 1
        assessment = output["assessment"]
        print(
            f"[{done:3d}/{len(case_set.case_ids)}] {case_id} {assessment['primary_issue']} "
            f"({assessment['case_status']}, conf={assessment['confidence']})",
            file=sys.stderr,
        )

    await asyncio.gather(*(run_case(case_id) for case_id in case_set.case_ids))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Day09 L3A student workflow")
    result.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-inputs", help="validate case-set.json and all 100 inputs")
    mcp_tools = commands.add_parser("mcp-tools", help="authenticate and list discovered MCP tools")
    mcp_tools.add_argument(
        "--details", action="store_true", help="show each tool's description and parameters"
    )
    mcp_tools.add_argument(
        "--json", action="store_true", help="print full tool metadata (schemas) as JSON"
    )
    commands.add_parser("run", help="run the implemented workflow for all cases")
    commands.add_parser("validate", help="validate outputs and observable trace")
    package = commands.add_parser("package", help="validate and build the submission ZIP")
    package.add_argument("--output", default="dist/submission.zip")
    return result


def main() -> None:
    args = parser().parse_args()
    root = _root(args.root)
    try:
        if args.command == "validate-inputs":
            case_set = load_case_set(root)
            print(
                f"OK: {case_set.variant_id} / {case_set.version} / {len(case_set.case_ids)} cases"
            )
        elif args.command == "mcp-tools":
            asyncio.run(_show_tools(root, details=args.details, as_json=args.json))
        elif args.command == "run":
            asyncio.run(_run(root))
        elif args.command == "validate":
            case_set = load_case_set(root)
            contracts = Contracts(root / "contracts" / "schemas")
            _, trace = validate_artifacts(root, case_set, contracts)
            print(f"OK: {len(case_set.case_ids)} outputs / {len(trace)} trace events")
        elif args.command == "package":
            destination = package_submission(root, root / args.output)
            print(f"OK: {destination}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
