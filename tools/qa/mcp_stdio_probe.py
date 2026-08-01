"""Drive a real ``lilbee mcp`` subprocess the way an agent does.

The unit suite connects a client to an in-process server over memory streams.
This spawns the actual stdio entry point and speaks to it over pipes, so the
launcher, the tool registration gates and the transport are all live. It is the
closest check to an agent (opencode, Claude Desktop) attaching to lilbee without
needing a model loaded.

By default it points the models dir at an empty directory, so the fleet finds
nothing installed and starts no engine. Pass --live-models on a pod, where an
engine is wanted, to run against the real models dir instead.

    uv run python tools/qa/mcp_stdio_probe.py
    uv run python tools/qa/mcp_stdio_probe.py --live-models --call-search "vector store"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters

_CORE_TOOLS = frozenset(
    {"search", "sync", "add", "remove", "status", "list_documents", "settings_list"}
)


class Probe:
    """Collects pass/fail checks so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.results.append((ok, name, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{f' -- {detail}' if detail else ''}")
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.results if not ok)


async def run(args: argparse.Namespace) -> int:
    probe = Probe()
    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="lilbee-mcp-probe-")
    env["LILBEE_DATA_ROOT"] = str(Path(tmp) / "data")
    if not args.live_models:
        # An empty models dir leaves every role uninstalled, so the fleet starts
        # no llama-server. Without this the probe would load models on whatever
        # machine it runs on.
        env["LILBEE_MODELS_DIR"] = str(Path(tmp) / "models")
    Path(env["LILBEE_DATA_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(env.get("LILBEE_MODELS_DIR", tmp)).mkdir(parents=True, exist_ok=True)

    params = StdioServerParameters(command=sys.executable, args=["-m", "lilbee", "mcp"], env=env)
    print(f"spawning: {sys.executable} -m lilbee mcp")

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        probe.check(
            init.server_info.name == "lilbee", "handshake names lilbee", init.server_info.name
        )

        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}
        probe.check(bool(tools), "tools/list is non-empty", f"{len(tools)} tools")
        missing = sorted(_CORE_TOOLS - names)
        probe.check(not missing, "core tools present", f"missing={missing}" if missing else "")

        dirty = [
            f"{t.name}.{p}"
            for t in tools
            for p, spec in t.input_schema.get("properties", {}).items()
            if "title" in spec or "default" in spec
        ]
        probe.check(not dirty, "wire schemas trimmed", f"{dirty[:3]}" if dirty else "")

        listed = await session.call_tool("settings_list", {})
        probe.check(not listed.is_error and bool(listed.content), "settings_list round trips")

        refused = await session.call_tool("search", {"query": "   "})
        probe.check(
            not refused.is_error and "must not be empty" in str(refused.content),
            "empty search returns the error envelope",
        )

        unknown = await session.call_tool("definitely_not_a_tool", {})
        probe.check(bool(unknown.is_error), "unknown tool is a protocol error")

        if args.call_search:
            hit = await session.call_tool("search", {"query": args.call_search})
            probe.check(not hit.is_error, "live search call", str(hit.content)[:120])

    total = len(probe.results)
    print(f"\n{total - probe.failed}/{total} checks passed")
    return 1 if probe.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-models",
        action="store_true",
        help="use the real models dir (starts an engine); pod-only",
    )
    parser.add_argument(
        "--call-search",
        metavar="QUERY",
        help="also issue a real search, which needs an embedder",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
