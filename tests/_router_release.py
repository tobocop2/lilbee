"""A llama-swap router that reports a chosen release, found through the real lookup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import swap_manager as sm

ROUTER = Path("/usr/bin/llama-swap")


def router_reports(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    """Find the router at :data:`ROUTER` through the real lookup; its ``--version`` prints *stdout*.

    Every other tool and command resolves and runs as in any other test. Return the
    argv of each version probe the router gets.
    """
    probes: list[list[str]] = []
    real_run = subprocess.run
    real_resolve = binary_mod.resolve_engine_tool

    def run(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if argv == [str(ROUTER), "--version"]:
            probes.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return real_run(argv, *args, **kwargs)

    def resolve(tool: binary_mod.EngineTool, **kwargs: Any) -> Path:
        if tool is binary_mod.EngineTool.LLAMA_SWAP:
            return ROUTER
        return real_resolve(tool, **kwargs)

    monkeypatch.setattr(sm, "resolve_llama_swap", binary_mod.resolve_llama_swap)
    monkeypatch.setattr(binary_mod, "resolve_engine_tool", resolve)
    monkeypatch.setattr(binary_mod.subprocess, "run", run)
    return probes
