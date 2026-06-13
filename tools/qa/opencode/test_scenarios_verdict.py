"""Verdict tests for the opencode QA scenario gate (run on the pod harness).

Flat-module imports match the harness's own layout, so run from this directory:
``cd tools/qa/opencode && python -m pytest test_scenarios_verdict.py``. The main
suite (``testpaths = ["tests"]``) does not collect these.
"""

from __future__ import annotations

import json
from pathlib import Path

from scenarios import Scenario, ScenarioStatus, _poll_verdict


def _scenario() -> Scenario:
    return Scenario(
        name="tool-turn",
        prompt="search the indexed reference",
        expected=(),
        forbidden=(),
        timeout_s=60.0,
    )


def _write_dispatch_event(workspace: Path) -> None:
    events = workspace / ".lilbee" / "qa-events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps({"type": "qa.tool.after", "tool": "lilbee_search"}) + "\n",
        encoding="utf-8",
    )


def _write_chat_completions(workspace: Path, count: int) -> None:
    log = workspace / ".lilbee" / "data" / "logs" / "launcher-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    line = '127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
    log.write_text(line * count, encoding="utf-8")


def _verdict(workspace: Path):
    return _poll_verdict(
        _scenario(),
        workspace,
        pane="opencode pane text",
        baseline_calls=0,
        baseline_dispatches=0,
        start=0.0,
    )


def test_dispatch_with_lilbee_chat_completion_passes(tmp_path: Path) -> None:
    _write_dispatch_event(tmp_path)
    _write_chat_completions(tmp_path, 1)
    verdict = _verdict(tmp_path)
    assert verdict is not None
    assert verdict.status is ScenarioStatus.PASS
    assert "lilbee chat completion" in verdict.detail


def test_dispatch_without_lilbee_chat_completion_fails_as_zen_fallback(tmp_path: Path) -> None:
    # The Zen-fallback signature: lilbee_search fired (MCP) but lilbee served no
    # chat completion because opencode's pin fell back to its own hosted model.
    _write_dispatch_event(tmp_path)
    _write_chat_completions(tmp_path, 0)
    verdict = _verdict(tmp_path)
    assert verdict is not None
    assert verdict.status is ScenarioStatus.FAIL
    assert "no chat completion" in verdict.detail
    assert "hosted provider" in verdict.detail


def test_no_dispatch_keeps_waiting(tmp_path: Path) -> None:
    _write_chat_completions(tmp_path, 1)  # chat but no search dispatch yet
    assert _verdict(tmp_path) is None
