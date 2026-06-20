"""Verdict tests for the opencode QA scenario gate (run on the pod harness).

Flat-module imports match the harness's own layout, so run from this directory:
``cd tools/qa/opencode && python -m pytest --noconftest test_scenarios_verdict.py``
(``--noconftest`` skips the parent ``tools/qa/conftest.py``, which pulls pod-only
deps). The main suite (``testpaths = ["tests"]``) does not collect these.
"""

from __future__ import annotations

import json
from pathlib import Path

from scenarios import (
    Scenario,
    ScenarioResult,
    ScenarioStatus,
    _poll_verdict,
    downgrade_if_ungrounded,
)


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


# downgrade_if_ungrounded runs on the SETTLED pane (after wait_for_answer_settle),
# so these inject a settled pane directly, which is exactly how the harness calls it.
def _passing() -> ScenarioResult:
    return ScenarioResult(name="x", status=ScenarioStatus.PASS, detail="1 dispatch + completion")


def test_downgrade_ungrounded_pass_becomes_fail() -> None:
    pane = (
        "The AStarGrid2D class and its get_id_path method are not found in the indexed reference."
    )
    out = downgrade_if_ungrounded(_passing(), pane)
    assert out.status is ScenarioStatus.FAIL
    assert "ungrounded" in out.detail


def test_downgrade_keeps_grounded_pass() -> None:
    # A substantive grounded answer is left untouched (no over-fire).
    pane = (
        "Object.connect signature: func connect(signal, callable, flags=0). CONNECT_DEFERRED is 1."
    )
    assert downgrade_if_ungrounded(_passing(), pane).status is ScenarioStatus.PASS


def test_downgrade_noop_on_non_pass() -> None:
    # The gate only downgrades a PASS; a FAIL (e.g. no dispatch) keeps its
    # accurate detail rather than being relabeled "ungrounded".
    fail = ScenarioResult(name="x", status=ScenarioStatus.FAIL, detail="no dispatch")
    pane = "AStarGrid2D get_id_path is not found in the indexed reference."
    out = downgrade_if_ungrounded(fail, pane)
    assert out.status is ScenarioStatus.FAIL
    assert out.detail == "no dispatch"


def test_downgrade_scans_only_the_tail() -> None:
    # An ungrounded phrase from earlier mid-search reasoning, pushed out of the
    # tail by a long grounded final answer, must not false-fail.
    early = "search 1: not found in the indexed set; retrying. "
    grounded = "get_id_path returns a PackedInt64Array of point ids. " * 80
    assert downgrade_if_ungrounded(_passing(), early + grounded).status is ScenarioStatus.PASS
