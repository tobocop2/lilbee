"""QA scenarios: tier prompts, the tool-dispatch PASS gate, answer-settle wait."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from harness_config import (
    _FAIL_FAST_MARKERS,
    _MULTI_TOOL_TIMEOUT_S,
    _OPENCODE_BOOT_SETTLE_S,
    _PANE_EXCERPT_TAIL,
    _PANE_IDLE_TIMEOUT_S,
    _POLL_INTERVAL_S,
    _RAW_MARKER_FORBIDDEN,
    ScenarioStatus,
)
from opencode_driver import tmux_capture, tmux_send
from serve import _count_ok_chat_completions


@dataclass(frozen=True)
class Scenario:
    """One QA cell: send *prompt*, wait for *expected* / *forbidden* in the pane."""

    name: str
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    timeout_s: float


_TOOL_DISPATCH_MARKER = "⚙ lilbee_search"
"""Glyph opencode renders ONLY when it dispatches a parsed tool call.

opencode draws U+2699 GEAR before the tool name once it has extracted a
structured ``tool_calls`` payload from lilbee's response and is invoking the
MCP tool. A model that emits raw ``{"name":"lilbee_search",...}`` JSON, a
model that never loads, or a serve that 500s never reaches this render path,
so the glyph cleanly separates real end-to-end dispatch from opencode chrome
(autocomplete hints, the MCP status panel, the picker badge) that merely
mentions the tool name. Cross-checked against the launcher-serve.log chat-
completion count so a stale pane can't carry a prior cell's glyph.
"""


# Tier prompts run against the indexed Godot 4 class reference. One prompt per
# tier (same yardstick across same-tier models). Natural phrasing -- the model
# should discover the lilbee_search MCP tool itself; only Smalls get a "look up"
# nudge. See tools/qa/opencode/README.md for the rationale.
_TIER_PROMPTS: dict[str, str] = {
    # Small models answer common Godot API from memory (Node._process etc.), so the
    # small-tier prompt is explicit ("search the indexed reference") and targets an
    # obscure class detail the model cannot fabricate, forcing a real lilbee_search.
    "small": (
        "Search the indexed Godot 4 class reference for the AStarGrid2D class. "
        "Then, using only what the search returns, tell me what its get_id_path "
        "method returns."
    ),
    # Mid models discover the tool on their own: natural phrasing, anchored to exact
    # signatures and flag values they would have to verify against the reference.
    "mid": (
        "In Godot 4 I am connecting signals between nodes. What is the exact "
        "signature of Object.connect, and what do the CONNECT_DEFERRED and "
        "CONNECT_ONE_SHOT flags do? Include their integer values."
    ),
    # Giants get the published level-generator prompt verbatim, from the
    # godot-level-generator RAG-vs-no-RAG benchmark (docs/benchmarks/
    # godot-level-generator.md). It only "passes" cleanly when the generated
    # GDScript uses real Godot 4 API (set_cell, AStarGrid2D.get_point_path, etc.),
    # verified via lilbee_search rather than hallucinated.
    "giant": (
        "make a procedural level generator that places wall and floor tiles and "
        "scatters collectibles using pathfinding"
    ),
}


def scenario_for_tier(tier: str) -> Scenario:
    """The single QA scenario for a model's tier (Godot-reference prompt)."""
    return Scenario(
        name=f"{tier} godot",
        prompt=_TIER_PROMPTS.get(tier, _TIER_PROMPTS["small"]),
        expected=(_TOOL_DISPATCH_MARKER,),
        forbidden=_RAW_MARKER_FORBIDDEN,
        timeout_s=_MULTI_TOOL_TIMEOUT_S,
    )


@dataclass
class ScenarioResult:
    name: str
    status: ScenarioStatus
    detail: str = ""
    pane_excerpt: str = ""
    elapsed_s: float = 0.0


_TOOL_TURN_MIN_COMPLETIONS = 2
"""Chat completions a genuine tool turn drives: the model's tool-call turn plus
the follow-up answer turn after opencode feeds the tool result back in.

A prose answer -- the model declining to call the tool and replying from
context -- drives only one completion. The scenarios share one opencode
session, so an earlier scenario's gear glyph stays visible in the pane; gating
on the glyph plus a single new completion let a prose turn pass on that stale
glyph. Requiring the per-scenario completion delta to reach two means the gear
must be backed by an actual ``opencode -> lilbee -> tool -> answer`` round-trip
in *this* scenario, not carried over from the previous one.
"""


def run_scenario(session: str, scenario: Scenario, workspace: Path) -> ScenarioResult:
    """Send the prompt and poll until a FRESH tool dispatch lands or idle/timeout.

    PASS requires the gear-glyph dispatch marker in the pane AND at least
    ``_TOOL_TURN_MIN_COMPLETIONS`` new ``POST /v1/chat/completions 200`` in the
    launcher log since this scenario started. The two-completion delta defeats
    the stale-glyph trap: opencode keeps the prior turn's transcript visible, so
    a later scenario would otherwise match an earlier scenario's gear off a
    single prose completion without ever re-dispatching the tool itself.
    """
    baseline_calls = _count_ok_chat_completions(workspace)
    tmux_send(session, scenario.prompt)
    start = time.time()
    deadline = start + scenario.timeout_s
    last_pane = ""
    last_change_at = start
    last_pane_len = 0
    missing: list[str] = list(scenario.expected)
    while time.time() < deadline:
        pane = tmux_capture(session)
        last_pane = pane
        pane_lower = pane.lower()
        if pane != "" and len(pane) != last_pane_len:
            last_pane_len = len(pane)
            last_change_at = time.time()
        fail_fast = next((m for m in _FAIL_FAST_MARKERS if m.lower() in pane_lower), None)
        if fail_fast is not None:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.FAIL,
                detail=f"fail-fast marker hit: {fail_fast!r}",
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        forbidden_hits = [s for s in scenario.forbidden if s.lower() in pane_lower]
        if forbidden_hits:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.FAIL,
                detail=f"forbidden substring(s) appeared: {forbidden_hits}",
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        missing = [s for s in scenario.expected if s.lower() not in pane_lower]
        new_calls = _count_ok_chat_completions(workspace) - baseline_calls
        fresh_call = new_calls >= _TOOL_TURN_MIN_COMPLETIONS
        if not missing and fresh_call:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.PASS,
                detail="gear dispatch + fresh chat completion",
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        idle_for = time.time() - last_change_at
        if idle_for > _PANE_IDLE_TIMEOUT_S and time.time() - start > _OPENCODE_BOOT_SETTLE_S:
            detail = (
                f"pane idle {idle_for:.0f}s; missing {missing}"
                if missing
                else (
                    f"pane idle {idle_for:.0f}s; gear seen but only {new_calls} new "
                    f"completion(s), need {_TOOL_TURN_MIN_COMPLETIONS} (prose, no re-dispatch)"
                )
            )
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.TIMEOUT,
                detail=detail,
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        time.sleep(_POLL_INTERVAL_S)
    return ScenarioResult(
        name=scenario.name,
        status=ScenarioStatus.TIMEOUT,
        detail=f"missing after {scenario.timeout_s:.0f}s: {missing}",
        pane_excerpt=last_pane[-_PANE_EXCERPT_TAIL:],
        elapsed_s=scenario.timeout_s,
    )


_ANSWER_SETTLE_TIMEOUT_S = 240.0
_ANSWER_SETTLE_QUIET_POLLS = 3
_ANSWER_SETTLE_INTERVAL_S = 4.0


def wait_for_answer_settle(session: str) -> None:
    """Wait for opencode to finish rendering the post-tool answer before capture.

    ``run_scenario`` returns the instant the gear marker plus a fresh chat
    completion appear, but opencode is still streaming the answer turn then, so a
    pane captured immediately shows the tool call and no answer. Poll until the
    pane stops changing for a few consecutive intervals (generation idle), capped
    by a timeout so a model that streams forever cannot hang the sweep.
    """
    deadline = time.monotonic() + _ANSWER_SETTLE_TIMEOUT_S
    prev = tmux_capture(session)
    quiet = 0
    while time.monotonic() < deadline:
        time.sleep(_ANSWER_SETTLE_INTERVAL_S)
        cur = tmux_capture(session)
        if cur == prev:
            quiet += 1
            if quiet >= _ANSWER_SETTLE_QUIET_POLLS:
                return
        else:
            quiet = 0
            prev = cur
