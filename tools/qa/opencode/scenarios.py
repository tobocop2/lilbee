"""QA scenarios: tier prompts, the tool-dispatch PASS gate, answer-settle wait."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from events import count_session_idles, count_tool_dispatches, has_session_error, read_events
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

_SEARCH_TOOL_SUBSTR = "lilbee_search"
"""Tool-name fragment the event tap matches for the dispatch gate (opencode
may namespace MCP tools with the server prefix, so substring, not equality)."""


@dataclass(frozen=True)
class Scenario:
    """One QA cell: send *prompt*, wait for *expected* / *forbidden* in the pane."""

    name: str
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    timeout_s: float
    # Minimum distinct lilbee_search dispatches the cell must drive to PASS. >1
    # forces sequential tool calls, which exercises the model's (and the engine's)
    # tool-call parse-back on more than the trivial single-call happy path.
    min_dispatches: int = 1


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
# tier (same yardstick across same-tier models). Each names TWO unrelated class
# details that no single search can return together, so a genuine answer needs
# at least two distinct lilbee_search calls -- a sequential-tool-use bar, not the
# trivial single-call one. Only Smalls get a "look up" nudge. See
# tools/qa/opencode/README.md for the rationale.
_TIER_PROMPTS: dict[str, str] = {
    # Small models answer common Godot API from memory, so the small-tier prompt is
    # explicit ("search the indexed reference") and targets two obscure, unrelated
    # class details the model cannot fabricate, forcing two real lilbee_search calls.
    "small": (
        "Search the indexed Godot 4 class reference for two separate things: first, "
        "what AStarGrid2D.get_id_path returns; second, what the TileMap.set_cell "
        "method does and its parameters. Look each up with its own search, then, "
        "using only what the searches return, give me both answers."
    ),
    # Mid models discover the tool on their own: natural phrasing, anchored to two
    # exact, unrelated details they would each have to verify against the reference.
    "mid": (
        "In Godot 4 I am wiring up node signals and grid pathfinding, and I need two "
        "things verified against the reference. First, the exact signature of "
        "Object.connect and what the CONNECT_DEFERRED flag does with its integer "
        "value. Second, what AStarGrid2D.get_id_path returns and the parameters it "
        "takes. Look each up separately and give me both, citing only the reference."
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


_MIN_TOOL_DISPATCHES = 2
"""A passing cell must drive at least this many distinct lilbee_search calls, so
the gate proves sequential tool use rather than a single lucky dispatch."""


def scenario_for_tier(tier: str) -> Scenario:
    """The single QA scenario for a model's tier (Godot-reference prompt)."""
    return Scenario(
        name=f"{tier} godot multi-tool",
        prompt=_TIER_PROMPTS.get(tier, _TIER_PROMPTS["small"]),
        expected=(_TOOL_DISPATCH_MARKER,),
        forbidden=_RAW_MARKER_FORBIDDEN,
        timeout_s=_MULTI_TOOL_TIMEOUT_S,
        min_dispatches=_MIN_TOOL_DISPATCHES,
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


def _poll_verdict(
    scenario: Scenario,
    workspace: Path,
    pane: str,
    *,
    baseline_calls: int,
    baseline_dispatches: int,
    start: float,
) -> ScenarioResult | None:
    """One poll iteration's verdict, or ``None`` to keep waiting.

    PASS gate: a fresh ``lilbee_search`` dispatch event past this scenario's
    baseline AND a fresh successful chat completion on lilbee's own serve, else
    (tap never loaded) the pane gear marker plus ``_TOOL_TURN_MIN_COMPLETIONS``
    fresh completions; forbidden-marker checks always run on the rendered pane.

    The chat-completion requirement closes the Zen-fallback hole: if opencode's
    model pin fails to resolve, opencode silently serves the chat from its own
    hosted provider while still calling lilbee's MCP search tool, so the dispatch
    event fires but lilbee serves no chat completion. Counting lilbee's own
    ``POST /v1/chat/completions`` 200s proves lilbee served the chat, not just the
    search.
    """

    def result(status: ScenarioStatus, detail: str) -> ScenarioResult:
        return ScenarioResult(
            name=scenario.name,
            status=status,
            detail=detail,
            pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
            elapsed_s=time.time() - start,
        )

    pane_lower = pane.lower()
    fail_fast = next((m for m in _FAIL_FAST_MARKERS if m.lower() in pane_lower), None)
    if fail_fast is not None:
        return result(ScenarioStatus.FAIL, f"fail-fast marker hit: {fail_fast!r}")
    forbidden_hits = [s for s in scenario.forbidden if s.lower() in pane_lower]
    if forbidden_hits:
        return result(ScenarioStatus.FAIL, f"forbidden substring(s) appeared: {forbidden_hits}")
    events = read_events(workspace)
    if has_session_error(events):
        return result(ScenarioStatus.FAIL, "session.error event from opencode")
    fresh_dispatches = count_tool_dispatches(events, _SEARCH_TOOL_SUBSTR) - baseline_dispatches
    fresh_call = _count_ok_chat_completions(workspace) - baseline_calls
    missing = [s for s in scenario.expected if s.lower() not in pane_lower]
    return _dispatch_verdict(
        result,
        fresh_dispatches,
        fresh_call,
        has_events=bool(events),
        missing=missing,
        min_dispatches=scenario.min_dispatches,
    )


def _dispatch_verdict(
    result: Callable[[ScenarioStatus, str], ScenarioResult],
    fresh_dispatches: int,
    fresh_call: int,
    *,
    has_events: bool,
    missing: list[str],
    min_dispatches: int,
) -> ScenarioResult | None:
    """Resolve the tool-dispatch PASS gate (event tap first, then pane fallback).

    PASS needs at least *min_dispatches* fresh ``lilbee_search`` calls, so a cell
    only clears once the model has driven the required number of sequential tool
    calls. A fresh dispatch without a fresh lilbee chat completion is the
    Zen-fallback signature (the model pin fell back to opencode's own hosted
    provider, which still calls the MCP search tool), so it FAILs rather than
    passing on the search alone.
    """
    if fresh_call >= 1 and 1 <= fresh_dispatches < min_dispatches:
        # The model called the tool and lilbee served the chat, but not enough
        # times yet; keep polling until it reaches min_dispatches or goes idle.
        return None
    if fresh_dispatches >= min_dispatches:
        if fresh_call < 1:
            return result(
                ScenarioStatus.FAIL,
                f"{fresh_dispatches} {_SEARCH_TOOL_SUBSTR} dispatch(es) but lilbee "
                "served no chat completion: the model pin fell back to opencode's "
                "own hosted provider, so lilbee served the search tool, not the chat",
            )
        return result(
            ScenarioStatus.PASS,
            f"{fresh_dispatches} {_SEARCH_TOOL_SUBSTR} dispatch(es) + lilbee chat completion",
        )
    if not has_events and not missing and fresh_call >= _TOOL_TURN_MIN_COMPLETIONS:
        return result(
            ScenarioStatus.PASS, "gear dispatch + fresh chat completion (pane fallback; no tap)"
        )
    return None


def run_scenario(session: str, scenario: Scenario, workspace: Path) -> ScenarioResult:
    """Send the prompt and poll until a FRESH tool dispatch lands or idle/timeout.

    Verdicts come from :func:`_poll_verdict` (event tap first, pane fallback);
    this loop owns only the prompt send, the pane-idle fail-fast, and the
    overall scenario timeout.
    """
    baseline_calls = _count_ok_chat_completions(workspace)
    baseline_dispatches = count_tool_dispatches(read_events(workspace), _SEARCH_TOOL_SUBSTR)
    tmux_send(session, scenario.prompt)
    start = time.time()
    deadline = start + scenario.timeout_s
    last_pane = ""
    last_change_at = start
    prev_pane = ""
    while time.time() < deadline:
        pane = tmux_capture(session)
        last_pane = pane
        # Full-content compare: a spinner swapping one glyph keeps the pane
        # LENGTH constant, which the old length check read as idle.
        if pane != "" and pane != prev_pane:
            prev_pane = pane
            last_change_at = time.time()
        verdict = _poll_verdict(
            scenario,
            workspace,
            pane,
            baseline_calls=baseline_calls,
            baseline_dispatches=baseline_dispatches,
            start=start,
        )
        if verdict is not None:
            return verdict
        idle_for = time.time() - last_change_at
        if idle_for > _PANE_IDLE_TIMEOUT_S and time.time() - start > _OPENCODE_BOOT_SETTLE_S:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.TIMEOUT,
                detail=(
                    f"pane idle {idle_for:.0f}s before {scenario.min_dispatches} "
                    f"{_SEARCH_TOOL_SUBSTR} dispatch(es)"
                ),
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        time.sleep(_POLL_INTERVAL_S)
    return ScenarioResult(
        name=scenario.name,
        status=ScenarioStatus.TIMEOUT,
        detail=(
            f"under {scenario.min_dispatches} {_SEARCH_TOOL_SUBSTR} dispatch(es) "
            f"within {scenario.timeout_s:.0f}s"
        ),
        pane_excerpt=last_pane[-_PANE_EXCERPT_TAIL:],
        elapsed_s=scenario.timeout_s,
    )


_ANSWER_SETTLE_TIMEOUT_S = 240.0
_ANSWER_SETTLE_QUIET_POLLS = 3
_ANSWER_SETTLE_INTERVAL_S = 4.0


def wait_for_answer_settle(session: str, workspace: Path) -> None:
    """Wait for opencode to finish rendering the post-tool answer before capture.

    ``run_scenario`` returns the instant the dispatch gate passes, but opencode
    is still streaming the answer turn then, so a pane captured immediately
    shows the tool call and no answer. The event tap ends the wait exactly when
    opencode reports the turn done (``session.idle`` as the latest event); the
    pane-quiet poll remains as the no-tap fallback, capped by a timeout so a
    model that streams forever cannot hang the sweep.
    """
    deadline = time.monotonic() + _ANSWER_SETTLE_TIMEOUT_S
    idles_at_entry = count_session_idles(read_events(workspace))
    prev_pane = tmux_capture(session)
    prev_event_count = -1
    quiet = 0
    while time.monotonic() < deadline:
        time.sleep(_ANSWER_SETTLE_INTERVAL_S)
        events = read_events(workspace)
        if events:
            # A fresh session.idle means opencode finished the turn. Trailing
            # bookkeeping events (session.status etc.) can follow it, so also
            # treat a quiet event stream as settled rather than requiring idle
            # to be the literal last record.
            if count_session_idles(events) > idles_at_entry:
                return
            if len(events) == prev_event_count:
                quiet += 1
                if quiet >= _ANSWER_SETTLE_QUIET_POLLS:
                    return
            else:
                quiet = 0
                prev_event_count = len(events)
            continue
        cur = tmux_capture(session)
        if cur == prev_pane:
            quiet += 1
            if quiet >= _ANSWER_SETTLE_QUIET_POLLS:
                return
        else:
            quiet = 0
            prev_pane = cur
