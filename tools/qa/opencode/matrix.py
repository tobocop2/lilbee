"""Local-only QA matrix that drives a real opencode binary against lilbee.

Run before tagging. Not for CI. ``--families <list>`` narrows the matrix;
``--no-pull`` skips the model download step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from harness_config import (
    _GODOT_CORPUS,
    _TMUX_SESSION_PREFIX,
    LOG_DIR,
    QA_DIR,
    RESULTS_DIR,
    ScenarioStatus,
)
from models import (
    ModelCell,
    _run_pull_with_group_kill,
    cleanup_cell_model,
    ensure_embedding_model_pulled,
    is_ref_registered,
    load_models,
    prewarm_model_blobs,
    restore_suspended_manifests,
)
from opencode_driver import (
    launch_opencode_in_tmux,
    reset_opencode_session_state,
    scope_opencode_tools,
    tmux_capture,
    tmux_kill,
)
from report import CellResult, render_report
from scenarios import (
    ScenarioResult,
    downgrade_if_ungrounded,
    run_multi_turn_scenario,
    run_scenario,
    scenario_for_tier,
    wait_for_answer_settle,
)
from serve import _count_ok_chat_completions, _scrape_serve_errors, stop_serve
from workspace import free_port, write_per_cell_workspace


def setup_cell(
    cell: ModelCell, args: argparse.Namespace, log_path: Path
) -> tuple[Path, int, subprocess.Popen[bytes] | None]:
    """Pull, seed, index, pin opencode default.

    ``lilbee launch opencode`` spawns its own lilbee serve internally; we used
    to also spawn one here for a /api/health smoke-check, but two parallel
    serves racing for the same workspace ``server.json`` / ``server.port``
    state caused opencode to land on a stale port whose lilbee serve had a
    different view of installed models, surfacing as 404 model_not_found at
    chat time. Letting the launcher own the serve lifecycle eliminates that
    race; the per-cell log still captures everything via the launcher's stdio.
    """
    reset_opencode_session_state()
    port = free_port()
    if not args.no_pull:
        # Exact file ref: a repo-level pull may install a different quant than
        # models.toml names, and a matrix that tests a different artifact than
        # it reports is not reproducible.
        print(f"[{cell.family}] pulling {cell.ref}")
        _run_pull_with_group_kill(cell.ref)
    if not is_ref_registered(cell.ref):
        # Without a registered model the launcher serves zero models and the
        # client silently falls back to its own provider; fail the cell now.
        raise RuntimeError(f"{cell.ref} is not registered after pull")
    print(f"[{cell.family}] prewarming model blobs into page cache")
    prewarm_model_blobs(cell.ref)
    workspace = write_per_cell_workspace(cell.family, cell.ref)
    print(f"[{cell.family}] seeded Godot corpus from {_GODOT_CORPUS}")
    print(f"[{cell.family}] scoping opencode tools to lilbee_search")
    scope_opencode_tools(workspace)
    return workspace, port, None


def run_smoke_scenarios(
    family: str, tier: str, session: str, workspace: Path
) -> list[ScenarioResult]:
    scen = scenario_for_tier(tier)
    print(f"[{family}] running {scen.name}: {scen.prompt[:60]}...")
    single = run_scenario(session, scen, workspace)
    # Settle the single-call answer before grading it (and before the multi-turn
    # sequence reuses the session): the verdict trips at dispatch + completion
    # while the answer is still streaming, so answer-grounding can only be judged
    # once it has rendered on the pane.
    wait_for_answer_settle(session, workspace)
    single = downgrade_if_ungrounded(single, tmux_capture(session))
    print(f"[{family}]   -> {single.status.value} ({single.elapsed_s:.1f}s) {single.detail}")
    print(f"[{family}] running {tier} multi-turn tool sequence...")
    multi = run_multi_turn_scenario(session, tier, workspace)
    print(
        f"[{family}]   multi-turn -> {multi.status.value} ({multi.elapsed_s:.1f}s) {multi.detail}"
    )
    return [single, multi]


def teardown_cell(session: str, serve_proc: subprocess.Popen[bytes] | None, keep: bool) -> None:
    """Kill the cell's tmux + opencode + lilbee-launch-spawned serve cleanly.

    The serve was spawned by ``lilbee launch opencode`` inside the tmux
    session, so killing the tmux drops its entire process group. The
    explicit ``stop_serve`` arm only fires if a separately-tracked
    ``Popen`` handle was passed (legacy boot_serve path).
    """
    if not keep:
        tmux_kill(session)
    if serve_proc is not None:
        stop_serve(serve_proc)
    # Reap any orphan ``lilbee serve`` AND its ``llama-server`` model children from
    # a previous cell's launch-opencode: a leaked llama-server holds the model's
    # RAM, which on a memory-constrained host starves (and would refuse) the next
    # cell's load.
    subprocess.run(["pkill", "-f", "lilbee serve"], check=False)
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    # llama-swap outlives its llama-server children and a stale proxy can
    # respawn the prior cell's model on any stray request to its port.
    subprocess.run(["pkill", "-f", "llama-swap"], check=False)


def run_cell(cell: ModelCell, args: argparse.Namespace) -> CellResult:
    """Orchestrate one matrix cell: setup, scenarios, teardown, cleanup."""
    result = CellResult(family=cell.family, ref=cell.ref)
    session = f"{_TMUX_SESSION_PREFIX}-{cell.family}"
    log_path = LOG_DIR / f"{cell.family}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[{cell.family}] starting ({cell.ref})")
    workspace: Path | None = None
    try:
        workspace, _port, serve_proc = setup_cell(cell, args, log_path)
        try:
            print(f"[{cell.family}] launching opencode in tmux session {session}")
            launch_opencode_in_tmux(workspace, session)
            result.scenarios = run_smoke_scenarios(cell.family, cell.tier, session, workspace)
            if any(s.status == ScenarioStatus.PASS for s in result.scenarios):
                print(f"[{cell.family}] tool call seen; waiting for answer to finish rendering")
                wait_for_answer_settle(session, workspace)
            pane = tmux_capture(session)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / f"{cell.family}.pane.txt").write_text(pane, encoding="utf-8")
            print(
                f"[{cell.family}] full pane -> results/{cell.family}.pane.txt ({len(pane)} chars)"
            )
        finally:
            keep = args.keep_on_fail and not result.passed
            teardown_cell(session, serve_proc, keep)
    except Exception as exc:
        result.setup_error = f"{type(exc).__name__}: {exc}"
        print(f"[{cell.family}] setup error: {result.setup_error}")
        log_path.write_text(f"setup_error: {result.setup_error}\n")
    if workspace is not None:
        result.serve_errors = _scrape_serve_errors(workspace)
        result.chat_completions_ok = _count_ok_chat_completions(workspace)
        print(f"[{cell.family}] {result.chat_completions_ok} successful chat completion(s)")
        if result.serve_errors:
            print(f"[{cell.family}] serve errors detected, downgrading to FAIL")
    if not args.keep_models:
        print(f"[{cell.family}] cleaning up chat GGUF")
        cleanup_cell_model(cell)
    return result


def arm_pod_watchdog() -> subprocess.Popen[bytes] | None:
    """On a pod, stop it when the run stalls (no log/model writes, idle GPUs).

    A hung or dead run leaves the pod billing for nothing; the watchdog powers
    it off instead of letting it idle. No-op off-pod (RUNPOD_POD_ID unset).
    """
    if not os.environ.get("RUNPOD_POD_ID"):
        return None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    watch_paths = [str(LOG_DIR)]
    for env_var in ("LILBEE_MODELS_DIR", "HF_HOME"):
        path = os.environ.get(env_var)
        if path:
            watch_paths.append(path)
    print(f"arming pod idle watchdog on {', '.join(watch_paths)}")
    return subprocess.Popen(["bash", str(QA_DIR / "pod_watchdog.sh"), *watch_paths])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", help="comma-separated family allowlist")
    parser.add_argument("--no-pull", action="store_true", help="skip lilbee model pull")
    parser.add_argument(
        "--keep-on-fail",
        action="store_true",
        default=False,
        help="leave tmux session up when a cell fails (default: False; opt-in for post-mortem)",
    )
    parser.add_argument(
        "--keep-models",
        action="store_true",
        default=False,
        help="keep the cell's chat GGUF on disk after each cell (default: delete)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_models(QA_DIR / "models.toml")
    if args.families:
        wanted = {f.strip() for f in args.families.split(",")}
        cells = [c for c in cells if c.family in wanted]
    cells = [c for c in cells if not c.skip]

    if not cells:
        print("no matching models in models.toml", file=sys.stderr)
        return 2

    print(f"running QA matrix for {len(cells)} model(s)")
    healed = restore_suspended_manifests()
    if healed:
        print(f"restored {healed} manifest(s) a crashed earlier run left suspended")
    watchdog = arm_pod_watchdog()
    try:
        if not args.no_pull:
            ensure_embedding_model_pulled()
        results = [run_cell(c, args) for c in cells]
    finally:
        if watchdog is not None:
            watchdog.terminate()
    (RESULTS_DIR / "results.md").write_text(render_report(results))
    print(f"\nreport: {RESULTS_DIR / 'results.md'}")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"FAIL: {len(failed)}/{len(results)} cell(s) failed")
        return 1
    print(f"PASS: all {len(results)} cell(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
