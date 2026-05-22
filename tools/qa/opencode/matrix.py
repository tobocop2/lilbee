"""Local-only QA matrix that drives a real opencode binary against lilbee.

Run before tagging. Not for CI. ``--families <list>`` narrows the matrix;
``--no-pull`` skips the model download step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import httpx

QA_DIR = Path(__file__).resolve().parent
REPO_ROOT = QA_DIR.parent.parent.parent
RESULTS_DIR = QA_DIR / "results"
WORKSPACE_DIR = QA_DIR / "workspace"
LOG_DIR = QA_DIR / "logs"
SHARED_WORKSPACE = QA_DIR / "_shared_workspace"

_TMUX_SESSION_PREFIX = "lilbee-qa"
_TMUX_HISTORY_LINES = 3000
_TMUX_WINDOW_COLS = 200
_TMUX_WINDOW_ROWS = 50
_POST_SEND_SLEEP_S = 0.1

_SERVE_BOOT_TIMEOUT_S = 30.0
_SERVE_TERMINATE_TIMEOUT_S = 10.0
_OPENCODE_BOOT_SETTLE_S = 30.0  # boot + first-prompt prefill warmup
_INDEX_TIMEOUT_S = 120.0
_MODEL_PULL_TIMEOUT_S = 3600.0  # residential bandwidth, multi-quant repos can run 30+ min
_POLL_INTERVAL_S = 2.0
_SCENARIO_TIMEOUT_S = 600.0  # 8B on Apple Silicon w/ 32K KV cache + tool round-trip is slow
_MULTI_TOOL_TIMEOUT_S = 600.0
_INTER_SCENARIO_SETTLE_S = 15.0  # let opencode finish the prior turn before queuing the next
# Fail-fast: declare a scenario dead when the pane stops changing for this long
# AFTER the model has emitted at least one ``Build · …`` activity marker. Keeps
# a quiet model from eating the full 600s timeout.
_PANE_IDLE_TIMEOUT_S = 90.0

_PANE_EXCERPT_TAIL = 2000
_OPENCODE_PICKER_STATE = Path.home() / ".local" / "state" / "opencode" / "model.json"
_LILBEE_PROVIDER_ID = "lilbee"

# Substrings whose appearance in the pane means the cell can't recover --
# record the scenario as FAIL immediately instead of polling to timeout.
_FAIL_FAST_MARKERS = (
    "does not support tool calls",
    "context_length_exceeded",
    "exceeds the usable budget",
    "Internal Server Error",
)
_RAW_MARKER_FORBIDDEN = ("<tool_call>", "[TOOL_CALLS]", "functools[", "Error:", "Traceback")
_SUSPENDED_SUFFIX = ".qa-suspended"
_CHAT_CTX_TARGET = 32768  # opencode's default system prompt is ~14K tokens, plus tools schema ~10K


def _models_manifests_dir() -> Path:
    """Locate lilbee's chat-manifests directory via cfg."""
    from lilbee.core.config import cfg

    return Path(cfg.models_dir) / "manifests"


def _list_chat_manifests() -> list[Path]:
    manifests_dir = _models_manifests_dir()
    if not manifests_dir.exists():
        return []
    chats: list[Path] = []
    for path in manifests_dir.rglob("*.gguf.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("task") == "chat":
            chats.append(path)
    return chats


def _ref_for_manifest(path: Path) -> str:
    """Convert a manifest path back to its canonical HF ref."""
    repo = path.parent.name.replace("--", "/")
    filename = path.name.removesuffix(".json")
    return f"{repo}/{filename}"


def _pull_ref_for(model_ref: str) -> str:
    """Strip the trailing ``/<file>.gguf`` so ``lilbee model pull`` accepts it."""
    return model_ref.rsplit("/", 1)[0] if model_ref.endswith(".gguf") else model_ref


@contextlib.contextmanager
def suspend_other_chat_manifests(target_ref: str) -> Iterator[None]:
    """Move every chat manifest except *target_ref* out of the registry.

    The launcher's ``_update_opencode_picker_state`` prepends every entry of
    ``sorted(installed_chat_model_refs())`` to opencode's recent list, so
    whichever ref sorts first wins ``recent[0]``. To pin a specific model
    per cell we suspend the others for the duration of the cell.
    """
    suspended: list[tuple[Path, Path]] = []
    try:
        for path in _list_chat_manifests():
            if _ref_for_manifest(path) == target_ref:
                continue
            suspended_path = path.with_name(path.name + _SUSPENDED_SUFFIX)
            path.rename(suspended_path)
            suspended.append((path, suspended_path))
        yield
    finally:
        for original, suspended_path in suspended:
            if suspended_path.exists():
                suspended_path.rename(original)


class ScenarioStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class Scenario:
    """One QA cell: send *prompt*, wait for *expected* / *forbidden* in the pane."""

    name: str
    prompt: str
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    timeout_s: float


SMOKE_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="S1 single tool call",
        prompt="search the indexed docs for the chat worker file",
        expected=("lilbee_search", "chat worker"),
        forbidden=_RAW_MARKER_FORBIDDEN,
        timeout_s=_SCENARIO_TIMEOUT_S,
    ),
    Scenario(
        name="S2 multi-tool turn",
        prompt=(
            "use lilbee_search to find information about the dispatch layer, "
            "then quote the named class that resolves the model"
        ),
        expected=("lilbee_search", "KnownModelCache"),
        forbidden=_RAW_MARKER_FORBIDDEN,
        timeout_s=_MULTI_TOOL_TIMEOUT_S,
    ),
    Scenario(
        name="S3 streaming visible",
        prompt=(
            "use lilbee_search to find information about tool extraction, "
            "then describe the parsing library and schema location"
        ),
        expected=("recursive_parse", "schemas"),
        forbidden=_RAW_MARKER_FORBIDDEN,
        timeout_s=_SCENARIO_TIMEOUT_S,
    ),
)


@dataclass
class ModelCell:
    family: str
    ref: str
    size_gb: float
    skip: bool = False


@dataclass
class ScenarioResult:
    name: str
    status: ScenarioStatus
    detail: str = ""
    pane_excerpt: str = ""
    elapsed_s: float = 0.0


@dataclass
class CellResult:
    family: str
    ref: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
    setup_error: str = ""

    @property
    def passed(self) -> bool:
        return not self.setup_error and all(s.status is ScenarioStatus.PASS for s in self.scenarios)


def load_models(path: Path) -> list[ModelCell]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return [
        ModelCell(
            family=raw["family"],
            ref=raw["ref"],
            size_gb=float(raw.get("size_gb", 0.0)),
            skip=bool(raw.get("skip", False)),
        )
        for raw in data.get("model", [])
    ]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def seed_shared_workspace() -> Path:
    """Drop three markdown fixtures so lilbee_search returns real results."""
    SHARED_WORKSPACE.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "chat_worker.md": (
            "# Chat worker\n\n"
            "The chat worker subprocess in lilbee runs llama-cpp inference. "
            "It receives ChatRequest payloads over a pipe transport and "
            "streams tokens back via SSE. Cancellation is enforced through "
            "an abort flag in shared memory."
        ),
        "dispatch.md": (
            "# Dispatch layer\n\n"
            "chat_dispatch.dispatch_chat is the canonical entry point that "
            "the OpenAI-compatible route forwards to. It resolves the model "
            "through KnownModelCache, enforces tool capability, and routes "
            "to either the native llama-cpp worker or the SDK backend."
        ),
        "tool_extraction.md": (
            "# Tool extraction\n\n"
            "Lilbee uses a schema-driven response parser based on the "
            "HuggingFace transformers chat_parsing_utils.recursive_parse. "
            "Each supported family ships a JSON schema under "
            "providers/worker/response_parser/schemas/."
        ),
    }
    for filename, content in fixtures.items():
        target = SHARED_WORKSPACE / filename
        if not target.exists() or target.read_text() != content:
            target.write_text(content)
    return SHARED_WORKSPACE


def write_per_cell_workspace(family: str, model_ref: str) -> Path:
    """Per-cell workspace dir seeded from the shared fixtures."""
    workspace = WORKSPACE_DIR / family
    workspace.mkdir(parents=True, exist_ok=True)
    for src in seed_shared_workspace().iterdir():
        if src.is_file():
            shutil.copy(src, workspace / src.name)
    config_dir = workspace / ".lilbee"
    config_dir.mkdir(exist_ok=True)
    embed_ref = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
    (config_dir / "config.toml").write_text(
        f'chat_model = "{model_ref}"\n'
        f'embedding_model = "{embed_ref}"\n'
        f"chat_n_ctx_target = {_CHAT_CTX_TARGET}\n"
    )
    return workspace


def pin_opencode_default_model(model_ref: str) -> None:
    """Rewrite opencode's picker state so *model_ref* is recent[0].

    ``lilbee launch opencode`` prepends every installed model to the
    ``recent`` list. Without this step opencode's default session model
    is non-deterministic across QA runs; with it the cell loads the
    intended model.
    """
    _OPENCODE_PICKER_STATE.parent.mkdir(parents=True, exist_ok=True)
    state: dict = {"recent": [], "favorite": [], "variant": {}}
    if _OPENCODE_PICKER_STATE.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            state = json.loads(_OPENCODE_PICKER_STATE.read_text(encoding="utf-8"))
    recent = [e for e in state.get("recent", []) if isinstance(e, dict)]
    target_entry = {"providerID": _LILBEE_PROVIDER_ID, "modelID": model_ref}
    recent = [
        e
        for e in recent
        if not (e.get("modelID") == model_ref and e.get("providerID") == _LILBEE_PROVIDER_ID)
    ]
    recent.insert(0, target_entry)
    state["recent"] = recent
    _OPENCODE_PICKER_STATE.write_text(json.dumps(state, indent=2))


def index_workspace(workspace: Path, log_path: Path) -> None:
    """Run lilbee add on each fixture so lilbee_search has content."""
    import os

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    with log_path.open("a") as log:
        log.write("=== lilbee add ===\n")
        log.flush()
        for fixture in ("chat_worker.md", "dispatch.md", "tool_extraction.md"):
            subprocess.run(
                ["uv", "run", "lilbee", "add", str(workspace / fixture)],
                cwd=workspace,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=_INDEX_TIMEOUT_S,
            )


def boot_serve(workspace: Path, port: int, log_path: Path) -> subprocess.Popen[bytes]:
    """Spawn lilbee serve on *port* in its own process group and wait for /api/health.

    The process group lets :func:`stop_serve` reap the whole tree (``uv`` plus
    the lilbee child it forks) rather than orphaning the child python.
    """
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        ["uv", "run", "lilbee", "serve", "--port", str(port)],
        cwd=workspace,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + _SERVE_BOOT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"lilbee serve exited before becoming ready (rc={proc.returncode})")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1.0)
            if resp.status_code == 200:
                return proc
        except (httpx.HTTPError, httpx.RequestError):
            pass
        time.sleep(0.5)
    stop_serve(proc)
    raise TimeoutError("lilbee serve did not become ready in time")


def stop_serve(proc: subprocess.Popen[bytes]) -> None:
    """Reap the full lilbee-serve process group (uv parent + lilbee python child)."""
    import os
    import signal

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=_SERVE_TERMINATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def tmux_kill(name: str) -> None:
    if tmux_session_exists(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=False)


def tmux_capture(name: str) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", name, "-p", "-S", f"-{_TMUX_HISTORY_LINES}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def tmux_send(name: str, keys: str) -> None:
    subprocess.run(["tmux", "send-keys", "-t", name, keys], check=False)
    time.sleep(_POST_SEND_SLEEP_S)
    subprocess.run(["tmux", "send-keys", "-t", name, "Enter"], check=False)


def launch_opencode_in_tmux(workspace: Path, session: str) -> None:
    tmux_kill(session)
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            str(_TMUX_WINDOW_COLS),
            "-y",
            str(_TMUX_WINDOW_ROWS),
            "bash",
            "-lc",
            f"cd {workspace} && exec uv run lilbee launch opencode",
        ],
        check=True,
    )
    time.sleep(_OPENCODE_BOOT_SETTLE_S)


def run_scenario(session: str, scenario: Scenario) -> ScenarioResult:
    """Send the prompt and poll the pane until expected/forbidden/idle settle."""
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
        if not missing:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.PASS,
                detail="all expected substrings present",
                pane_excerpt=pane[-_PANE_EXCERPT_TAIL:],
                elapsed_s=time.time() - start,
            )
        idle_for = time.time() - last_change_at
        if idle_for > _PANE_IDLE_TIMEOUT_S and time.time() - start > _OPENCODE_BOOT_SETTLE_S:
            return ScenarioResult(
                name=scenario.name,
                status=ScenarioStatus.TIMEOUT,
                detail=f"pane idle {idle_for:.0f}s; missing {missing}",
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


def _run_pull_with_group_kill(pull_ref: str) -> None:
    """Run ``lilbee model pull`` in its own process group so a timeout reaps the
    full tree (otherwise ``uv``'s child python orphans and keeps the download
    running, contending for bandwidth with the next cell's pull).

    Progress bars are suppressed so the matrix stdout stays grep-able.
    """
    import os
    import signal

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    proc = subprocess.Popen(
        ["uv", "run", "lilbee", "model", "pull", pull_ref],
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        proc.wait(timeout=_MODEL_PULL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        raise


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
    workspace = write_per_cell_workspace(cell.family, cell.ref)
    port = free_port()
    if not args.no_pull:
        pull_ref = _pull_ref_for(cell.ref)
        print(f"[{cell.family}] pulling {pull_ref}")
        _run_pull_with_group_kill(pull_ref)
    print(f"[{cell.family}] indexing fixtures")
    index_workspace(workspace, log_path)
    print(f"[{cell.family}] pinning opencode default model")
    pin_opencode_default_model(cell.ref)
    return workspace, port, None


def run_smoke_scenarios(family: str, session: str) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for i, scen in enumerate(SMOKE_SCENARIOS):
        print(f"[{family}] running {scen.name}")
        result = run_scenario(session, scen)
        results.append(result)
        print(f"[{family}]   -> {result.status.value} ({result.elapsed_s:.1f}s) {result.detail}")
        if i < len(SMOKE_SCENARIOS) - 1:
            time.sleep(_INTER_SCENARIO_SETTLE_S)
    return results


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
    # Make sure no orphan ``lilbee serve`` from a previous cell's
    # launch-opencode survives into the next cell.
    subprocess.run(["pkill", "-f", "lilbee serve"], check=False)


def cleanup_cell_model(cell: ModelCell) -> None:
    """Delete the cell's chat GGUF from the HF cache + lilbee's manifest.

    Disk pressure on a dev laptop is the real limiter for an exhaustive
    sweep (qwen3-coder alone is 17 GB). Freeing the blob between cells
    keeps total disk usage flat at roughly the largest single model
    instead of the cumulative pull set.
    """
    from lilbee.core.config import cfg

    repo = _pull_ref_for(cell.ref)
    # HF-cache directories ("models--Qwen--Qwen3-4B-GGUF") use the ``models--``
    # prefix; lilbee's manifests dir ("Qwen--Qwen3-4B-GGUF") does NOT. Cleaning
    # only the prefixed paths left stale manifests behind, which then convinced
    # the next pull that the model was cached and convinced lilbee's registry
    # the install was complete -- both inconsistent with the missing blob,
    # which surfaced to opencode as a 404 model_not_found at chat time.
    hf_cache_dir = "models--" + repo.replace("/", "--")
    manifest_dir = repo.replace("/", "--")
    models_root = Path(cfg.models_dir)
    for target in (
        models_root / hf_cache_dir,
        models_root / ".locks" / hf_cache_dir,
        models_root / "manifests" / manifest_dir,
    ):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def run_cell(cell: ModelCell, args: argparse.Namespace) -> CellResult:
    """Orchestrate one matrix cell: setup, scenarios, teardown, cleanup."""
    result = CellResult(family=cell.family, ref=cell.ref)
    session = f"{_TMUX_SESSION_PREFIX}-{cell.family}"
    log_path = LOG_DIR / f"{cell.family}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[{cell.family}] starting ({cell.ref})")
    try:
        workspace, _port, serve_proc = setup_cell(cell, args, log_path)
        try:
            with suspend_other_chat_manifests(cell.ref):
                print(f"[{cell.family}] launching opencode in tmux session {session}")
                launch_opencode_in_tmux(workspace, session)
                result.scenarios = run_smoke_scenarios(cell.family, session)
        finally:
            keep = args.keep_on_fail and not result.passed
            teardown_cell(session, serve_proc, keep)
    except Exception as exc:
        result.setup_error = f"{type(exc).__name__}: {exc}"
        print(f"[{cell.family}] setup error: {result.setup_error}")
        log_path.write_text(f"setup_error: {result.setup_error}\n")
    if not args.keep_models:
        print(f"[{cell.family}] cleaning up chat GGUF")
        cleanup_cell_model(cell)
    return result


def render_report(results: list[CellResult]) -> str:
    lines = [
        "# QA Matrix Results",
        "",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Family | Ref | Status | Scenarios |",
        "|--------|-----|--------|-----------|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.setup_error:
            cells = f"setup: {r.setup_error}"
        else:
            cells = ", ".join(f"{s.name.split()[0]}={s.status.value}" for s in r.scenarios)
        lines.append(f"| {r.family} | `{r.ref}` | {status} | {cells} |")
    lines.append("")
    for r in results:
        if r.passed and not r.setup_error:
            continue
        lines.append(f"## {r.family} ({r.ref})")
        lines.append("")
        if r.setup_error:
            lines.append(f"Setup error: {r.setup_error}")
        for s in r.scenarios:
            if s.status is ScenarioStatus.PASS:
                continue
            lines.append(f"### {s.name} -> {s.status.value}")
            lines.append(f"Detail: {s.detail}")
            lines.append("```")
            lines.append(s.pane_excerpt or "(no pane captured)")
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


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
    results = [run_cell(c, args) for c in cells]
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
