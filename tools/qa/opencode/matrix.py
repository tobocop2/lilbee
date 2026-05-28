"""Local-only QA matrix that drives a real opencode binary against lilbee.

Run before tagging. Not for CI. ``--families <list>`` narrows the matrix;
``--no-pull`` skips the model download step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
_OPENCODE_SHARE_DIR = Path.home() / ".local" / "share" / "opencode"
_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
# Pre-built Godot 4 class-reference corpus (one-time `lilbee add /root/godot/doc/classes`
# into LILBEE_DATA=<this>); each cell copies its data/ + documents/ so lilbee_search
# has the reference without re-embedding. Override with LILBEE_QA_CORPUS.
_GODOT_CORPUS = Path(os.environ.get("LILBEE_QA_CORPUS", str(Path.home() / "godot_corpus")))
# opencode's built-in tools that compete with lilbee_search; disabled per cell so
# the model uses lilbee's MCP search instead of webfetch/read/grep (search mode).
_TOOLS_OFF = (
    "webfetch",
    "read",
    "write",
    "edit",
    "patch",
    "bash",
    "glob",
    "grep",
    "list",
    "todowrite",
    "todoread",
    "task",
)
_LILBEE_PROVIDER_ID = "lilbee"

# Substrings whose appearance in the pane means the cell can't recover --
# record the scenario as FAIL immediately instead of polling to timeout.
_FAIL_FAST_MARKERS = (
    "does not support tool calls",
    "context_length_exceeded",
    "exceeds the usable budget",
    "Internal Server Error",
)
_RAW_MARKER_FORBIDDEN = (
    "<tool_call>",
    "[TOOL_CALLS]",
    "functools[",
    "Error:",
    "Traceback",
    # Model emitted the tool call as raw JSON text instead of using opencode's
    # tool-call channel. lilbee's per-family extractor did not pick the
    # payload up, so opencode renders the JSON as a chat reply. The cell
    # cannot be marked "supported" if this happens.
    '{"name": "lilbee_',
    '{"name":"lilbee_',
)
_SUSPENDED_SUFFIX = ".qa-suspended"
_CHAT_CTX_TARGET = 131072  # ~24K goes to opencode's system + tools schema; the
# AGENTS.md plan/search/verify workflow needs the rest. resolve_chat_ctx clamps this
# to min(model trained ctx, host VRAM), so small-context models are unaffected and
# giants (qwen3-coder 256K, etc.) get the window the multi-search codegen turn needs.
_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_EMBED_PULL_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF"


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


def _installed_ref_for_repo(repo: str) -> str | None:
    """Return the chat model ref actually installed under *repo*, if any.

    ``lilbee model pull <repo>`` installs the repo's default quant, which need
    not be the specific file named in models.toml (e.g. legraphista/glm-4-9b
    installs Q4_K_S, internlm2 installs fp16). Pinning opencode to the
    models.toml ref then points at an uninstalled file: ``/v1/models`` omits
    it, opencode never dispatches, and the cell logs zero chat completions.
    Resolving the installed ref after the pull keeps the cell aligned with
    whatever quant lilbee actually fetched.
    """
    for path in _list_chat_manifests():
        ref = _ref_for_manifest(path)
        if _pull_ref_for(ref) == repo:
            return ref
    return None


def _pull_ref_for(model_ref: str) -> str:
    """Return the HuggingFace repo id (owner/name) for *model_ref*.

    GGUF refs in models.toml have the shape ``owner/repo[/subdir]/file.gguf``,
    e.g. ``Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf`` (3 parts, no subdir) or
    ``unsloth/GLM-4.5-Air-GGUF/Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf``
    (4 parts, with a quant-tier subdir). ``lilbee model pull`` wants exactly
    ``owner/repo`` regardless, so keep only the first two segments.
    """
    return "/".join(model_ref.split("/")[:2])


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
class ModelCell:
    family: str
    ref: str
    size_gb: float
    skip: bool = False
    tier: str = "small"  # small | mid | giant -> picks the prompt from _TIER_PROMPTS


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
    serve_errors: str = ""
    """Worker / dispatch errors scraped from the cell's launcher-serve.log.
    Non-empty means the chat or embed worker raised an exception during the
    cell, so even an all-green substring check downgrades to FAIL.
    """
    chat_completions_ok: int = 0
    """Count of ``POST /v1/chat/completions ... 200`` lines in launcher-serve.log.
    Zero means opencode never got a successful chat back (model failed to load,
    or every turn 500'd), so the cell cannot be a real PASS regardless of pane.
    """

    @property
    def passed(self) -> bool:
        return (
            not self.setup_error
            and not self.serve_errors
            and self.chat_completions_ok >= len(self.scenarios)
            and bool(self.scenarios)
            and all(s.status is ScenarioStatus.PASS for s in self.scenarios)
        )


def load_models(path: Path) -> list[ModelCell]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return [
        ModelCell(
            family=raw["family"],
            ref=raw["ref"],
            size_gb=float(raw.get("size_gb", 0.0)),
            skip=bool(raw.get("skip", False)),
            tier=str(raw.get("tier", "small")),
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


_AGENTS_MD = """\
## Instructions

You are working with Godot 4.4. Your training data is outdated -- many classes and
methods were renamed or replaced in Godot 4.x, so you cannot rely on memory.

**You MUST use the `lilbee_search` MCP tool to look up Godot APIs.** The Godot 4.4
class reference is indexed locally in lilbee. Do NOT guess or use web search -- call
`lilbee_search` with the exact class or method name.

### Workflow

1. **Plan** -- list every Godot class and method your answer or code needs.
2. **Search** -- call `lilbee_search` for each one individually. Do not skip a class
   even if you are confident; verify every method signature.
3. **Answer / write** -- use only class names, methods, and properties exactly as
   confirmed by the search results. If something was not found, do not use it.
4. **Verify** -- before finishing, re-check every class and method you used against a
   `lilbee_search` result and fix anything that does not match.
"""


def write_per_cell_workspace(family: str, model_ref: str) -> Path:
    """Per-cell workspace seeded by copying the pre-built Godot corpus lancedb.

    Wipes any prior ``.lilbee/`` from previous cells, then copies the shared
    corpus's ``data/`` + ``documents/`` (a fast file copy, no re-embed) so
    lilbee_search has the Godot class reference. The cell's config.toml pins the
    target chat model. Requires the one-time corpus build (see README); a missing
    corpus is a hard error so cells never silently run against an empty index.
    """
    if not (_GODOT_CORPUS / "data").is_dir():
        raise RuntimeError(
            f"Godot corpus not found at {_GODOT_CORPUS}/data. Build it once: "
            f"LILBEE_DATA={_GODOT_CORPUS} uv run lilbee add <godot>/doc/classes"
        )
    workspace = WORKSPACE_DIR / family
    workspace.mkdir(parents=True, exist_ok=True)
    config_dir = workspace / ".lilbee"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    for sub in ("data", "documents"):
        src = _GODOT_CORPUS / sub
        if src.is_dir():
            shutil.copytree(src, config_dir / sub)
    (config_dir / "config.toml").write_text(
        f'chat_model = "{model_ref}"\n'
        f'embedding_model = "{_EMBED_REF}"\n'
        f"chat_n_ctx_target = {_CHAT_CTX_TARGET}\n"
    )
    # opencode reads AGENTS.md from the project dir. This is the published
    # godot-with-lilbee workflow: it tells the model its Godot knowledge is stale
    # and it MUST look every class/method up via lilbee_search before using it.
    # Without it a capable coder model just writes GDScript from memory and never
    # calls the tool, so the integration looks broken when it is only untriggered.
    (workspace / "AGENTS.md").write_text(_AGENTS_MD)
    return workspace


def scope_opencode_tools() -> None:
    """Disable opencode's built-in tools so the model uses lilbee_search.

    Models drift to opencode's built-in webfetch/read/grep over the lilbee MCP
    search unless those are turned off (search mode). The launcher merges the
    lilbee provider + MCP into this same config and preserves the tools key.
    """
    cfg: dict[str, object] = {}
    if _OPENCODE_CONFIG.exists():
        with contextlib.suppress(json.JSONDecodeError):
            cfg = json.loads(_OPENCODE_CONFIG.read_text(encoding="utf-8"))
    cfg["tools"] = {tool: False for tool in _TOOLS_OFF}
    _OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _OPENCODE_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


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
    """Run lilbee add on each fixture so lilbee_search has content.

    Sets ``LILBEE_DATA`` explicitly so the `lilbee add` subprocess writes
    into the workspace lancedb. matrix.py inherits a polluted ``LILBEE_DATA``
    pointing at the global data root via ``lilbee.core.config``'s module-import
    side effect, so without this override the fixtures index into the wrong
    lancedb and `lilbee_search` calls return empty in opencode.
    """
    import os

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["LILBEE_DATA"] = str(workspace / ".lilbee")
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
    """Boot opencode in a tmux session pinned to the per-cell lilbee data dir.

    The ``LILBEE_DATA=workspace/.lilbee`` env override is critical: matrix.py
    imports ``lilbee.core.config``, whose module-import side effect sets
    ``LILBEE_DATA`` in matrix.py's own env to the GLOBAL data root. Every
    tmux session and subprocess matrix.py spawns inherits that polluted env,
    so the launched lilbee serve would read the global config.toml (default
    chat_model=Qwen3-0.6B) instead of the workspace's. Setting the env
    explicitly per cell breaks the inheritance: the launched serve resolves
    workspace/.lilbee/config.toml, picks up chat_model=<cell.ref>, and the
    worker pool spawns with the right model. See bb-hef0.
    """
    import os

    tmux_kill(session)
    workspace_data = workspace / ".lilbee"
    env_flags = ["-e", f"LILBEE_DATA={workspace_data}"]
    # Forward QA diagnostic flags into the launched serve + its worker
    # subprocesses (multiprocessing-spawn inherits the tmux session env), so
    # LILBEE_QA_LOG_RAW reaches the chat worker where the raw-output tap lives.
    if os.environ.get("LILBEE_QA_LOG_RAW"):
        env_flags += ["-e", "LILBEE_QA_LOG_RAW=1"]
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
            *env_flags,
            "bash",
            "-lc",
            f"cd {workspace} && exec uv run lilbee launch opencode --no-prompt",
        ],
        check=True,
    )
    time.sleep(_OPENCODE_BOOT_SETTLE_S)


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


def reset_opencode_session_state() -> None:
    """Wipe opencode's per-user state so the prior cell can't bleed into the new pane.

    Two persistence locations need scrubbing:

    1. ``~/.local/share/opencode/`` -- session DB + storage. Holds the prior
       cell's conversation transcripts. Without the wipe opencode's recent-
       sessions panel surfaces tokens from the previous PASS (e.g.
       ``KnownModelCache``) and the next cell's smoke matches them without
       its own model ever loading.

    2. ``~/.local/state/opencode/model.json`` -- the picker state. Opencode
       picks the first installed model in ``recent[]`` as its default. If
       the previous cell pinned a model that's still installed (e.g.
       ``Qwen3-8B`` left from the qwen3 cell), opencode silently falls back
       to it for the next cell whose own ref isn't pulled yet -- and the
       smoke ends up testing the WRONG model with a fake PASS.

    :func:`pin_opencode_default_model` rewrites the picker after this scrub
    so the cell's intended model wins the default.
    """
    if _OPENCODE_SHARE_DIR.exists():
        shutil.rmtree(_OPENCODE_SHARE_DIR)
    if _OPENCODE_PICKER_STATE.exists():
        _OPENCODE_PICKER_STATE.unlink()


def ensure_embedding_model_pulled() -> None:
    """Idempotent: ensure the shared embedding model is in the registry.

    The matrix's per-cell `lilbee add` step requires the workspace's
    configured embedding model to be registered, otherwise indexing skips
    every fixture with "Model not found in registry" and `lilbee_search`
    comes up empty in opencode. The registry is keyed off ``cfg.models_dir``
    (global), so one pull at matrix start serves every cell -- no per-cell
    re-pull needed.
    """
    print(f"ensuring embedding model {_EMBED_PULL_REF} is registered")
    _run_pull_with_group_kill(_EMBED_PULL_REF)


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
    reset_opencode_session_state()
    port = free_port()
    if not args.no_pull:
        pull_ref = _pull_ref_for(cell.ref)
        print(f"[{cell.family}] pulling {pull_ref}")
        _run_pull_with_group_kill(pull_ref)
        installed = _installed_ref_for_repo(pull_ref)
        if installed is not None and installed != cell.ref:
            print(
                f"[{cell.family}] pull installed {installed}; using it (models.toml had {cell.ref})"
            )
            cell.ref = installed
    workspace = write_per_cell_workspace(cell.family, cell.ref)
    print(f"[{cell.family}] seeded Godot corpus from {_GODOT_CORPUS}")
    print(f"[{cell.family}] scoping opencode tools to lilbee_search")
    scope_opencode_tools()
    print(f"[{cell.family}] pinning opencode default model")
    pin_opencode_default_model(cell.ref)
    return workspace, port, None


def run_smoke_scenarios(
    family: str, tier: str, session: str, workspace: Path
) -> list[ScenarioResult]:
    scen = scenario_for_tier(tier)
    print(f"[{family}] running {scen.name}: {scen.prompt[:60]}...")
    result = run_scenario(session, scen, workspace)
    print(f"[{family}]   -> {result.status.value} ({result.elapsed_s:.1f}s) {result.detail}")
    return [result]


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


def _scrape_serve_errors(workspace: Path) -> str:
    """Pull worker/dispatch exceptions out of the cell's launcher-serve.log.

    A cell whose scenarios pass the pane substring check but whose lilbee serve
    raised a chat-worker exception (e.g., tool-call shape mismatch, context
    overflow, ProviderError) cannot be marked "supported" -- the model
    + parser combination is broken, even if opencode happened to render
    enough text to satisfy the smoke substrings.
    """
    log_file = workspace / ".lilbee" / "data" / "logs" / "launcher-serve.log"
    if not log_file.exists():
        return ""
    text = log_file.read_text(encoding="utf-8", errors="replace")
    needles = ("Traceback (most recent call last)", "ProviderError:", "WorkerError:", "TypeError:")
    hits = [line for line in text.splitlines() if any(n in line for n in needles)]
    return "\n".join(hits[-8:]) if hits else ""


def _count_ok_chat_completions(workspace: Path) -> int:
    """Count successful ``POST /v1/chat/completions`` responses in launcher-serve.log.

    uvicorn logs each request as ``... "POST /v1/chat/completions HTTP/1.1" 200 OK``.
    A cell where opencode never received a 200 chat back (model never loaded,
    every turn 500'd) has a count of zero and cannot be a real PASS. Paired
    with :func:`_scrape_serve_errors`, this catches the SSE-200-then-crash case
    too: the header logs 200 but the mid-stream exception lands in serve_errors.
    """
    log_file = workspace / ".lilbee" / "data" / "logs" / "launcher-serve.log"
    if not log_file.exists():
        return 0
    text = log_file.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if 'POST /v1/chat/completions HTTP/1.1" 200' in line)


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
            with suspend_other_chat_manifests(cell.ref):
                print(f"[{cell.family}] launching opencode in tmux session {session}")
                launch_opencode_in_tmux(workspace, session)
                result.scenarios = run_smoke_scenarios(cell.family, cell.tier, session, workspace)
                if any(s.status == ScenarioStatus.PASS for s in result.scenarios):
                    print(f"[{cell.family}] tool call seen; waiting for answer to finish rendering")
                    wait_for_answer_settle(session)
                pane = tmux_capture(session)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / f"{cell.family}.pane.txt").write_text(pane, encoding="utf-8")
                print(f"[{cell.family}] full pane -> results/{cell.family}.pane.txt ({len(pane)} chars)")
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


def render_report(results: list[CellResult]) -> str:
    lines = [
        "# QA Matrix Results",
        "",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Family | Ref | Status | Chat 200s | Scenarios |",
        "|--------|-----|--------|-----------|-----------|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.setup_error:
            cells = f"setup: {r.setup_error}"
        else:
            cells = ", ".join(f"{s.name.split()[0]}={s.status.value}" for s in r.scenarios)
        lines.append(f"| {r.family} | `{r.ref}` | {status} | {r.chat_completions_ok} | {cells} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.family} ({r.ref})")
        lines.append("")
        if r.setup_error:
            lines.append(f"Setup error: {r.setup_error}")
        if r.serve_errors:
            lines.append("### Worker / dispatch errors in launcher-serve.log")
            lines.append("```")
            lines.append(r.serve_errors)
            lines.append("```")
        for s in r.scenarios:
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
    if not args.no_pull:
        ensure_embedding_model_pulled()
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
