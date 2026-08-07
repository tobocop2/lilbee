"""Run one cell on real hardware and record what happened.

The planner is invoked in a child process rather than in-process: planning warms
up a fleet, and a warm-up racing the launch under test allocates the VRAM the
launch is being measured for. Everything here returns an Observation; nothing
here decides pass or fail (see oracles.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from tools.qa.placement_matrix.cells import Cell
from tools.qa.placement_matrix.oracles import Observation

_GB = 1024**3
_PORT = 8412
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_LOAD_TIMEOUT_S = 1800
_ROUNDS = 3
_GENERATION_TOKENS = 256
# Measured against Llama 3.3 70B: the filler phrase runs about this many tokens
# per repeat, which is only used to size a prompt near the window.
_TOKENS_PER_FILLER = 10.1
_FILLER = "The quick brown fox jumps over the lazy dog. "

_PLAN_SCRIPT = """
import json, sys
from lilbee.core.config import cfg
cfg.chat_model = sys.argv[1]
cfg.usable_vram_fraction = float(sys.argv[2])
cfg.chat_n_ctx_target = int(sys.argv[3])
if sys.argv[4] == "none":
    cfg.embedding_model = "__absent__/__absent__.gguf"
if sys.argv[5] != "none":
    cfg.num_ctx = int(sys.argv[5])
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
plan = planning.plan_all_launches()
chat = next((l for l in plan.launches if l.role.value == "chat"), None)
devices = planning.resolve_devices(planning.resolve_llama_server())
out = {
    "refusal": next((v for k, v in plan.skipped_unusable_ctx.items() if k.value == "chat"), None),
    "min_usable_ctx": engine_params.min_usable_chat_ctx(),
    "total_free_bytes": sum(d.free_bytes for d in devices),
    "device_count": len(devices),
}
if chat is not None:
    out.update(
        planned=True, ctx=chat.ctx, slots=chat.slots, argv=list(chat.argv),
        env=chat.env_overrides, est_by_device=dict(chat.est_vram_by_device),
        weights_bytes=chat.weights_bytes,
    )
else:
    out["planned"] = False
print("<<<PLAN>>>" + json.dumps(out))
"""


def _plan(cell: Cell, env: dict[str, str], *, force_ctx: int | None = None) -> dict[str, object]:
    """The planner's decision for this cell, from a clean child process."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PLAN_SCRIPT,
            cell.model.ref,
            str(cell.usable_fraction),
            str(cell.ctx_target),
            "embed" if cell.with_embed else "none",
            "none" if force_ctx is None else str(force_ctx),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
        check=False,
    )
    marker = result.stdout.rfind("<<<PLAN>>>")
    if marker < 0:
        raise RuntimeError(
            f"planner produced no decision: {result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    return json.loads(result.stdout[marker + len("<<<PLAN>>>") :])


def _smi(field: str) -> list[str]:
    """One nvidia-smi per-GPU column, or an empty list where there is no driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def gpu_count() -> int:
    """How many GPUs this host exposes."""
    return len(_smi("index"))


def _smi_used_bytes() -> list[int]:
    return [int(v) * 1024 * 1024 for v in _smi("memory.used") if v.isdigit()]


def _wait_for_idle_vram(limit_bytes: int = 512 * 1024 * 1024, timeout_s: int = 180) -> None:
    """Block until nothing else holds VRAM, so a load is measured alone."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _smi_used_bytes()
        if not used or max(used) < limit_bytes:
            return
        time.sleep(3)


def _serve_and_measure(
    argv: list[str], env: dict[str, str], ctx: int, log_path: Path
) -> tuple[bool, bool, dict[str, int]]:
    """Load the launch, fill its window repeatedly, and read back per-device use.

    Returns ``(loaded, sustained, actual_by_device)``.
    """
    from lilbee.providers.fleet.readback import parse_device_buffers

    _wait_for_idle_vram()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [*argv, "--port", str(_PORT)], stdout=log, stderr=subprocess.STDOUT, env=env
        )
        try:
            loaded = _await_health(proc)
            if not loaded:
                return False, False, {}
            time.sleep(5)
            actual = {
                label: size
                for label, size in parse_device_buffers(
                    log_path.read_text(errors="replace")
                ).items()
                if not label.upper().startswith(("CPU", "HOST"))
            }
            return True, _sustains_full_window(ctx), actual
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()


def _await_health(proc: subprocess.Popen[bytes]) -> bool:
    deadline = time.time() + _LOAD_TIMEOUT_S
    while time.time() < deadline and proc.poll() is None:
        try:
            if httpx.get(f"{_BASE_URL}/health", timeout=5).status_code == httpx.codes.OK:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def _sustains_full_window(ctx: int) -> bool:
    """Consecutive requests that fill the served window, which is where a
    load-time fit that was too generous shows up."""
    gen = min(_GENERATION_TOKENS, max(64, ctx // 4))
    repeats = max(1, int((ctx - gen - 80) / _TOKENS_PER_FILLER))
    body = {
        "messages": [{"role": "user", "content": _FILLER * repeats + "\n\nSummarize."}],
        "max_tokens": gen,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    for _ in range(_ROUNDS):
        try:
            response = httpx.post(f"{_BASE_URL}/v1/chat/completions", json=body, timeout=1800)
            if response.status_code != httpx.codes.OK:
                return False
            response.json()
        except (httpx.HTTPError, ValueError):
            return False
    return True


def run_cell(cell: Cell, *, workdir: Path, available_cards: int) -> Observation:
    """Plan, launch and measure one cell, or record why it was skipped."""
    base = Observation(
        cell_id=cell.id,
        model_key=cell.model.key,
        cards=cell.cards,
        total_free_bytes=0,
        weights_bytes=0,
        planned=False,
    )
    if cell.cards > available_cards:
        return _replace(base, skipped=f"needs {cell.cards} cards, host has {available_cards}")
    if any(cell.ballast_gib):
        return _replace(base, skipped="ballast tenants are not supported on this host")

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(cell.cards)),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    decision = _plan(cell, env)
    argv: tuple[str, ...] = tuple(decision.get("argv") or ())
    observed = _replace(
        base,
        total_free_bytes=int(decision.get("total_free_bytes", 0)),  # type: ignore[arg-type]
        weights_bytes=int(decision.get("weights_bytes", 0)),  # type: ignore[arg-type]
        planned=bool(decision.get("planned")),
        refusal=decision.get("refusal"),
        ctx=int(decision.get("ctx", 0)),  # type: ignore[arg-type]
        slots=int(decision.get("slots", 0)),  # type: ignore[arg-type]
        argv=argv,
        est_by_device=dict(decision.get("est_by_device") or {}),
        min_usable_ctx=int(decision.get("min_usable_ctx", 0)),  # type: ignore[arg-type]
        # Only the tight placement leaves a multi-card group without a ratio.
        tight=cell.cards > 1 and "--tensor-split" not in argv,
    )
    launch_env = {**env, **dict(decision.get("env") or {})}
    if observed.planned:
        loaded, sustained, actual = _serve_and_measure(
            list(observed.argv), launch_env, observed.ctx, workdir / f"{cell.id}.engine.log"
        )
        return _replace(observed, loaded=loaded, sustained=sustained, actual_by_device=actual)

    # Refused: the claim is that no usable window exists. Force one and find out.
    forced = _plan(cell, env, force_ctx=observed.min_usable_ctx)
    if not forced.get("planned"):
        return _replace(observed, forced_loaded=False, forced_sustained=False)
    loaded, sustained, _actual = _serve_and_measure(
        list(forced["argv"]),  # type: ignore[arg-type]
        {**env, **dict(forced.get("env", {}))},  # type: ignore[arg-type]
        int(forced["ctx"]),  # type: ignore[arg-type]
        workdir / f"{cell.id}.forced.log",
    )
    return _replace(observed, forced_loaded=loaded, forced_sustained=sustained)


def _replace(observation: Observation, **changes: object) -> Observation:
    from dataclasses import replace

    return replace(observation, **changes)  # type: ignore[arg-type]
