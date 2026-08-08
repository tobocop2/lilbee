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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from tools.qa.placement_matrix.cells import Cell
from tools.qa.placement_matrix.oracles import Observation


@dataclass(frozen=True)
class PlanDecision:
    """What the planner decided for one cell, parsed rather than read loosely.

    Every field is required or explicitly defaulted here, so a key the planner
    stopped emitting raises instead of arriving as a plausible zero.
    """

    planned: bool
    total_free_bytes: int
    min_usable_ctx: int
    refusal: str | None = None
    ctx: int = 0
    slots: int = 0
    argv: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    est_by_device: dict[str, int] = field(default_factory=dict)
    weights_bytes: int = 0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanDecision:
        planned = bool(payload["planned"])
        decision = cls(
            planned=planned,
            total_free_bytes=int(payload["total_free_bytes"]),
            min_usable_ctx=int(payload["min_usable_ctx"]),
            refusal=payload["refusal"],
        )
        if not planned:
            return decision
        return replace(
            decision,
            ctx=int(payload["ctx"]),
            slots=int(payload["slots"]),
            argv=tuple(str(a) for a in payload["argv"]),
            env={str(k): str(v) for k, v in payload["env"].items()},
            est_by_device={str(k): int(v) for k, v in payload["est_by_device"].items()},
            weights_bytes=int(payload["weights_bytes"]),
        )


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
# A round has to occupy most of the window, or it is not evidence the window works.
_MIN_WINDOW_FILL = 0.8

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


def _plan(cell: Cell, env: dict[str, str], *, force_ctx: int | None = None) -> PlanDecision:
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
    return PlanDecision.from_payload(json.loads(result.stdout[marker + len("<<<PLAN>>>") :]))


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


def _wait_for_idle_vram(limit_bytes: int = 512 * 1024 * 1024, timeout_s: int = 180) -> bool:
    """Block until nothing else holds VRAM; False if it never went idle.

    Reported rather than swallowed: a load that raced another tenant produced
    memory numbers describing that tenant, and a silent proceed makes those
    numbers indistinguishable from a clean measurement.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _smi_used_bytes()
        if not used or max(used) < limit_bytes:
            return True
        time.sleep(3)
    return False


@dataclass(frozen=True)
class ServeResult:
    """What one launch did once it was up."""

    loaded: bool
    sustained: bool = False
    actual_by_device: dict[str, int] = field(default_factory=dict)
    vram_was_idle: bool = True
    generated_tokens: int = 0
    prompt_tokens: int = 0


def _serve_and_measure(
    argv: list[str], env: dict[str, str], ctx: int, log_dir: Path, model_id: str
) -> ServeResult:
    """Load the launch, fill its window repeatedly, and read back per-device use.

    The engine's log environment comes from lilbee's own ``engine_log_env``, and
    the report is read from the file it names. Spawning without it prints no
    per-device buffers at all (the default verbosity omits them), which would
    make every cell look like an engine whose log format had changed.
    """
    from lilbee.providers.fleet.readback import (
        _is_host_device,
        engine_log_env,
        engine_log_path,
        parse_device_buffers,
    )

    env = {**env, **engine_log_env(log_dir, model_id)}
    report_path = engine_log_path(log_dir, model_id)
    was_idle = _wait_for_idle_vram()
    with (log_dir / f"{model_id}.stdout.log").open("wb") as log:
        proc = subprocess.Popen(
            [*argv, "--port", str(_PORT)], stdout=log, stderr=subprocess.STDOUT, env=env
        )
        try:
            if not _await_health(proc):
                return ServeResult(loaded=False, vram_was_idle=was_idle)
            time.sleep(5)
            report = report_path.read_text(errors="replace") if report_path.exists() else ""
            # lilbee's own predicate, not a prefix guess: the engine labels
            # host-pinned memory CUDA_Host, which no "CPU"/"HOST" test catches.
            actual = {
                label: size
                for label, size in parse_device_buffers(report).items()
                if not _is_host_device(label)
            }
            sustained, generated, prompted = _sustains_full_window(ctx)
            return ServeResult(
                loaded=True,
                sustained=sustained,
                actual_by_device=actual,
                vram_was_idle=was_idle,
                generated_tokens=generated,
                prompt_tokens=prompted,
            )
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


def _sustains_full_window(ctx: int) -> tuple[bool, int, int]:
    """Consecutive requests that fill the served window.

    Returns ``(sustained, generated_tokens, prompt_tokens)``. A 200 with an empty
    completion, or a prompt the server quietly truncated, is not a served window,
    so the token counts are checked rather than the status code alone.
    """
    gen = min(_GENERATION_TOKENS, max(64, ctx // 4))
    repeats = max(1, int((ctx - gen - 80) / _TOKENS_PER_FILLER))
    body = {
        "messages": [{"role": "user", "content": _FILLER * repeats + "\n\nSummarize."}],
        "max_tokens": gen,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    generated = prompted = 0
    for _ in range(_ROUNDS):
        try:
            response = httpx.post(f"{_BASE_URL}/v1/chat/completions", json=body, timeout=1800)
            if response.status_code != httpx.codes.OK:
                return False, generated, prompted
            usage = response.json().get("usage") or {}
        except (httpx.HTTPError, ValueError):
            return False, generated, prompted
        generated = int(usage.get("completion_tokens", 0))
        prompted = int(usage.get("prompt_tokens", 0))
        # The point of the round is that the window was really occupied: a
        # truncated prompt or a one-token answer is a pass the harness must not give.
        if generated < gen or prompted + generated < ctx * _MIN_WINDOW_FILL:
            return False, generated, prompted
    return True, generated, prompted


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
        return replace(base, skipped=f"needs {cell.cards} cards, host has {available_cards}")
    if any(cell.ballast_gib):
        return replace(base, skipped="ballast tenants are not supported on this host")

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(cell.cards)),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    decision = _plan(cell, env)
    observed = replace(
        base,
        total_free_bytes=decision.total_free_bytes,
        weights_bytes=decision.weights_bytes,
        planned=decision.planned,
        refusal=decision.refusal,
        ctx=decision.ctx,
        slots=decision.slots,
        argv=decision.argv,
        est_by_device=decision.est_by_device,
        min_usable_ctx=decision.min_usable_ctx,
        # Only the tight placement leaves a multi-card group without a ratio.
        tight=cell.cards > 1 and "--tensor-split" not in decision.argv,
    )
    if decision.planned:
        served = _serve_and_measure(
            list(decision.argv), {**env, **decision.env}, decision.ctx, workdir, cell.id
        )
        return replace(
            observed,
            loaded=served.loaded,
            sustained=served.sustained,
            actual_by_device=served.actual_by_device,
            vram_was_idle=served.vram_was_idle,
            generated_tokens=served.generated_tokens,
            prompt_tokens=served.prompt_tokens,
        )

    # Refused: the claim is that no usable window exists. Force one and find out.
    forced = _plan(cell, env, force_ctx=decision.min_usable_ctx)
    if not forced.planned:
        return replace(observed, forced_planned=False)
    served = _serve_and_measure(
        list(forced.argv), {**env, **forced.env}, forced.ctx, workdir, f"{cell.id}.forced"
    )
    return replace(
        observed,
        forced_planned=True,
        forced_loaded=served.loaded,
        forced_sustained=served.sustained,
        vram_was_idle=served.vram_was_idle,
    )
