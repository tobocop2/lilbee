"""Real-hardware smoke test for the multi-GPU llama-server fleet (run on a multi-GPU host).

Validates the parts that can only be proven on iron: device enumeration matches the
driver, the planner pins each role to the intended GPUs (a giant tensor-splits across
cards), concurrent chat+embed stay correct under load, a killed upstream is respawned by
llama-swap, and shutdown strands no GPU processes or VRAM. Unit tests mock all of this;
this script does not.

Usage (on the pod, after `lilbee model pull <chat-gguf>` and `<embedding-gguf>` and
pointing config at them -- chat_model / embedding_model in config.toml, plus optional
embed_replicas):
    python tools/qa/multi_gpu_smoke.py --concurrency 16

Exits non-zero on the first failed check. See tools/qa/MULTI_GPU_QA.md.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

from lilbee.providers.base import LLMProvider
from lilbee.providers.roles import WorkerRole

_CHAT_PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
_SMI_QUERY = "index,memory.total,memory.free"


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def nvidia_smi_free_mib() -> dict[int, int]:
    """``{gpu_index: free_MiB}`` from nvidia-smi, or ``{}`` if it isn't available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            free[int(parts[0])] = int(parts[2])
    return free


def _upstream_pids() -> list[int]:
    """PIDs of the llama-server upstreams llama-swap spawned (empty if none)."""
    out = subprocess.run(
        ["pgrep", "-f", "llama-server"], capture_output=True, text=True, check=False
    ).stdout
    return [int(p) for p in out.split() if p.isdigit()]


def _check_enumeration() -> None:
    from lilbee.providers.fleet.binary import resolve_llama_server
    from lilbee.providers.fleet.devices import probe_devices

    binary = resolve_llama_server()
    print(f"[1] binary: {binary}")
    devices = probe_devices(binary)
    print(f"[1] llama-server --list-devices: {len(devices)} device(s)")
    for dev in devices:
        print(f"      {dev.backend}{dev.index}: {dev.name} free={dev.free_bytes // 1024**2} MiB")
    _require(len(devices) >= 1, "no GPUs enumerated by the binary")
    smi = nvidia_smi_free_mib()
    if smi:
        _require(
            len(devices) == len(smi),
            f"device count mismatch: binary={len(devices)} nvidia-smi={len(smi)}",
        )
        print(f"[1] nvidia-smi agrees on {len(smi)} device(s)")


def _check_placement() -> None:
    """Plan the fleet and print each server's GPU pinning + tensor-split.

    The planner is the source of placement truth; ``nvidia-smi`` after the fleet loads
    (the concurrency check below) is where you confirm the giant actually landed on the
    cards the plan named and that an unequal split is by capacity.
    """
    from lilbee.providers.fleet.planning import plan_all_launches

    launches = plan_all_launches()
    _require(bool(launches), "planner produced no servers (every role unplaceable)")
    roles = {launch.role for launch in launches}
    print(f"[2] planned {len(launches)} server(s) across roles {sorted(r.value for r in roles)}")
    for launch in launches:
        env = launch.env_overrides
        pinned = env.get("CUDA_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES") or "(host)"
        n_gpu = len(pinned.split(",")) if pinned != "(host)" else 0
        split = " (tensor-split)" if n_gpu > 1 else ""
        print(f"      {launch.model_id}: GPU(s)={pinned}{split}")
    _require(
        WorkerRole.CHAT in roles,
        "chat role was not placed -- it is the model under test for tensor-split",
    )


def _build_provider() -> LLMProvider:
    """Reset services and warm the fleet with one chat call (loads every role's server).

    Relies on the configured ``chat_model`` / ``embedding_model`` (native GGUF refs route
    to the local llama-server fleet under the default ``auto`` provider). The first chat
    builds the llama-swap config and loads the upstreams.
    """
    from lilbee.app.services import get_services, reset_services

    reset_services()
    provider = get_services().provider
    print(f"[3] provider: {type(provider).__name__}")
    provider.chat(_CHAT_PROMPT)  # first call builds + loads the fleet
    return provider


def _check_concurrency(provider: LLMProvider, n: int) -> None:
    errors: list[str] = []
    barrier = threading.Barrier(n)

    def _hit(i: int) -> None:
        barrier.wait()
        try:
            if i % 2 == 0:
                text = provider.chat(_CHAT_PROMPT)
                _require(bool(text), "empty chat response")
            else:
                vec = provider.embed(["concurrency probe"])
                _require(bool(vec and vec[0]), "empty embedding")
        except Exception as exc:
            errors.append(f"req {i}: {exc!r}")

    threads = [threading.Thread(target=_hit, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _require(not errors, f"{len(errors)} concurrent request(s) failed: {errors[:3]}")
    print(f"[4] {n} concurrent chat+embed requests all succeeded")


def _check_restart(provider: LLMProvider) -> None:
    """Kill an upstream llama-server; llama-swap must respawn it and keep serving.

    The fleet no longer manages per-server processes itself -- llama-swap owns the
    upstream lifecycle, so a crashed upstream is restarted on the next request to it.
    """
    before = _upstream_pids()
    _require(bool(before), "no upstream llama-server processes after fleet build")
    victim = before[0]
    os.kill(victim, signal.SIGKILL)
    time.sleep(2.0)
    # Drive both roles so whichever upstream was killed is the one forced to respawn.
    _require(bool(provider.chat(_CHAT_PROMPT)), "chat failed after an upstream was killed")
    _require(bool(provider.embed(["restart probe"])), "embed failed after an upstream was killed")
    _require(victim not in _upstream_pids(), f"killed upstream {victim} is still listed")
    print(f"[5] killed upstream pid {victim}; llama-swap respawned it and served requests")


def _check_no_orphans(provider: LLMProvider, baseline: dict[int, int]) -> None:
    """Assert shutdown stranded no process or VRAM, reporting EVERY class at once.

    Both ``llama-server`` and ``llama-swap`` are checked and all failures are
    collected before failing: bailing on the first orphan once hid a leaked
    llama-swap supervisor behind a leaked upstream (bb-dpp).
    """
    provider.shutdown()
    time.sleep(2.0)
    failures: list[str] = []
    for name in ("llama-server", "llama-swap"):
        survivors = subprocess.run(
            ["pgrep", "-fa", name], capture_output=True, text=True, check=False
        ).stdout.strip()
        if survivors:
            failures.append(f"orphaned {name} processes after shutdown:\n{survivors}")
    after = nvidia_smi_free_mib()
    for idx, free_after in after.items():
        free_before = baseline.get(idx, free_after)
        leaked = free_before - free_after
        if leaked >= 1024:
            failures.append(f"GPU {idx} did not release VRAM: {leaked} MiB still held")
    _require(not failures, "; ".join(failures))
    print("[6] no orphaned llama-server / llama-swap processes; VRAM returned to baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU fleet hardware smoke test")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    baseline = nvidia_smi_free_mib()
    _check_enumeration()
    _check_placement()
    provider = _build_provider()
    _check_concurrency(provider, args.concurrency)
    _check_restart(provider)
    _check_no_orphans(provider, baseline)
    print("\nPASS: multi-GPU fleet smoke test")


if __name__ == "__main__":
    main()
