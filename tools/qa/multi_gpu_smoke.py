"""Real-hardware smoke test for the multi-GPU fleet (run on a multi-GPU host).

Validates the parts that can only be proven on iron: device enumeration matches
the driver, placement lands models on the intended GPUs, concurrent chat+embed
stay correct under load, a killed server is restarted, and shutdown strands no
GPU processes or VRAM. Unit tests mock all of this; this script does not.

Usage (on the pod, after `pip install 'lilbee[multi-gpu]'` and pulling models):
    python tools/qa/multi_gpu_smoke.py --concurrency 16

Exits non-zero on the first failed check. See tools/qa/MULTI_GPU_QA.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

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


def _check_enumeration() -> None:
    from lilbee.providers.multi_gpu.binary import resolve_llama_server_binary
    from lilbee.providers.multi_gpu.devices import probe_devices

    binary = resolve_llama_server_binary()
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


def _build_provider() -> object:
    from lilbee.app.services import get_services, reset_services
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import LlmProvider
    from lilbee.providers.multi_gpu.provider import FleetProvider

    cfg.llm_provider = LlmProvider.MULTI_GPU
    reset_services()
    provider = get_services().provider
    _require(isinstance(provider, FleetProvider), f"provider is {type(provider).__name__}")
    provider.chat(_CHAT_PROMPT)  # first call builds the fleet
    fleet = provider._fleet
    print(f"[2] placement: {len(fleet._servers)} server(s)")
    for server in fleet._servers:
        print(
            f"      {server.role.value}: pid={server._proc.pid} env={server._launch.env_overrides}"
        )
    return provider


def _check_concurrency(provider: object, n: int) -> None:
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
    print(f"[3] {n} concurrent chat+embed requests all succeeded")


def _check_restart(provider: object) -> None:
    fleet = provider._fleet
    server = fleet._servers[0]
    old_pid = server._proc.pid
    server.stop()  # hard-kill its process group, as a crash would
    fleet._restart_dead()
    _require(server.is_alive(), "server not restarted after a kill")
    _require(server._proc.pid != old_pid, "restart reused the dead pid")
    _require(bool(provider.chat(_CHAT_PROMPT)), "chat failed after restart")
    print(f"[4] killed pid {old_pid}; restarted as pid {server._proc.pid} and served a request")


def _check_no_orphans(provider: object, baseline: dict[int, int]) -> None:
    provider.shutdown()
    time.sleep(2.0)
    survivors = subprocess.run(
        ["pgrep", "-fa", "llama-server"], capture_output=True, text=True, check=False
    ).stdout.strip()
    _require(not survivors, f"orphaned llama-server processes after shutdown:\n{survivors}")
    after = nvidia_smi_free_mib()
    for idx, free_after in after.items():
        free_before = baseline.get(idx, free_after)
        leaked = free_before - free_after
        _require(leaked < 1024, f"GPU {idx} did not release VRAM: {leaked} MiB still held")
    print("[5] no orphaned processes; VRAM returned to baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU fleet hardware smoke test")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    baseline = nvidia_smi_free_mib()
    _check_enumeration()
    provider = _build_provider()
    _check_concurrency(provider, args.concurrency)
    _check_restart(provider)
    _check_no_orphans(provider, baseline)
    print("\nPASS: multi-GPU fleet smoke test")


if __name__ == "__main__":
    main()
