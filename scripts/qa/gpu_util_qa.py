#!/usr/bin/env python3
"""Live GPU-utilization QA harness.

Verifies the per-vendor live-util backends against real hardware. It drives the
exact path the fleet panel uses -- probe the engine's devices, then
``probe_gpu_stats`` -- so a green run here means the panel reads live util on this
box. It also captures the raw vendor-tool JSON so it can be saved as a fixture,
and offers a watch mode to confirm the bar moves under load.

Why this exists: every backend's unit test mocks an *assumed* tool format, so a
passing suite never proves the assumption matches the real tool. That is how the
Apple backend shipped reading a stuck 0%. Only real hardware settles it, and this
harness is how you settle it in ~10 minutes on a rented box.

Usage on the box::

    # one-shot: probe + parse + PASS/FAIL, and print the raw tool JSON
    uv run python scripts/qa/gpu_util_qa.py

    # capture the raw JSON to turn into a test fixture
    uv run python scripts/qa/gpu_util_qa.py --capture /tmp/amd_metric.json

    # watch util for 30s while a job runs, to confirm it tracks load
    uv run python scripts/qa/gpu_util_qa.py --watch 30

    # skip device probe (no engine binary); force a backend + indices
    uv run python scripts/qa/gpu_util_qa.py --backend SYCL --indices 0,1

PASS when at least one GPU reports an integer utilization in [0, 100].
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.gpu_backends.base import run_smi
from lilbee.providers.fleet.gpu_stats import probe_gpu_stats

# The exact command each backend runs, for raw capture. Mirrors the backend
# modules (nvidia._QUERY, amd._AMD_SMI_ARGS, intel._xpu_smi_output); keep in step
# with them. {i} is filled per device index.
_RAW_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "CUDA": (
        "nvidia-smi",
        (
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ),
    ),
    "ROCm": ("amd-smi", ("metric", "--usage", "--temperature", "--json")),
    "HIP": ("amd-smi", ("metric", "--usage", "--temperature", "--json")),
    "SYCL": ("xpu-smi", ("stats", "-d", "{i}", "-j")),
}


def _probe_real_devices() -> list[FleetDevice]:
    """Probe the engine's devices, or [] when no engine binary resolves."""
    try:
        from lilbee.providers.fleet.binary import resolve_llama_server
        from lilbee.providers.fleet.devices import probe_devices
    except ImportError:
        return []
    try:
        return list(probe_devices(resolve_llama_server()))
    except Exception as exc:
        print(f"  device probe unavailable ({exc}); use --backend/--indices")
        return []


def _forced_devices(backend: str, indices: list[int]) -> list[FleetDevice]:
    """Synthesize FleetDevices for a forced backend + indices (structural VRAM 0)."""
    return [FleetDevice(backend, i, f"{backend} device {i}", 0, 0) for i in indices]


def _capture_raw(backend: str, indices: list[int]) -> str:
    """Run the vendor tool exactly as the backend does; return concatenated stdout."""
    spec = _RAW_COMMANDS.get(backend)
    if spec is None:
        return ""
    tool, arg_template = spec
    if "{i}" in arg_template:
        chunks = []
        for i in indices:
            args = [a.format(i=i) for a in arg_template]
            chunks.append(f"# {tool} {' '.join(args)}\n{run_smi(tool, args)}")
        return "\n".join(chunks)
    return run_smi(tool, list(arg_template))


def _print_stats(devices: list[FleetDevice]) -> dict[int, int | None]:
    """Print per-GPU stats via the real orchestrator; return index -> util."""
    stats = probe_gpu_stats(devices)
    utils: dict[int, int | None] = {}
    for index in sorted(stats):
        s = stats[index]
        utils[index] = s.utilization_pct
        util = "--" if s.utilization_pct is None else f"{s.utilization_pct}%"
        temp = "--" if s.temperature_c is None else f"{s.temperature_c}C"
        gib = 1024**3
        vram = (
            "structural"
            if s.total_bytes == 0
            else f"{s.free_bytes / gib:.1f}/{s.total_bytes / gib:.1f} GiB free"
        )
        print(f"  gpu {index}: util={util:>5}  temp={temp:>5}  vram={vram}")
    return utils


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", help="Force a backend (CUDA/ROCm/HIP/SYCL/Metal/MTL)")
    ap.add_argument("--indices", help="Comma-separated device indices for --backend")
    ap.add_argument("--capture", type=Path, help="Write raw vendor-tool JSON to this path")
    ap.add_argument("--watch", type=int, default=0, help="Re-probe every second for N seconds")
    args = ap.parse_args()

    if args.backend:
        indices = [int(x) for x in (args.indices or "0").split(",")]
        devices = _forced_devices(args.backend, indices)
    else:
        devices = _probe_real_devices()

    if not devices:
        print("FAIL: no GPU devices to probe (pass --backend and --indices to force)")
        return 1

    backend = devices[0].backend
    indices = [d.index for d in devices]
    print(f"backend={backend} indices={indices}")

    raw = _capture_raw(backend, indices)
    if args.capture and raw:
        args.capture.write_text(raw)
        print(f"  raw tool output -> {args.capture}")
    else:
        print("--- raw vendor-tool output ---")
        print(raw or "  (no raw output; tool absent or non-zero exit)")
        print("--- end raw ---")

    print("parsed via probe_gpu_stats (the fleet-panel path):")
    utils = _print_stats(devices)

    if args.watch:
        print(f"watching util for {args.watch}s (run a GPU job now to see it move):")
        for _ in range(args.watch):
            time.sleep(1)
            snapshot = probe_gpu_stats(devices)
            cells = []
            for i in sorted(snapshot):
                util = snapshot[i].utilization_pct
                cells.append(f"gpu{i}={'--' if util is None else f'{util}%'}")
            print("  " + "  ".join(cells))

    live = [u for u in utils.values() if isinstance(u, int) and 0 <= u <= 100]
    if not live:
        print("FAIL: no GPU reported an integer utilization in [0,100]")
        return 1
    print(f"PASS: {len(live)}/{len(utils)} GPU(s) report live utilization")
    return 0


if __name__ == "__main__":
    sys.exit(main())
