"""Per-device VRAM snapshot: empirical proof the fleet split a model across GPUs.

The matrix verdict proves a model tool-calls through opencode; it says nothing
about *where* the weights landed. For the multi-GPU claim we need direct
evidence, and a giant gives it for free: a 130 GB model cannot fit on one 80 GB
card, so if it loaded and served at all, the fleet must have tensor-split it.
Capturing ``nvidia-smi`` per-device memory while that model is resident shows the
split concretely: two-plus cards each holding tens of GB.

Run while a model is loaded (e.g. right after the reel's warm serve reports
ready, or before a matrix cell tears down). Writes ``gpu.json`` and asserts that
at least ``--min-devices`` cards each hold at least ``--min-used-mb``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_QUERY_FIELDS = "index,memory.used,memory.total,utilization.gpu"


@dataclass(frozen=True)
class DeviceState:
    index: int
    used_mb: int
    total_mb: int
    util_pct: int


@dataclass(frozen=True)
class GpuSnapshot:
    label: str
    devices: list[DeviceState]
    min_devices: int
    min_used_mb: int

    @property
    def loaded_devices(self) -> list[DeviceState]:
        return [d for d in self.devices if d.used_mb >= self.min_used_mb]

    @property
    def spans_multi_gpu(self) -> bool:
        return len(self.loaded_devices) >= self.min_devices

    @property
    def total_used_mb(self) -> int:
        return sum(d.used_mb for d in self.devices)


def parse_nvidia_smi(csv_text: str) -> list[DeviceState]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` rows into device states."""
    devices: list[DeviceState] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            devices.append(
                DeviceState(
                    index=int(parts[0]),
                    used_mb=int(float(parts[1])),
                    total_mb=int(float(parts[2])),
                    util_pct=int(float(parts[3])),
                )
            )
        except ValueError:
            continue
    return devices


def snapshot(label: str, min_devices: int, min_used_mb: int) -> GpuSnapshot:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    devices = parse_nvidia_smi(out.stdout)
    return GpuSnapshot(
        label=label, devices=devices, min_devices=min_devices, min_used_mb=min_used_mb
    )


def render(snap: GpuSnapshot) -> str:
    lines = [
        f"# Multi-GPU snapshot: {snap.label}",
        "",
        f"Weights span {len(snap.loaded_devices)} card(s) "
        f"(need >={snap.min_devices} each holding >={snap.min_used_mb} MB). "
        f"Verdict: {'PASS' if snap.spans_multi_gpu else 'FAIL'}",
        f"Total VRAM in use: {snap.total_used_mb} MB",
        "",
        "| GPU | Used MB | Total MB | Util % |",
        "|-----|---------|----------|--------|",
    ]
    for d in snap.devices:
        lines.append(f"| {d.index} | {d.used_mb} | {d.total_mb} | {d.util_pct} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="model", help="model/family label")
    parser.add_argument(
        "--min-devices",
        type=int,
        default=2,
        help="cards that must each hold weights for the multi-GPU claim",
    )
    parser.add_argument(
        "--min-used-mb",
        type=int,
        default=2000,
        help="per-card VRAM that counts as 'holding weights' (excludes idle/ctx)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="where to write gpu.json/md"
    )
    args = parser.parse_args()

    snap = snapshot(args.label, args.min_devices, args.min_used_mb)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gpu.json").write_text(json.dumps(asdict(snap), indent=2))
    (args.out_dir / "gpu.md").write_text(render(snap))
    print(render(snap))
    if not snap.devices:
        print("[gpu_snapshot] no GPUs visible to nvidia-smi", file=sys.stderr)
        return 1
    if not snap.spans_multi_gpu:
        print(
            f"[gpu_snapshot] {args.label}: weights on {len(snap.loaded_devices)} card(s), "
            f"need {args.min_devices}",
            file=sys.stderr,
        )
        return 1
    print(f"[gpu_snapshot] {args.label}: PASS, spans {len(snap.loaded_devices)} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
