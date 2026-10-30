"""Acceptance rows for #1211: one host-capability module.

Run: PYTHONPATH=<checkout>/src python -B accept_host_caps.py [--seed N]
Prints one line per row: ROW <id> <GREEN|RED> <detail>. Exit 1 when any row is red.

Contract the rows pin (module name is the lane's choice; the rows import it):
  lilbee.providers.fleet.host_caps.collect(system, machine, mac_ver, cpu_info, nvml, npu_devices)
      system "Linux"|"Darwin"|"Windows" (platform.system()), machine (platform.machine()),
      mac_ver (platform.mac_ver()[0]), cpu_info (py-cpuinfo get_cpu_info() dict),
      nvml (None, or {"driver", "cuda_driver": int, "gpus": [{"name", "cc": [major, minor]}]}),
      npu_devices (names found in the device lists).
  returns HostCaps: os, arch, macos_version (tuple|None), avx2, neon (.present, .reason),
      gpus (.name, .compute tuple), nvidia_driver_cuda (tuple|None), nvidia_reason, npus.
  detect() -> HostCaps reads the live host.

Oracles are not the code under test: /proc/cpuinfo, sysctl, nvidia-smi and uname text recorded
beside each fixture, and for generated inputs the flag list itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
MOD = "lilbee.providers.fleet.host_caps"
ROWS: list[tuple[str, bool, str]] = []


def row(rid: str, ok: bool, detail: str = "") -> None:
    ROWS.append((rid, bool(ok), detail))
    print(f"ROW {rid} {'GREEN' if ok else 'RED'} {detail}")


def load():
    try:
        return importlib.import_module(MOD)
    except ImportError as exc:
        return exc


def version(code: int) -> tuple[int, int]:
    """NVML encodes CUDA 12.4 as 12040."""
    return code // 1000, (code % 1000) // 10


def oracle_gate(avx2: bool, cc: tuple[int, int] | None, cuda: tuple[int, int] | None,
                system: str, machine: str) -> dict[str, str]:
    """Reference rules from #1197 and #1194; independent of the module."""
    out = {"ik_llama.cpp": "in" if avx2 else "out"}
    cuda_ok = (system == "Linux" and machine == "x86_64" and cc is not None
               and cc >= (8, 0) and cuda is not None and cuda >= (12, 8))
    out["mistral.rs-cuda"] = "in" if cuda_ok else "out"
    return out


def fixtures() -> list[tuple[str, dict, dict]]:
    out = []
    for p in sorted(FIX.glob("*.probe.json")):
        name = p.name[: -len(".probe.json")]
        probe = json.loads(p.read_text())
        oracle = (FIX / f"{name}.oracle.txt").read_text().split()
        out.append((name, probe, {"raw": oracle}))
    return out


EXPECT = {  # host -> (avx2, neon, gpus, driver_cuda, os, arch)
    "mac-m1pro": (False, True, [], None, "macos", "arm64"),
    "home-xeon-1070ti": (False, False, [("NVIDIA GeForce GTX 1070 Ti", (6, 1))], (12, 4), "linux", "x86_64"),
    "gcp-epyc-nogpu": (True, False, [], None, "linux", "x86_64"),
}


def host_rows(mod) -> None:
    for name, probe, _ in fixtures():
        avx2, neon, gpus, drv, osn, arch = EXPECT[name]
        try:
            caps = mod.collect(system=probe["system"], machine=probe["machine"], mac_ver=probe["mac_ver"],
                               cpu_info={k: probe[k] for k in ("arch", "arch_string_raw", "brand_raw", "flags")},
                               nvml=probe.get("nvml"), npu_devices=[])
        except Exception as exc:  # a crash is a red row
            row(f"fixture/{name}/collect", False, repr(exc)[:80])
            continue
        row(f"fixture/{name}/avx2", caps.avx2.present is avx2 and (avx2 or bool(caps.avx2.reason)),
            f"present={caps.avx2.present} reason={caps.avx2.reason!r}")
        row(f"fixture/{name}/neon", caps.neon.present is neon and (neon or bool(caps.neon.reason)),
            f"present={caps.neon.present} reason={caps.neon.reason!r}")
        row(f"fixture/{name}/gpus", [(g.name, tuple(g.compute)) for g in caps.gpus] == gpus, str(caps.gpus))
        row(f"fixture/{name}/driver_cuda", caps.nvidia_driver_cuda == drv
            and (drv is not None or "no NVIDIA driver" in caps.nvidia_reason),
            f"{caps.nvidia_driver_cuda} reason={caps.nvidia_reason!r}")
        row(f"fixture/{name}/os_arch", (str(caps.os), str(caps.arch)) == (osn, arch), f"{caps.os} {caps.arch}")
        row(f"fixture/{name}/macos_version",
            (caps.macos_version == (14, 6, 1)) if osn == "macos" else caps.macos_version is None,
            str(caps.macos_version))
        row(f"fixture/{name}/npus", tuple(caps.npus) == (), str(caps.npus))


def oracle_rows() -> None:
    """The recorded oracle text agrees with the expected table (guards the table itself)."""
    for name, _probe, o in fixtures():
        raw = o["raw"]
        avx2, neon, gpus, drv, _osn, _arch = EXPECT[name]
        if name == "mac-m1pro":
            ok = raw[0] == "1" and raw[1] == "arm64" and neon and not avx2
        elif name == "home-xeon-1070ti":
            ok = raw[0] == "0" and raw[1] == "550.135," and raw[2] == "6.1" and "12.4" in raw[-1]
        else:
            ok = int(raw[0]) > 0 and avx2
        row(f"oracle/{name}", ok, " ".join(raw))


def generated_rows(mod, seed: int, n: int) -> None:
    rng = random.Random(seed)
    bad = []
    for i in range(n):
        system, machine = rng.choice([("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"),
                                      ("Darwin", "x86_64"), ("Windows", "AMD64"), ("Windows", "ARM64")])
        pool = ["sse2", "avx", "avx2", "fma", "bmi2", "neon", "asimd", "sve", "aes"]
        flags = sorted(rng.sample(pool, rng.randint(0, len(pool))))
        x86 = machine in ("x86_64", "AMD64")
        if not x86:
            flags = [f for f in flags if f not in ("avx", "avx2", "bmi2")] + ["asimd"]
        cc = (rng.choice([6, 7, 8, 9, 10, 12]), rng.randint(0, 9))
        cuda = rng.choice([(11, 8), (12, 4), (12, 8), (13, 0)])
        nvml = rng.choice([None, {"driver": "570.1", "cuda_driver": cuda[0] * 1000 + cuda[1] * 10,
                                  "gpus": [{"name": "GPU", "cc": list(cc)}]}])
        cpu = {"arch": "X86_64" if x86 else "ARM_8", "arch_string_raw": machine, "brand_raw": "x", "flags": flags}
        caps = mod.collect(system=system, machine=machine, mac_ver="14.6.1" if system == "Darwin" else "",
                           cpu_info=cpu, nvml=nvml, npu_devices=[])
        want_avx2 = "avx2" in flags and x86
        if caps.avx2.present is not want_avx2:
            bad.append((i, "avx2"))
        if nvml is None and (caps.gpus or caps.nvidia_driver_cuda is not None):
            bad.append((i, "nvml-none"))
        if nvml and (caps.nvidia_driver_cuda != cuda or tuple(caps.gpus[0].compute) != cc):
            bad.append((i, "nvml"))
        if not want_avx2 and not caps.avx2.reason:
            bad.append((i, "avx2-reason"))
    row(f"generated/{n}-inputs-seed-{seed}", not bad, f"bad={bad[:3]}")


def static_rows() -> None:
    root = Path(importlib.import_module("lilbee").__file__).resolve().parent
    fleet = root / "providers" / "fleet"
    src = (fleet / "host_caps.py")
    text = src.read_text() if src.exists() else ""
    row("static/module-exists", src.exists(), str(src.relative_to(root)))
    for token in ("/proc/cpuinfo", "sysctl", "nvidia-smi", "IsProcessorFeaturePresent", "ctypes", "subprocess"):
        row(f"static/no-{token}", src.exists() and token not in text, "")
    mc = (root / "providers" / "model_cache.py").read_text()
    row("static/model_cache-no-own-nvml", "import pynvml" not in mc, "model_cache imports the NVML package itself")
    others = [p for p in root.rglob("*.py") if p != src and ("import pynvml" in p.read_text())]
    row("static/one-nvml-reader", src.exists() and not others, ", ".join(p.name for p in others))
    row("static/smi-utilization-stays", "nvidia-smi" in (fleet / "gpu_backends" / "nvidia.py").read_text(), "")
    row("static/smi-topology-stays", "nvidia-smi" in (fleet / "devices.py").read_text(), "")
    row("static/no-cpuinfo-reader-elsewhere",
        not [p.name for p in root.rglob("*.py") if p != src and "/proc/cpuinfo" in p.read_text()
             and "cpuinfo" in p.name], "")


def unchanged_rows(seed: int) -> None:
    """Contexts that must not change: head equals base."""
    from lilbee.providers import model_cache
    from lilbee.providers.fleet import gpu_hardware

    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetCount = lambda: 2
    fake.nvmlDeviceGetHandleByIndex = lambda i: i
    fake.nvmlDeviceGetUUID = lambda h: f"GPU-{h}"
    fake.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=(h + 1) * 1000)
    saved = sys.modules.get("pynvml")
    sys.modules["pynvml"] = fake
    try:
        row("unchanged/model_cache-nvml-totals", model_cache._nvidia_device_totals() == [("GPU-0", 1000), ("GPU-1", 2000)],
            str(model_cache._nvidia_device_totals()))
    finally:
        if saved is None:
            del sys.modules["pynvml"]
        else:
            sys.modules["pynvml"] = saved
    with tempfile.TemporaryDirectory() as d:
        for dev, cls, ven in [("0000:01:00.0", "0x030000", "0x10de"), ("0000:00:1f.0", "0x060100", "0x8086"),
                              ("0000:02:00.0", "0x038000", "0x1002")]:
            (Path(d) / dev).mkdir()
            (Path(d) / dev / "class").write_text(cls + "\n")
            (Path(d) / dev / "vendor").write_text(ven + "\n")
        old = gpu_hardware._SYSFS_PCI_DEVICE_DIR
        gpu_hardware._SYSFS_PCI_DEVICE_DIR = Path(d)
        real = sys.platform
        sys.platform = "linux"
        try:
            row("unchanged/gpu_hardware-vendor-ids",
                gpu_hardware.installed_gpu_vendor_ids() == frozenset({0x10DE, 0x1002}), "")
        finally:
            gpu_hardware._SYSFS_PCI_DEVICE_DIR = old
            sys.platform = real


def gate_rows() -> None:
    """The expected engine table: reference gates over recorded answers."""
    want = {"mac-m1pro": {"ik_llama.cpp": "out", "mistral.rs-cuda": "out"},
            "home-xeon-1070ti": {"ik_llama.cpp": "out", "mistral.rs-cuda": "out"},
            "gcp-epyc-nogpu": {"ik_llama.cpp": "in", "mistral.rs-cuda": "out"}}
    for name, probe, _ in fixtures():
        avx2, _n, gpus, drv, _o, _a = EXPECT[name]
        got = oracle_gate(avx2, gpus[0][1] if gpus else None, drv, probe["system"], probe["machine"])
        row(f"gate/{name}", got == want[name], str(got))


def live_rows(mod) -> None:
    """detect() on this host against the host's own raw text (the CI runner guard)."""
    if sys.platform.startswith("linux"):
        raw = Path("/proc/cpuinfo").read_text()
        has = any(" avx2" in ln for ln in raw.splitlines() if ln.startswith("flags"))
        row("live/avx2-vs-proc", mod.detect().avx2.present is has, f"proc={has}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1211)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    oracle_rows()
    gate_rows()
    mod = load()
    try:
        static_rows()
    except Exception as exc:
        row("static/error", False, repr(exc)[:80])
    try:
        unchanged_rows(a.seed)
    except Exception as exc:
        row("unchanged/error", False, repr(exc)[:80])
    if isinstance(mod, ImportError):
        row("module/import", False, str(mod))
        for name, *_ in fixtures():
            for cell in ("avx2", "neon", "gpus", "driver_cuda", "os_arch", "macos_version", "npus"):
                row(f"fixture/{name}/{cell}", False, "module absent")
        row(f"generated/{a.n}-inputs-seed-{a.seed}", False, "module absent")
        row("live/avx2-vs-proc", False, "module absent")
    else:
        host_rows(mod)
        generated_rows(mod, a.seed, a.n)
        live_rows(mod)
    red = [r for r in ROWS if not r[1]]
    print(f"SUMMARY rows={len(ROWS)} red={len(red)} green={len(ROWS) - len(red)} inputs={a.n}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
