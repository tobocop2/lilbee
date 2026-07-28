"""Inspect the engine binary's linked runtimes and its device probe's output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_LDD_TIMEOUT_S = 10
# Substrings that mark a device-init failure in the probe output. The HIP
# backend is compiled from ggml-cuda, so its error lines also say "cuda".
_ERROR_MARKERS = ("error", "fail", "no cuda")
# How much of the probe output to quote when no specific error line is found.
_DIAGNOSTIC_TAIL_CHARS = 300


def ldd_output(binary: Path, env: dict[str, str]) -> str | None:
    """``ldd`` stdout for *binary* under *env*; None when ldd can't run on it."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - ldd path and the resolved binary
            [ldd, str(binary)],
            capture_output=True,
            text=True,
            timeout=_LDD_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # Not an ELF, a static binary, or a timeout: nothing to inspect.
        return None
    return proc.stdout


def links_any(binary: Path, env: dict[str, str], sonames: tuple[str, ...]) -> bool:
    """True when *binary* lists any of *sonames*, resolved or not."""
    out = ldd_output(binary, env)
    if out is None:
        return False
    return any(soname in out for soname in sonames)


def device_probe_diagnostic(probe_output: str) -> str:
    """The probe's device-init error line, or a short tail of its output."""
    out = probe_output.strip()
    for line in out.splitlines():
        lowered = line.lower()
        if "cuda" in lowered and any(marker in lowered for marker in _ERROR_MARKERS):
            return line.strip()
    return out[-_DIAGNOSTIC_TAIL_CHARS:] if out else "(the engine's device probe printed nothing)"
