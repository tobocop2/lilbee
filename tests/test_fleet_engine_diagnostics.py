"""Tests for the shared engine-binary linkage and probe-output inspection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.fleet import engine_diagnostics

_LINKS_CUDA = "\tlibcudart.so.12 => /usr/lib/libcudart.so.12 (0x00007f00)\n"


def test_ldd_output_none_when_ldd_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: None)
    assert engine_diagnostics.ldd_output(Path("/bin/llama-server"), {}) is None


def test_ldd_output_none_when_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: "/usr/bin/ldd")

    def _raise(*_a: object, **_k: object) -> None:
        raise OSError("not an ELF binary")

    monkeypatch.setattr(engine_diagnostics.subprocess, "run", _raise)
    assert engine_diagnostics.ldd_output(Path("/bin/llama-server"), {}) is None


def test_links_any_matches_a_listed_soname_resolved_or_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: "/usr/bin/ldd")
    monkeypatch.setattr(
        engine_diagnostics.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=_LINKS_CUDA, stderr=""),
    )
    binary = Path("/bin/llama-server")
    assert engine_diagnostics.links_any(binary, {}, ("libcudart.so.12",)) is True
    assert engine_diagnostics.links_any(binary, {}, ("libamdhip64.so",)) is False


def test_links_any_false_when_binary_cannot_be_inspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No linkage evidence is not evidence of linkage."""
    monkeypatch.setattr(engine_diagnostics, "ldd_output", lambda *_a: None)
    assert engine_diagnostics.links_any(Path("/bin/llama-server"), {}, ("libc.so.6",)) is False


def test_device_probe_diagnostic_returns_tail_when_no_error_line() -> None:
    out = engine_diagnostics.device_probe_diagnostic("CUDA0: NVIDIA L40 (45 GiB)\n")
    assert "NVIDIA L40" in out  # no error marker -> falls through to the output tail


def test_device_probe_diagnostic_picks_the_cuda_error_line() -> None:
    output = "loading backends\nggml_cuda_init: CUDA error: unknown error\ntrailing noise\n"
    assert (
        engine_diagnostics.device_probe_diagnostic(output)
        == "ggml_cuda_init: CUDA error: unknown error"
    )


def test_device_probe_diagnostic_when_no_output() -> None:
    assert (
        engine_diagnostics.device_probe_diagnostic("")
        == "(the engine's device probe printed nothing)"
    )
