"""Tests for gpu_backends/base.py shared helpers."""

from __future__ import annotations

import subprocess

import pytest

from lilbee.providers.fleet.gpu_backends import base as base_mod
from lilbee.providers.fleet.gpu_backends.base import extract_int, parse_device_index, run_smi


# ---------------------------------------------------------------------------
# extract_int
# ---------------------------------------------------------------------------


def test_extract_int_integer_value() -> None:
    assert extract_int({"a": 10}, ("a",)) == 10


def test_extract_int_float_value() -> None:
    assert extract_int({"a": 35.9}, ("a",)) == 35


def test_extract_int_decimal_string() -> None:
    """rocm-smi emits values like '35.0'; must parse as 35, not raise."""
    assert extract_int({"a": "35.0"}, ("a",)) == 35


def test_extract_int_decimal_string_with_fraction() -> None:
    """Fractional decimal strings are truncated, not rounded."""
    assert extract_int({"a": "35.7"}, ("a",)) == 35


def test_extract_int_finds_first_matching_key() -> None:
    obj = {"a": 10, "b": 20}
    assert extract_int(obj, ("b", "a")) == 20


def test_extract_int_returns_none_for_missing_keys() -> None:
    assert extract_int({}, ("x", "y")) is None


def test_extract_int_skips_non_numeric_strings() -> None:
    obj = {"a": "bad", "b": 7}
    assert extract_int(obj, ("a", "b")) == 7


def test_extract_int_returns_none_for_non_dict() -> None:
    assert extract_int("string", ("a",)) is None


def test_extract_int_returns_none_for_none_value() -> None:
    assert extract_int({"a": None}, ("a",)) is None


# ---------------------------------------------------------------------------
# parse_device_index
# ---------------------------------------------------------------------------


def test_parse_device_index_card_prefix() -> None:
    assert parse_device_index("card0") == 0
    assert parse_device_index("card12") == 12


def test_parse_device_index_bracket_format() -> None:
    assert parse_device_index("GPU[3]") == 3


def test_parse_device_index_no_digit_returns_none() -> None:
    assert parse_device_index("GPU") is None


# ---------------------------------------------------------------------------
# run_smi
# ---------------------------------------------------------------------------


def test_run_smi_returns_stdout_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="output\n", stderr="")

    monkeypatch.setattr(base_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(base_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert run_smi("nvidia-smi", ["--version"]) == "output\n"


def test_run_smi_returns_empty_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="output\n", stderr="")

    monkeypatch.setattr(base_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(base_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert run_smi("nvidia-smi", []) == ""


def test_run_smi_returns_empty_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise OSError("no such file")

    monkeypatch.setattr(base_mod.subprocess, "run", _boom)
    monkeypatch.setattr(base_mod.shutil, "which", lambda _: None)
    assert run_smi("missing-tool", []) == ""


def test_run_smi_resolves_which_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved: list[str] = []

    def _fake_which(name: str) -> str | None:
        resolved.append(name)
        return f"/custom/{name}"

    monkeypatch.setattr(base_mod.shutil, "which", _fake_which)
    monkeypatch.setattr(
        base_mod.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    run_smi("my-tool", [])
    assert "my-tool" in resolved


def test_run_smi_falls_back_to_tool_name_when_which_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When shutil.which returns None the tool name itself is used as binary."""
    called_with: list[list[str]] = []

    def _fake_run(*args: object, **_k: object) -> subprocess.CompletedProcess:
        called_with.append(list(args[0]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(base_mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(base_mod.subprocess, "run", _fake_run)
    run_smi("my-tool", ["--flag"])
    assert called_with[0][0] == "my-tool"
