"""A driver that never answers costs each card-reading caller one bounded wait, not a stall."""

from __future__ import annotations

import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from lilbee.providers import model_cache
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import visible_nvidia_cards
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in tool is a shell script"
)

_BOUND_S = 0.5
_ALLOWED_S = 5.0
_T = TypeVar("_T")


@pytest.fixture(autouse=True)
def _wedged_nvidia_smi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put an nvidia-smi on PATH that sleeps far past the probe's time limit."""
    tool = tmp_path / "nvidia-smi"
    tool.write_text("#!/bin/sh\nexec sleep 600\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/bin:/usr/bin")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(nvidia_mod, "_TIMEOUT_S", _BOUND_S)
    nvidia_mod._cache.reset()


def _timed(call: Callable[[], _T]) -> tuple[_T, float]:
    started = time.monotonic()
    result = call()
    return result, time.monotonic() - started


def test_the_visible_card_read_returns_within_the_bound() -> None:
    cards, took = _timed(visible_nvidia_cards)
    assert cards == []
    assert took < _ALLOWED_S


def test_the_memory_budget_read_returns_within_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_cache.platform, "system", lambda: "Linux")
    present, took = _timed(model_cache.has_nvidia_gpu)
    assert present is False
    assert took < _ALLOWED_S


def test_the_planning_card_count_returns_within_the_bound() -> None:
    count, took = _timed(lambda: planning.HostCards([]).nvidia_driver)
    assert count == 0
    assert took < _ALLOWED_S
