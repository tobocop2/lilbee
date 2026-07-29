"""Teardown has to finish inside the window an init system gives it.

A best-effort lock waiter proceeds without the lock anyway, so waiting the full
build timeout first buys nothing and spends the whole budget: systemd sends
SIGTERM, lilbee waits, gets SIGKILLed mid-teardown, and the llama-servers it was
about to stop survive holding VRAM.
"""

from __future__ import annotations

import time

import pytest
from filelock import FileLock

from lilbee.runtime import engine_lock


def test_a_best_effort_waiter_gives_up_quickly(tmp_path) -> None:
    held = FileLock(tmp_path / "engine.lock")
    held.acquire()
    try:
        started = time.monotonic()
        with engine_lock.build_lock(tmp_path, best_effort=True):
            pass
        waited = time.monotonic() - started
    finally:
        held.release()
    assert waited < engine_lock._BUILD_LOCK_TIMEOUT_S / 4, f"waited {waited:.1f}s"


def test_a_build_caller_still_waits_the_full_timeout(monkeypatch, tmp_path) -> None:
    # A build that cannot get the lock must not proceed regardless; only the
    # teardown path is allowed to give up early.
    monkeypatch.setattr(engine_lock, "_BUILD_LOCK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(engine_lock, "_PROBE_TIMEOUT_S", 0.05)
    held = FileLock(tmp_path / "engine.lock")
    held.acquire()
    try:
        with pytest.raises(Exception, match=r"(?i)timeout|lock"), engine_lock.build_lock(tmp_path):
            pass
    finally:
        held.release()
