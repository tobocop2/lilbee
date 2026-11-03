"""With no hardware proof row, every shipped adapter launches with the argv it had at the base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.fleet.engines import proof
from tests._engine_launch_argv import launch_argvs

# The argv of each adapter at 6c19ed322, recorded before any proof table existed.
BASE = Path(__file__).parent / "fixtures" / "engine_launch_argv_base.json"


def test_l4_with_no_row_every_shipped_engine_launches_with_the_base_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof, "HARDWARE_PROOF", {})
    base = json.loads(BASE.read_text(encoding="utf-8"))
    now = launch_argvs(monkeypatch)
    assert set(base) == {"ik_llama.cpp", "llama.cpp", "mistral.rs", "mlx", "tei"}
    assert {name: now[name] for name in base} == base
