"""With no hardware proof row, every shipped adapter launches with the argv it had at the base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.fleet.engines import proof
from tests._engine_launch_argv import launch_argvs

# The argv of each adapter at 6c19ed322, recorded before any proof table existed.
# Updated by hand after commit 05be33fae, which put the engine subcommand after
# `serve` for mistral.rs: the `text` word in chat and the `vision` word in vision.
# Updated again after commits 37f0ca6eb and 1e7fbaf0e: the mistral.rs vision
# subcommand is `multimodal`, and the TEI roles carry `--max-batch-tokens`.
# No other line changed.
BASE = Path(__file__).parent / "fixtures" / "engine_launch_argv_base.json"


def test_l4_with_no_row_every_shipped_engine_launches_with_the_base_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(proof, "HARDWARE_PROOF", {})
    base = json.loads(BASE.read_text(encoding="utf-8"))
    now = launch_argvs(monkeypatch, tmp_path / "models")
    assert set(base) == {"ik_llama.cpp", "llama.cpp", "mistral.rs", "mlx", "tei"}
    assert {name: now[name] for name in base} == base
