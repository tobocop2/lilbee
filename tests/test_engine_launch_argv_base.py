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
# The mistral.rs vision argv carries no context flag, as its multimodal loader refuses one.
# The mlx chat argv carries `--max-tokens` at the window, because mlx-lm otherwise stops
# every answer at 512 tokens. It also carries llama-server's sampling defaults, because
# mlx-lm otherwise decodes greedily and a thinking model repeats itself without end.
# Each ik_llama.cpp argv ends with `--no-context-shift`, because with context shift on
# ik_llama.cpp cuts an oversize prompt and then aborts instead of refusing it.
# The mlx embed and rerank rows are new: they run lilbee's own server of the mlx pack.
# No other line changed.
BASE = Path(__file__).parent / "fixtures" / "engine_launch_argv_base.json"


def test_l4_with_no_row_every_shipped_engine_launches_with_the_base_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(proof, "HARDWARE_PROOF", {})
    recorded = json.loads(BASE.read_text(encoding="utf-8"))

    # The base holds the POSIX form of each path; the adapters write the host's form.
    def host_form(arg: str) -> str:
        return str(Path(arg)) if arg.startswith("/") else arg

    base = {
        name: {role: [host_form(arg) for arg in argv] for role, argv in roles.items()}
        for name, roles in recorded.items()
    }
    now = launch_argvs(monkeypatch, tmp_path / "models")
    assert set(base) == {"ik_llama.cpp", "llama.cpp", "mistral.rs", "mlx", "tei"}
    assert {name: now[name] for name in base} == base
