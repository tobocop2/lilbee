"""The real_gguf_parser fixture reads the engine pack from the real engine store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.conftest as suite_conftest
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines import LLAMA


def _plant_pack(models_dir: Path) -> Path:
    """Write a complete build of the pinned pack beside *models_dir*; return its gguf-parser."""
    pin = LLAMA.pack_pin()
    build = models_dir.parent / "engines" / LLAMA.name.value / f"{pin['version']}-{pin['backend']}"
    parser = build / "bin" / "gguf-parser"
    parser.parent.mkdir(parents=True)
    parser.write_text("#!/bin/sh\n", encoding="utf-8")
    parser.chmod(0o755)
    record = {
        "engine": LLAMA.name.value,
        "version": pin["version"],
        "backend": pin["backend"],
        "sha256": "0" * 64,
        "licences": ["MIT"],
        "installed_at": "2026-01-01T00:00:00+00:00",
        "binary": "bin/gguf-parser",
    }
    (build / "ENGINE-RECORD.json").write_text(json.dumps(record), encoding="utf-8")
    return parser


def test_the_fixture_returns_the_parser_from_the_pack_in_the_real_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    real_models = tmp_path / "real" / "models"
    planted = _plant_pack(real_models)
    monkeypatch.setattr(suite_conftest, "REAL_MODELS_DIR", real_models)
    assert not (cfg.models_dir.parent / "engines").exists()

    try:
        found = request.getfixturevalue("real_gguf_parser")
    except pytest.skip.Exception as skipped:
        pytest.fail(f"the fixture skipped although the pack is planted: {skipped}")

    assert found == planted
    assert cfg.models_dir == real_models


def test_the_fixture_skips_with_the_missing_binary_named_when_no_pack_is_planted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    empty_models = tmp_path / "empty" / "models"
    monkeypatch.setattr(suite_conftest, "REAL_MODELS_DIR", empty_models)
    monkeypatch.delenv("LILBEE_REQUIRE_GGUF_PARSER", raising=False)

    with pytest.raises(pytest.skip.Exception, match="gguf-parser is not on PATH"):
        request.getfixturevalue("real_gguf_parser")


def test_the_fixture_fails_instead_of_skipping_when_the_parser_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(suite_conftest, "REAL_MODELS_DIR", tmp_path / "empty" / "models")
    monkeypatch.setenv("LILBEE_REQUIRE_GGUF_PARSER", "1")

    with pytest.raises(pytest.fail.Exception, match="gguf-parser is not on PATH"):
        request.getfixturevalue("real_gguf_parser")
