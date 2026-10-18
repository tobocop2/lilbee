"""main_gpu is not a setting: llama-server reads it only under split mode none.

A fleet instance on one card has nothing to choose, and one spread over several
cards was sized for that spread, so the setting is refused rather than ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from conftest import clean_env
from lilbee.app.settings import apply_settings_update
from lilbee.core.config import Config
from lilbee.core.config.load_warnings import collecting
from lilbee.core.config.model import _build_cfg, toml_value, toml_values
from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.roles import WorkerRole

_HINT = "choose the cards with gpu_devices or placement"


def _built_with(tmp_path: Path, toml: str, **env: str) -> tuple[Config, tuple[str, ...]]:
    """cfg built from *toml* in a fresh data root and *env*, with its load warnings."""
    (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
    with mock.patch.dict(os.environ, {**clean_env(tmp_path), **env}, clear=True):
        return _build_cfg()


def test_config_has_no_main_gpu_field() -> None:
    assert "main_gpu" not in Config.model_fields
    assert "gpu_devices" in Config.model_fields


def test_a_stored_main_gpu_is_refused_by_name(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path, "main_gpu = 1\ntop_k = 7\n")

    assert warnings == (f"config.toml: main_gpu is not a setting and is ignored; {_HINT}",)
    assert _built.top_k == 7


def test_a_later_read_of_another_file_names_that_file(tmp_path) -> None:
    other = tmp_path / "other.toml"
    other.write_text("main_gpu = 1\ntop_k = 7\n", encoding="utf-8")

    with collecting() as found:
        values = toml_values(other)

    assert values["top_k"] == 7
    assert found == [f"{other}: main_gpu is not a setting and is ignored; {_HINT}"]


def test_a_read_of_one_other_key_says_nothing_about_main_gpu(tmp_path) -> None:
    other = tmp_path / "other.toml"
    other.write_text("main_gpu = 1\ntop_k = 7\n", encoding="utf-8")

    with collecting() as found:
        value = toml_value(other, "top_k")

    assert (value, found) == (7, [])


def test_the_environment_variable_is_refused_by_name(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path, "", LILBEE_MAIN_GPU="1")

    assert warnings == (f"LILBEE_MAIN_GPU is not a setting and is ignored; {_HINT}",)


def test_a_blank_environment_variable_says_nothing(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path, "", LILBEE_MAIN_GPU="")

    assert warnings == ()


def test_a_settings_write_refuses_main_gpu() -> None:
    with pytest.raises(ValueError, match=r"^Unknown or read-only setting: main_gpu$"):
        apply_settings_update({"main_gpu": 1})


def test_a_multi_device_launch_sends_no_main_gpu() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/m/c.gguf"),
        devices=(0, 1),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
    )

    assert "--main-gpu" not in argv
    assert "--model" in argv
