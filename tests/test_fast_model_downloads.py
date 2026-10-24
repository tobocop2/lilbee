"""The download settings reach hf_xet and the hub through the environment at startup.

hf_xet reads `HF_XET_HIGH_PERFORMANCE` in its Rust layer, and huggingface_hub
parses `HF_HUB_DISABLE_XET` once at import. Both are published once on the main
thread at startup; no download thread writes the environment, and a download
child process inherits them at spawn.
"""

from __future__ import annotations

import os

import pytest

import lilbee
from lilbee.core.config.model import cfg, export_download_environment

_FAST_ENV = "HF_XET_HIGH_PERFORMANCE"
_DISABLE_ENV = "HF_HUB_DISABLE_XET"


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, "1"), (False, None)],
    ids=["on", "off"],
)
def test_the_setting_publishes_the_variable_xet_reads(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: str | None
) -> None:
    monkeypatch.delenv(_FAST_ENV, raising=False)
    monkeypatch.setattr(cfg, "fast_model_downloads", enabled)

    export_download_environment()

    assert os.environ.get(_FAST_ENV) == expected


def test_turning_it_off_clears_an_inherited_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell may already export it. Off has to mean off."""
    monkeypatch.setenv(_FAST_ENV, "1")
    monkeypatch.setattr(cfg, "fast_model_downloads", False)

    export_download_environment()

    assert _FAST_ENV not in os.environ


def test_the_variable_name_is_the_one_hf_xet_reads() -> None:
    """A rename upstream must fail here, not silently stop applying."""
    from huggingface_hub import constants

    assert hasattr(constants, _FAST_ENV)


def test_the_cli_data_root_publishes_the_download_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A --data-dir config.toml can turn the setting on after import."""
    import importlib

    app = importlib.import_module("lilbee.cli.app")

    monkeypatch.delenv(_FAST_ENV, raising=False)
    monkeypatch.setattr(
        app, "overlay_persisted_settings", lambda _root: setattr(cfg, "fast_model_downloads", True)
    )
    monkeypatch.setattr(cfg, "fast_model_downloads", False)

    app._apply_data_root(tmp_path)

    assert os.environ.get(_FAST_ENV) == "1"


@pytest.mark.parametrize("platform", ["win32", "linux"], ids=["windows", "posix"])
def test_xet_is_disabled_only_on_windows(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """hf_xet transfers stall or deadlock on Windows (xet-core #446/#789/#850),
    so lilbee falls back to the plain HTTP path there."""
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.delenv(_DISABLE_ENV, raising=False)

    lilbee._disable_xet_where_it_stalls()

    assert (os.environ.get(_DISABLE_ENV) == "1") is (platform == "win32")


def test_an_explicit_xet_choice_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user who exported the variable keeps whatever they chose."""
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv(_DISABLE_ENV, "0")

    lilbee._disable_xet_where_it_stalls()

    assert os.environ[_DISABLE_ENV] == "0"


def test_the_xet_disable_variable_name_is_the_one_the_hub_reads() -> None:
    """A rename upstream must fail here, not silently stop applying."""
    from huggingface_hub import constants

    assert hasattr(constants, _DISABLE_ENV)
