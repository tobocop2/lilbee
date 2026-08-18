"""The fast-downloads toggle reaches hf_xet, which only reads the environment.

huggingface_hub exposes `HF_XET_HIGH_PERFORMANCE` as a constant it never acts
on; hf_xet reads the variable itself in its Rust layer. Setting the constant
does nothing, so lilbee has to publish the variable.
"""

from __future__ import annotations

import os

import pytest

from lilbee.catalog import download as dl


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, "1"), (False, None)],
    ids=["on", "off"],
)
def test_the_setting_publishes_the_variable_xet_reads(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: str | None
) -> None:
    from lilbee.core.config.model import cfg

    monkeypatch.delenv(dl._XET_HIGH_PERFORMANCE_ENV, raising=False)
    monkeypatch.setattr(cfg, "fast_model_downloads", enabled)

    dl._apply_fast_download_mode()

    assert os.environ.get(dl._XET_HIGH_PERFORMANCE_ENV) == expected


def test_turning_it_off_clears_an_inherited_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell may already export it. Off has to mean off."""
    from lilbee.core.config.model import cfg

    monkeypatch.setenv(dl._XET_HIGH_PERFORMANCE_ENV, "1")
    monkeypatch.setattr(cfg, "fast_model_downloads", False)

    dl._apply_fast_download_mode()

    assert dl._XET_HIGH_PERFORMANCE_ENV not in os.environ


def test_the_variable_name_is_the_one_hf_xet_reads() -> None:
    """A rename upstream must fail here, not silently stop applying."""
    from huggingface_hub import constants

    assert hasattr(constants, dl._XET_HIGH_PERFORMANCE_ENV)


def test_the_download_path_applies_it_before_transferring(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Xet caches its config when the session is built, so this has to run
    before the transfer, not after. It also has to run in the parent, so a
    download child process inherits the variable through its environment."""
    calls: list[str] = []
    monkeypatch.setattr(dl, "_apply_fast_download_mode", lambda: calls.append("applied"))
    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)

    def _fake_fetch(entry: object, models_dir: object, token: object, **_kw: object) -> object:
        calls.append("downloaded")
        raise RuntimeError("stop here")

    monkeypatch.setattr(dl, "fetch_model_files", _fake_fetch)

    with pytest.raises(RuntimeError):
        dl.download_model(
            dl.CatalogModel(
                hf_repo="acme/x",
                gguf_filename="x.gguf",
                size_gb=1.0,
                min_ram_gb=2.0,
                description="",
                featured=False,
                downloads=0,
                task="chat",
            )
        )

    assert calls == ["applied", "downloaded"]


@pytest.mark.parametrize("platform", ["win32", "linux"], ids=["windows", "posix"])
def test_xet_is_disabled_only_on_windows(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """hf_xet transfers stall or deadlock on Windows (xet-core #446/#789/#850),
    so lilbee falls back to the plain HTTP path there."""
    from huggingface_hub import constants

    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.delenv(dl._XET_DISABLE_ENV, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)

    dl._disable_xet_where_it_stalls()

    on_windows = platform == "win32"
    assert (os.environ.get(dl._XET_DISABLE_ENV) == "1") is on_windows
    assert constants.HF_HUB_DISABLE_XET is on_windows


def test_an_explicit_xet_choice_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user who exported the variable keeps whatever they chose."""
    from huggingface_hub import constants

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv(dl._XET_DISABLE_ENV, "0")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)

    dl._disable_xet_where_it_stalls()

    assert os.environ[dl._XET_DISABLE_ENV] == "0"
    assert constants.HF_HUB_DISABLE_XET is False


def test_xet_disable_mutates_the_hub_constant_not_just_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """huggingface_hub parses the variable once at import, so setting the
    environment alone after that parse changes nothing in this process."""
    from huggingface_hub import constants

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.delenv(dl._XET_DISABLE_ENV, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)

    dl._disable_xet_where_it_stalls()

    from huggingface_hub.utils._runtime import is_xet_available

    assert is_xet_available() is False


def test_the_xet_disable_variable_name_is_the_one_the_hub_reads() -> None:
    """A rename upstream must fail here, not silently stop applying."""
    from huggingface_hub import constants

    assert hasattr(constants, dl._XET_DISABLE_ENV)


def test_the_download_path_disables_xet_before_transferring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must land before the transfer starts, like the
    fast-download mode above."""
    calls: list[str] = []
    monkeypatch.setattr(dl, "_disable_xet_where_it_stalls", lambda: calls.append("disabled"))

    def _fake_download(**_kw: object) -> str:
        calls.append("downloaded")
        return "/tmp/file"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download)

    from lilbee.catalog.models import CatalogModel

    entry = CatalogModel(
        hf_repo="user/repo",
        gguf_filename="f.gguf",
        size_gb=1.0,
        min_ram_gb=2,
        description="d",
        featured=False,
        downloads=0,
        task="chat",
    )
    dl._hf_download_or_translate(
        entry, dl.DownloadConfig(repo_id="user/repo", filename="f.gguf", token=None)
    )

    assert calls == ["disabled", "downloaded"]
