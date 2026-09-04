"""Startup sweep of the onefile extraction directories left by older releases."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest

import lilbee.runtime.onefile_cache as onefile_cache
from lilbee.runtime.onefile_cache import (
    BOOTSTRAP_MANIFEST_NAME,
    cleanup_stale_onefile_caches,
    remove_stale_extractions,
)

_RUNNING = "0.6.91.0-macos-arm64"


def _extraction(root: Path, name: str, *, manifest: bool = True) -> Path:
    path = root / name
    (path / "lilbee").mkdir(parents=True)
    if manifest:
        (path / BOOTSTRAP_MANIFEST_NAME).write_text("1\tlilbee/__init__.py\n", encoding="utf-8")
    return path


@pytest.fixture
def running(tmp_path: Path) -> Path:
    return _extraction(tmp_path, _RUNNING)


def test_removes_the_extraction_of_an_older_release(running: Path, caplog) -> None:
    stale = _extraction(running.parent, "0.6.90.0-macos-arm64")

    with caplog.at_level(logging.INFO, logger=onefile_cache.__name__):
        removed = remove_stale_extractions(running)

    assert removed == [stale]
    assert not stale.exists()
    assert running.is_dir()
    assert any(str(stale) in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("name", "manifest"),
    [
        pytest.param("0.6.91.0-linux-x86_64", True, id="same-version-other-build"),
        pytest.param("0.6.91.0", True, id="same-version-unkeyed"),
        pytest.param("0.6.90.0-macos-arm64", False, id="no-manifest"),
        pytest.param("bin", False, id="shared-root-bin"),
    ],
)
def test_keeps_a_sibling_that_is_not_a_stale_extraction(
    running: Path, name: str, manifest: bool
) -> None:
    kept = _extraction(running.parent, name, manifest=manifest)

    assert remove_stale_extractions(running) == []
    assert kept.is_dir()


def test_keeps_a_plain_file_beside_the_extractions(running: Path) -> None:
    config = running.parent / "config.toml"
    config.write_text("", encoding="utf-8")

    assert remove_stale_extractions(running) == []
    assert config.is_file()


def test_tolerates_an_oserror_and_keeps_going(running: Path, caplog) -> None:
    locked = _extraction(running.parent, "0.6.89.0-macos-arm64")
    free = _extraction(running.parent, "0.6.90.0-macos-arm64")
    real_rmtree = onefile_cache.shutil.rmtree

    def rmtree(path: Path) -> None:
        if path == locked:
            raise PermissionError("mapped DLL")
        real_rmtree(path)

    with (
        mock.patch.object(onefile_cache.shutil, "rmtree", side_effect=rmtree),
        caplog.at_level(logging.DEBUG, logger=onefile_cache.__name__),
    ):
        removed = remove_stale_extractions(running)

    assert removed == [free]
    assert locked.is_dir()
    assert any("mapped DLL" in record.message for record in caplog.records)


def test_tolerates_an_unreadable_cache_root(tmp_path: Path, caplog) -> None:
    missing = tmp_path / "gone" / _RUNNING

    with caplog.at_level(logging.DEBUG, logger=onefile_cache.__name__):
        assert remove_stale_extractions(missing) == []

    assert any(record.levelno == logging.DEBUG for record in caplog.records)


def test_startup_hook_is_a_noop_outside_the_compiled_binary() -> None:
    with (
        mock.patch.object(onefile_cache, "is_frozen", return_value=False),
        mock.patch.object(onefile_cache, "remove_stale_extractions") as remove,
    ):
        cleanup_stale_onefile_caches()

    remove.assert_not_called()


def test_startup_hook_sweeps_beside_the_running_extraction(running: Path) -> None:
    stale = _extraction(running.parent, "0.6.90.0-macos-arm64")

    with (
        mock.patch.object(onefile_cache, "is_frozen", return_value=True),
        mock.patch.object(
            onefile_cache.lilbee, "__file__", str(running / "lilbee" / "__init__.py")
        ),
    ):
        cleanup_stale_onefile_caches()

    assert not stale.exists()
    assert running.is_dir()
