"""Concurrent Chromium bootstraps must not unpack into the same directory at once."""

from __future__ import annotations

import asyncio

import pytest

from lilbee.crawler import bootstrap


@pytest.fixture(autouse=True)
def browsers_root(tmp_path, monkeypatch):
    """Point the browser cache (and so the lock file) at a temp dir."""
    monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
    return tmp_path


async def test_a_second_bootstrap_waits_and_then_skips_the_install(monkeypatch):
    """The loser of the race must not run a second install.

    Two callers reach here whenever a POST /setup/crawler races another one or
    the first-use bootstrap a crawl triggers. Without the lock both spawn
    ``playwright install chromium`` into one directory and corrupt it.
    """
    installs = 0
    first_inside = asyncio.Event()
    release_first = asyncio.Event()
    installed = False

    def _installed() -> bool:
        return installed

    async def _fake_install(_on_progress) -> None:
        nonlocal installs, installed
        installs += 1
        first_inside.set()
        await release_first.wait()
        installed = True

    monkeypatch.setattr(bootstrap, "chromium_installed", _installed)
    monkeypatch.setattr(bootstrap, "_install_chromium", _fake_install)

    first = asyncio.create_task(bootstrap.bootstrap_chromium())
    await asyncio.wait_for(first_inside.wait(), timeout=5)

    second = asyncio.create_task(bootstrap.bootstrap_chromium())
    await asyncio.sleep(0.05)
    assert not second.done(), "second caller must block on the lock, not install"

    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    assert installs == 1


async def test_a_lock_timeout_reports_the_wait_instead_of_installing(monkeypatch):
    """Rather than unpack alongside another install, say what is happening."""
    monkeypatch.setattr(bootstrap, "chromium_installed", lambda: False)
    monkeypatch.setattr(bootstrap, "_BOOTSTRAP_LOCK_TIMEOUT_S", 0.05)

    async def _unexpected(_on_progress) -> None:
        raise AssertionError("must not install while another holds the lock")

    monkeypatch.setattr(bootstrap, "_install_chromium", _unexpected)

    held = bootstrap.FileLock(str(bootstrap._bootstrap_lock_path()))
    held.acquire()
    try:
        with pytest.raises(bootstrap.CrawlerBrowserError, match="Timed out"):
            await bootstrap.bootstrap_chromium()
    finally:
        held.release()
