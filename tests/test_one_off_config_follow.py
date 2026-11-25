"""A one-off command binds the owner's fleet after a setting write, and keeps its library settings.

A one-off command (add, sync, rebuild, ask, search, topics, wiki) is bound to the owner's
fleet on flash_attention on. The owner writes flash_attention off to config.toml and
restarts its fleet on it. The one-off peer's call in flight fails once to connect and
retries. The retry must bind the owner's restarted fleet and build no private fleet.

The refresh the peer uses is the one that the services of a one-off process hand the
engine, so no long-running process registered anything. The test does not touch the
provider's refresh by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lilbee.app.services import build_services
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from tests.test_fleet_reload_plan import real_planner
from tests.test_fleet_settings_bind import (
    _call_failing_once_then_binding,
    _peer_mid_call_over_an_owner_restart,
)

__all__ = ["real_planner"]


class _ProviderSeen(Exception):
    """Stops the services build once the engine's refresh is known."""


def _refresh_a_one_off_process_gets(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """The config refresh that the services of a one-off command hand the engine."""
    seen: dict[str, object] = {}

    def capture(_config: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _ProviderSeen

    monkeypatch.setattr("lilbee.providers.factory.create_provider", capture)
    with pytest.raises(_ProviderSeen):
        build_services(cfg)
    refresh = seen.get("config_refresh")
    assert callable(refresh), "the services give the engine no config refresh"
    return refresh  # type: ignore[return-value]


def test_a_one_off_peer_binds_the_owner_fleet_after_a_setting_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    peer, states, built = _peer_mid_call_over_an_owner_restart(
        monkeypatch, tmp_path, flash_attention=False
    )
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    monkeypatch.setattr(peer, "_config_refresh", refresh, raising=False)
    # The owner's write, on disk, after the one-off command started.
    persistent_settings.save(cfg.data_root, {"flash_attention": False})
    result = peer._with_rediscover(_call_failing_once_then_binding(peer))
    assert set(peer._swaps) == set(states), "the one-off peer did not bind the owner's fleet"
    assert all(swap.bound for swap in peer._swaps.values())
    assert built == [], "the one-off peer built a private fleet"
    assert result is True


def test_a_one_off_peer_keeps_its_start_time_library_settings(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """A key that shapes the library (chunk_size) stays at its start value for the run."""
    start = cfg.chunk_size
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"chunk_size": start * 2})
    refresh()
    assert cfg.chunk_size == start


def test_a_one_off_peer_keeps_a_flag_value_when_it_takes_an_engine_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A --num-ctx flag stays while the retry takes the owner's flash_attention write."""
    monkeypatch.setattr(cfg, "flash_attention", True)
    # The flag lands on cfg before the services are built, as the CLI callback sets it.
    monkeypatch.setattr(cfg, "num_ctx", 8192)
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"flash_attention": False})
    refresh()
    assert cfg.flash_attention is False
    assert cfg.num_ctx == 8192
