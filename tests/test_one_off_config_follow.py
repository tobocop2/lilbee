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

from pathlib import Path

import pytest

from lilbee.app.services import follow_config_one_off, get_services, set_services
from lilbee.cli.app import apply_overrides
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


def _refresh_the_engine_gets(monkeypatch: pytest.MonkeyPatch, build: object) -> object:
    """The config refresh that *build*, a services build, hands the engine."""
    seen: dict[str, object] = {}

    def capture(_config: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _ProviderSeen

    monkeypatch.setattr("lilbee.providers.factory.create_provider", capture)
    with pytest.raises(_ProviderSeen):
        build()  # type: ignore[operator]
    return seen.get("config_refresh")


def _refresh_a_one_off_process_gets(monkeypatch: pytest.MonkeyPatch) -> object:
    """The config refresh that the services of a one-off command hand the engine."""
    set_services(None)
    refresh = _refresh_the_engine_gets(monkeypatch, get_services)
    assert callable(refresh), "the services give the engine no config refresh"
    return refresh


def test_a_one_off_peer_binds_the_owner_fleet_after_a_setting_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    peer, states, built = _peer_mid_call_over_an_owner_restart(
        monkeypatch, tmp_path, flash_attention=False
    )
    follow_config_one_off(frozenset())  # what the CLI entry does for a command with no flags
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
    follow_config_one_off(frozenset())
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"chunk_size": start * 2})
    refresh()  # type: ignore[operator]
    assert cfg.chunk_size == start


def test_ask_with_num_ctx_keeps_the_flag_over_the_owner_num_ctx_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ask --num-ctx 4096` while the owner writes num_ctx: the retry keeps 4096.

    The retry still takes the owner's flash_attention write, an engine key no flag set.
    """
    monkeypatch.setattr(cfg, "flash_attention", True)
    apply_overrides(data_dir=cfg.data_root, num_ctx=4096)
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"num_ctx": 8192, "flash_attention": False})
    refresh()  # type: ignore[operator]
    assert cfg.num_ctx == 4096
    assert cfg.flash_attention is False


def test_a_long_running_follower_stays_over_a_later_cli_entry() -> None:
    """serve, MCP and the TUI keep their full take when a CLI entry runs after them."""
    from lilbee.app import services as svc_mod

    follow_config_one_off(frozenset({"num_ctx"}))
    first = svc_mod.follow_config_long_running(frozenset())
    follow_config_one_off(frozenset({"num_ctx"}))
    assert svc_mod.follow_config_long_running(frozenset()) is first
    assert svc_mod._state.follower is first
    assert first.hold == frozenset()


def test_a_library_api_container_takes_nothing_from_config_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Lilbee built without a provider gets no refresh, even in a process with a follower."""
    import lilbee

    follow_config_one_off(frozenset())
    refresh = _refresh_the_engine_gets(
        monkeypatch, lambda: lilbee.Lilbee(documents_dir=tmp_path / "kb")
    )
    assert refresh is None
