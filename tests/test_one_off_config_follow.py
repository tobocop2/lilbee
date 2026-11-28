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


def test_a_one_off_retry_takes_the_owner_gpu_memory_fraction_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """gpu_memory_fraction is an engine tuning key: a one-off retry follows the owner's write."""
    from lilbee.providers.reload_keys import ENGINE_TUNING_KEYS

    assert "gpu_memory_fraction" in ENGINE_TUNING_KEYS
    new = 0.5 if cfg.gpu_memory_fraction != 0.5 else 0.6
    follow_config_one_off(frozenset())
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"gpu_memory_fraction": new})
    refresh()  # type: ignore[operator]
    assert cfg.gpu_memory_fraction == new


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


@pytest.mark.parametrize(
    ("key", "written"),
    [
        ("vision_model", "org/other-vision/other-vision.gguf"),
        ("embedding_model", "org/other-embed/other-embed.gguf"),
    ],
)
def test_a_one_off_retry_keeps_its_start_model_for_a_role(
    monkeypatch: pytest.MonkeyPatch, key: str, written: str
) -> None:
    """A model-role write during `lilbee add` waits for the next command."""
    from lilbee.app import services as svc_mod

    start = getattr(cfg, key)
    follow_config_one_off(frozenset())
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {key: written})
    refresh()  # type: ignore[operator]
    assert getattr(cfg, key) == start
    follower = svc_mod._state.follower
    assert follower is not None
    assert key in follower.hold


def test_every_fleet_reload_key_is_either_taken_or_held_by_a_one_off() -> None:
    """A new fleet reload key fails here until it is put in the take set or the held list."""
    from lilbee.app.config_follow import ONE_OFF_HOLD
    from lilbee.core.config import Config
    from lilbee.providers.reload_keys import ENGINE_TUNING_KEYS, FLEET_RELOAD_KEYS

    held_reload_keys = {
        "chat_model",
        "embedding_model",
        "vision_model",
        "reranker_model",
        "reranker_type",
        "ocr",
        "chunk_size",
    }
    assert ENGINE_TUNING_KEYS | held_reload_keys == FLEET_RELOAD_KEYS
    assert not ENGINE_TUNING_KEYS & held_reload_keys
    every_field = frozenset(Config.model_fields)
    assert every_field >= ENGINE_TUNING_KEYS
    assert every_field == ONE_OFF_HOLD | ENGINE_TUNING_KEYS
    assert not ONE_OFF_HOLD & ENGINE_TUNING_KEYS


def test_the_engine_takes_through_the_follower_that_replaced_the_one_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Services built before a long-running follower still take through it at a retry."""
    from lilbee.app import services as svc_mod

    follow_config_one_off(frozenset())
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    svc_mod.follow_config_long_running(frozenset(), keeper="the TUI")
    start = cfg.chunk_size
    persistent_settings.save(cfg.data_root, {"chunk_size": start * 2})
    refresh()  # type: ignore[operator]
    assert cfg.chunk_size == start * 2


def test_the_engine_retry_takes_nothing_in_a_process_with_no_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.app import services as svc_mod

    monkeypatch.setattr(svc_mod._state, "follower", None)
    monkeypatch.setattr(cfg, "flash_attention", True)
    refresh = _refresh_a_one_off_process_gets(monkeypatch)
    persistent_settings.save(cfg.data_root, {"flash_attention": False})
    refresh()  # type: ignore[operator]
    assert cfg.flash_attention is True


def test_a_long_running_follower_stays_over_a_later_cli_entry() -> None:
    """serve, MCP and the TUI keep their full take when a CLI entry runs after them."""
    from lilbee.app import services as svc_mod

    follow_config_one_off(frozenset({"num_ctx"}))
    first = svc_mod.follow_config_long_running(frozenset(), keeper="the TUI")
    follow_config_one_off(frozenset({"num_ctx"}))
    assert svc_mod.follow_config_long_running(frozenset(), keeper="the TUI") is first
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
