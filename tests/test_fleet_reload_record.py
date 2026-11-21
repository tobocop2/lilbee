"""A settings reload that starts a role again drops the record of that role's failed start."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole
from tests import test_fleet_reload_plan as plan_tests
from tests.test_fleet_reload_plan import (
    _CHAT_REF,
    _OTHER_CHAT_REF,
    _fail_first_starts,
    _join_fleet_threads,
)

# The fixtures are shared by assignment: pytest finds them by module attribute.
engine_fleet = plan_tests.engine_fleet
real_planner = plan_tests.real_planner

CHAT = WorkerRole.CHAT


def test_a_settings_reload_that_starts_a_recorded_role_drops_its_record(
    engine_fleet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reload names no forced role, so only a clear of the whole record empties it."""
    provider, _swaps = engine_fleet
    starts = _fail_first_starts(monkeypatch, 1)
    cfg.chat_model = _OTHER_CHAT_REF
    provider.reload_role(CHAT)
    _join_fleet_threads()
    assert CHAT in provider._failed_starts
    assert not provider._clients.get(CHAT)

    cfg.chat_model = _CHAT_REF
    provider.reload_role(CHAT)
    _join_fleet_threads()

    assert provider._clients.get(CHAT)
    assert len(starts) == 2
    assert provider._failed_starts == {}
