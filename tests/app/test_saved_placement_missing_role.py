"""A saved placement that lacks an entry for a configured role is refused with the set path's text.

The planner is the real one. The launch falls back to the automatic plan for a spec of cards;
the read path says why, in the sentence the set path raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 4), Role(WorkerRole.EMBED, 1))
CHAT_ONLY = '{"chat": {"devices": [0]}}'
NO_ENTRY_FOR_EMBED = (
    "embed has a model but the placement has no entry for it. "
    "Add an entry for embed with devices or a target, or clear the placement."
)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    engine = CardEngine("engine-a", tmp_path / "a", ROLES, cards_of((24,)))
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    monkeypatch.setattr(cfg, "placement", CHAT_ONLY)


def test_the_read_path_gives_the_refusal_the_set_path_raises(host: None) -> None:
    spec = PlacementSpec.from_json(CHAT_ONLY)
    with pytest.raises(PlacementError) as raised:
        app_placement.preview_placement(spec)
    view = app_placement.get_placement()
    assert str(raised.value) == NO_ENTRY_FOR_EMBED
    assert view.refusal == NO_ENTRY_FOR_EMBED
    assert view.refusal_kind is app_placement.PlacementRefusal.REFUSED
    assert view.rejected_spec_json == spec.to_json()
    assert view.manual is False
