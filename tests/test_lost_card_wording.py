"""A card the machine lost is worded once, on the set path and on the view path."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import lost_card_reason
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, engines_for_all_roles, prepare_host

GIB = 1 << 30
CHAT_ONLY = (Role(WorkerRole.CHAT, 1),)
CARD_ZERO = FleetDevice("CUDA", 0, "NVIDIA GeForce GTX 1650 Ti", 4 * GIB, 4 * GIB)
LOST_CARD_SPEC = '{"chat": {"devices": [1], "target": "gpu"}}'


def _engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, CHAT_ONLY, 64 * GIB)
    return CardEngine("engine-a", tmp_path / "engine-server", CHAT_ONLY, [CARD_ZERO])


def _set_sentence(engine: CardEngine) -> str:
    with pytest.raises(PlacementError) as caught:
        planning.resolve_placement_plan(
            PlacementSpec.from_json(LOST_CARD_SPEC), engines=engines_for_all_roles(engine)
        )
    return str(caught.value)


def _get_sentence(engine: CardEngine, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(cfg, "placement", LOST_CARD_SPEC)
    plan = planning.plan_launches(
        None, {CARD_ZERO.index: CARD_ZERO}, [CARD_ZERO], engines=engines_for_all_roles(engine)
    )
    return plan.refused_roles[WorkerRole.CHAT]


def test_two_lost_cards_and_a_host_count_are_named() -> None:
    sentence = lost_card_reason(WorkerRole.CHAT, [1, 2], [0], on_host=3, hint="Hint.")
    assert sentence == (
        "chat is pinned to devices 1 and 2, but this machine lists only device 0 "
        "(3 cards on this host). Hint. "
        "Set the chat devices to a card that exists, or set its target to cpu."
    )


def test_a_machine_that_lists_nothing_says_so() -> None:
    assert lost_card_reason(WorkerRole.CHAT, [0], []) == (
        "chat is pinned to device 0, but this machine lists no devices. "
        "Set the chat devices to a card that exists, or set its target to cpu."
    )


@pytest.mark.parametrize(
    ("on_host", "count"),
    [(1, "(1 card on this host)"), (2, "(2 cards on this host)")],
)
def test_the_host_count_agrees_with_its_noun(on_host: int, count: str) -> None:
    sentence = lost_card_reason(WorkerRole.CHAT, [0], [], on_host=on_host)
    assert f"lists no devices {count}. " in sentence


def test_three_listed_devices_are_joined_with_commas_and_an_and() -> None:
    assert lost_card_reason(WorkerRole.CHAT, [9], [2, 0, 1]).startswith(
        "chat is pinned to device 9, but this machine lists only devices 0, 1 and 2. "
    )


def test_the_shared_wording_counts_the_host_cards() -> None:
    host_cards = planning.HostCards([CARD_ZERO])
    host_cards.__dict__["total"] = 3
    wording = planning._lost_card_wording([CARD_ZERO], host_cards)
    assert wording(WorkerRole.CHAT, [1]) == (
        "chat is pinned to device 1, but this machine lists only device 0 "
        "(3 cards on this host). Set the chat devices to a card that exists, "
        "or set its target to cpu."
    )


def test_a_role_without_an_entry_in_the_spec_is_not_refused() -> None:
    spec = PlacementSpec.from_json('{"chat": {"devices": [1], "target": "gpu"}}')
    roles = {WorkerRole.CHAT: "chat-ref", WorkerRole.EMBED: "embed-ref"}
    host_cards = planning.HostCards([CARD_ZERO])
    host_cards.__dict__["total"] = 1
    refused = planning._roles_on_missing_cards(spec, roles, [CARD_ZERO], host_cards)
    assert list(refused) == [WorkerRole.CHAT]


def test_set_and_get_word_a_lost_card_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path)
    monkeypatch.setattr(planning, "_hidden_device_hint", lambda _devices: "Hidden.")
    on_set = _set_sentence(engine)
    on_get = _get_sentence(engine, monkeypatch)
    assert on_set == on_get
    assert on_get == (
        "chat is pinned to device 1, but this machine lists only device 0. Hidden. "
        "Set the chat devices to a card that exists, or set its target to cpu."
    )
