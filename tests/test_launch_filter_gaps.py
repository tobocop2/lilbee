"""The low free memory finding of an auto plan, and the chat launch filter beside other roles."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)
from tests.test_cpu_role_memory import _unified_card

GIB = 1 << 30
CHAT = WorkerRole.CHAT
EMBED = WorkerRole.EMBED
ROLES = (Role(CHAT, 1), Role(EMBED, 1))
SNAPSHOT_FREE_GIB = 0.5


@pytest.fixture
def snapshot_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A unified-memory host: the plan snapshot holds 0.5 GiB free, the live read 64."""
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    engine = CardEngine(
        "engine-a", tmp_path / "engine-server", ROLES, _unified_card(64), spans=True
    )
    engines = engines_for_all_roles(engine)
    planning.clear_plan_probe()
    monkeypatch.setattr(
        planning.model_cache, "free_system_memory", lambda: int(SNAPSHOT_FREE_GIB * GIB)
    )
    planning.capture_plan_probe(engines)
    monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: 64 * GIB)
    yield engines
    planning.clear_plan_probe()


def _warned_roles(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The role each plan finding names, in the order logged: the launch warns "Running the"."""
    return [
        word
        for record in caplog.records
        if record.getMessage().startswith("Running the") and "is free" in record.getMessage()
        for word in ("chat", "embed")
        if record.getMessage().startswith(f"Running the {word} model")
    ]


def test_every_role_with_no_layer_on_the_card_gets_the_low_free_finding(
    snapshot_free, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cfg, "n_gpu_layers", 0)
    with caplog.at_level("WARNING"):
        plan = planning.plan_all_launches(engines=snapshot_free)

    assert {launch.role for launch in plan.launches} == {CHAT, EMBED}
    assert _warned_roles(caplog) == ["chat", "embed"]
    assert f"only {SNAPSHOT_FREE_GIB} GiB is free" in caplog.text


def test_the_free_figure_of_the_finding_is_the_one_in_the_plan_snapshot(
    snapshot_free, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The live read says 64 GiB are free; only the snapshot's figure makes the plan short."""
    monkeypatch.setattr(cfg, "n_gpu_layers", 0)
    with caplog.at_level("WARNING"):
        planning.plan_all_launches(engines=snapshot_free)

    running = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Running the")]
    assert len(running) == 2
    assert all("only 0.5 GiB is free" in message for message in running)


def test_a_role_with_every_layer_on_the_card_gets_no_finding(
    snapshot_free, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    with caplog.at_level("WARNING"):
        planning.plan_all_launches(engines=snapshot_free)

    assert _warned_roles(caplog) == []


CHAT_GIB = 30
CHAT_REF = "org/repo/chat.gguf"
EMBED_REF = "org/repo/embed.gguf"


def _roles_that_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embed_gib: int,
    *,
    swap_classes: tuple[frozenset[WorkerRole], ...] = (),
) -> set[WorkerRole]:
    """The roles of a chat split over two cards and an embed on the first card that get a server.

    A 30 GiB chat leaves each 24 GiB card enough room for its window unless the embed's
    footprint on the first card takes it: 8 GiB do, 1 GiB do not.
    """
    roles = (Role(CHAT, CHAT_GIB), Role(EMBED, embed_gib))
    prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
    cards = cards_of((24, 24))
    engine = CardEngine("engine-a", tmp_path / "engine-server", roles, cards, spans=True)
    placed = Placement(
        instances=(
            InstancePlan(role=CHAT, devices=(0, 1), tensor_split=(1, 1)),
            InstancePlan(role=EMBED, devices=(0,)),
        ),
        unplaceable_roles=(),
        swap_classes=swap_classes,
    )
    inputs = [
        ModelPlacementInput(CHAT, CHAT_GIB * GIB),
        ModelPlacementInput(EMBED, embed_gib * GIB),
    ]
    launching = planning._launching_instances(
        placed,
        inputs,
        {CHAT: CHAT_REF, EMBED: EMBED_REF},
        cards,
        engines=engines_for_all_roles(engine),
    )
    return {inst.role for inst in launching}


def test_a_chat_beside_an_embed_that_leaves_it_room_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _roles_that_launch(monkeypatch, tmp_path, embed_gib=1) == {CHAT, EMBED}


def test_a_chat_beside_an_embed_that_takes_its_room_is_refused_and_the_embed_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _roles_that_launch(monkeypatch, tmp_path, embed_gib=8) == {EMBED}


def test_an_embed_in_another_swap_class_is_not_charged_against_the_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The embed is evicted while chat is resident, so its card room is chat's to use."""
    classes = (frozenset({CHAT}), frozenset({EMBED}))
    assert _roles_that_launch(monkeypatch, tmp_path, embed_gib=8, swap_classes=classes) == {
        CHAT,
        EMBED,
    }


def test_an_embed_in_the_class_of_the_chat_is_charged_against_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    classes = (frozenset({CHAT, EMBED}),)
    assert _roles_that_launch(monkeypatch, tmp_path, embed_gib=8, swap_classes=classes) == {EMBED}


class _ReserveEngine(CardEngine):
    """An engine with a fixed reserve per server, which counts the sizes it is asked for."""

    memory_kind = MemoryKind.FIXED_RESERVE

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.estimates = 0

    def estimate_memory(self, request):  # type: ignore[no-untyped-def]
        self.estimates += 1
        return super().estimate_memory(request)


def test_the_launch_filter_does_not_ask_for_the_declared_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fixed-reserve engine declares its figure; the filter only needs the window."""
    roles = (Role(CHAT, 4),)
    prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
    cards = cards_of((24,))
    engine = _ReserveEngine("engine-a", tmp_path / "engine-server", roles, cards, spans=True)
    chat = InstancePlan(role=CHAT, devices=(0,))
    planning._launch_for(
        chat, CHAT_REF, {0: cards[0]}, reserved_by_device={}, est_vram_bytes=0, engine=engine
    )
    sizes_for_one_launch = engine.estimates
    engine.estimates = 0

    launching = planning._launching_instances(
        Placement(instances=(chat,), unplaceable_roles=()),
        [ModelPlacementInput(CHAT, 4 * GIB)],
        {CHAT: CHAT_REF},
        cards,
        engines=engines_for_all_roles(engine),
    )

    assert [inst.role for inst in launching] == [CHAT]
    assert sizes_for_one_launch > 0
    assert engine.estimates == sizes_for_one_launch
