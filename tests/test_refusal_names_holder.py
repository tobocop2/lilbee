"""A refusal names the role that holds a card's room, on real files of real size."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.cli.tui.widgets import gpu_fleet_panel
from lilbee.core.config import cfg
from lilbee.mcp_server import get_placement_tool
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engines.mistral import MistralAdapter
from lilbee.providers.fleet.placement import InstancePlan, Placement, RoomHolder, card_refusal
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_route
from tests._async_wait import wait_until
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host
from tests._litestar_test_client import create_test_client
from tests.test_tui_fleet_drawer import _DrawerApp

GIB = 1024**3
CHAT_REF = "org/repo/chat.gguf"
EMBED_REF = "org/repo/embed.gguf"
ROOM_OF_24 = 23192823398  # 0.9 of 24 GiB is this figure plus 0.4 byte.
PIN_BOTH_AND_CPU_RERANK = (
    '{"chat": {"devices": [0]}, "embed": {"devices": [0]}, "rerank": {"target": "cpu"}}'
)
AUTO_CHAT_CPU_EMBED = '{"chat": {"target": "auto"}, "embed": {"target": "cpu"}}'


class _RealSizeEngine(MistralAdapter):
    """The real adapter, which reads each model's true size, on cards this host lacks."""

    def __init__(self, binary: Path, cards: tuple[int, ...]) -> None:
        super().__init__()
        self._binary = binary
        self._cards = cards

    def binary(self) -> Path:
        return self._binary

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=cards_of(self._cards), output="")


def plan_real_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chat: int,
    embed: int,
    card_gib: int,
    spec: str | None = None,
):
    """Plan a chat and an embed GGUF of these byte sizes through the real adapter."""
    prepare_host(
        monkeypatch, tmp_path, (Role(WorkerRole.CHAT, 1), Role(WorkerRole.EMBED, 1)), 64 * GIB
    )
    for name, size in (("chat.gguf", chat), ("embed.gguf", embed)):
        with (tmp_path / name).open("r+b") as model:
            model.truncate(size)
    monkeypatch.setattr(cfg, "placement", spec)
    binary = tmp_path / "engine-server"
    binary.write_text("", encoding="utf-8")
    engine = _RealSizeEngine(binary, (card_gib,))
    return planning.plan_all_launches(engines=dict.fromkeys(WorkerRole, engine))


def _launched(plan) -> set[WorkerRole]:
    return {launch.role for launch in plan.launches}


class TestAChatRefusedBesideAnEmbed:
    def test_a_drained_card_names_the_embed_and_sends_the_user_to_it(
        self, tmp_path, monkeypatch
    ) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, GIB, 8 * GIB, 8)
        assert _launched(plan) == {WorkerRole.EMBED}
        assert plan.refused_roles[WorkerRole.CHAT] == (
            f"The chat model {CHAT_REF} needs 1.0 GiB but device 0 has 0.0 GiB usable "
            "(8.0 GiB total, 90% headroom). "
            f"The embed model {EMBED_REF} (8.0 GiB) holds the rest. "
            "Use a smaller embed model, or set the target of one of these models "
            "to cpu or another card."
        )

    def test_a_partly_full_card_names_the_embed_with_the_true_room(
        self, tmp_path, monkeypatch
    ) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, 14 * GIB, 8 * GIB, 24)
        reason = plan.refused_roles[WorkerRole.CHAT]
        assert "needs 14.0 GiB but device 0 has 13.6 GiB usable" in reason
        assert f"The embed model {EMBED_REF} (8.0 GiB) holds the rest." in reason
        assert "Use a smaller embed model" in reason

    def test_a_chat_too_big_for_its_card_alone_is_told_to_shrink_the_chat(
        self, tmp_path, monkeypatch
    ) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, 30 * GIB, 8 * GIB, 24)
        reason = plan.refused_roles[WorkerRole.CHAT]
        assert "Use a smaller model, or set its target to cpu." in reason
        assert f"The embed model {EMBED_REF} (8.0 GiB) holds the rest." in reason
        assert "smaller embed model" not in reason

    def test_a_pair_that_fits_launches_both(self, tmp_path, monkeypatch) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, 4 * GIB, 4 * GIB, 24)
        assert _launched(plan) == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert plan.refused_roles == {}

    def test_a_chat_with_a_small_neighbour_on_a_card_it_overflows_keeps_todays_remedy(
        self, tmp_path, monkeypatch
    ) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, 22 * GIB, 4096, 24)
        reason = plan.refused_roles[WorkerRole.CHAT]
        assert reason.startswith(f"The chat model {CHAT_REF} needs 22.0 GiB")
        assert "Use a smaller model, or set its target to cpu." in reason


class TestAPinnedChatBesideAPinnedEmbed:
    def test_the_refusal_names_the_pinned_embed_and_its_model(self, tmp_path, monkeypatch) -> None:
        roles = (Role(WorkerRole.EMBED, 6), Role(WorkerRole.CHAT, 4), Role(WorkerRole.RERANK, 1))
        prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", PIN_BOTH_AND_CPU_RERANK)
        engine = CardEngine("engine-a", tmp_path / "engine-server", roles, cards_of((8,)))
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert WorkerRole.CHAT not in _launched(plan)
        reason = plan.refused_roles[WorkerRole.CHAT]
        assert reason.startswith("chat pinned to device 0 needs 4.0 GiB but device 0 has 1.2 GiB")
        assert f"The embed model {EMBED_REF} (6.0 GiB) holds the rest." in reason
        assert "Use a smaller embed model" in reason


class TestTheShortfallIsExact:
    def test_under_one_byte_past_the_room_is_refused(self, tmp_path, monkeypatch) -> None:
        # The embed stays off this card, so the room is the whole usable share.
        plan = plan_real_sizes(monkeypatch, tmp_path, ROOM_OF_24 + 1, 4096, 24, AUTO_CHAT_CPU_EMBED)
        assert WorkerRole.CHAT not in _launched(plan)
        assert plan.refused_roles[WorkerRole.CHAT] == (
            f"The chat model {CHAT_REF} needs 21.6 GiB but device 0 has 21.6 GiB usable "
            "(24.0 GiB total, 90% headroom). Use a smaller model, or set its target to cpu."
        )

    def test_the_last_whole_byte_of_the_room_launches(self, tmp_path, monkeypatch) -> None:
        plan = plan_real_sizes(monkeypatch, tmp_path, ROOM_OF_24, 4096, 24, AUTO_CHAT_CPU_EMBED)
        assert WorkerRole.CHAT in _launched(plan)

    def test_both_sites_agree_on_the_same_bytes(self) -> None:
        over = ROOM_OF_24 + 1
        assert card_refusal("chat", 0, over, ROOM_OF_24 + 0.4, 24 * GIB, fix=True) is not None
        assert card_refusal("chat", 0, ROOM_OF_24, ROOM_OF_24 + 0.4, 24 * GIB, fix=True) is None


class TestTheText:
    def test_no_holder_keeps_the_text_of_a_lone_model(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
        assert card_refusal("m", 0, 22 * GIB, 21.6 * GIB, 24 * GIB, fix=True) == (
            "m needs 22.0 GiB but device 0 has 21.6 GiB usable "
            "(24.0 GiB total, 90% headroom). Use a smaller model, or set its target to cpu."
        )

    def test_two_holders_are_both_named_with_their_roles(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
        holders = (
            RoomHolder(WorkerRole.EMBED, "e", 4 * GIB),
            RoomHolder(WorkerRole.RERANK, "r", 3 * GIB),
        )
        reason = card_refusal("m", 0, 2 * GIB, 0.2 * GIB, 8 * GIB, fix=True, holders=holders)
        assert reason is not None
        assert (
            "The embed model e (4.0 GiB) and the rerank model r (3.0 GiB) hold the rest." in reason
        )
        assert "Use a smaller embed or rerank model" in reason

    def test_a_refusal_with_a_fallback_names_the_holder_and_offers_no_fix(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
        holders = (RoomHolder(WorkerRole.EMBED, "e", 4 * GIB),)
        assert card_refusal("m", 0, 5 * GIB, 3.2 * GIB, 8 * GIB, fix=False, holders=holders) == (
            "m needs 5.0 GiB but device 0 has 3.2 GiB usable (8.0 GiB total, 90% headroom). "
            "The embed model e (4.0 GiB) holds the rest."
        )


class TestEachSurfaceReadsTheRefusal:
    """The saved spec refuses the pinned chat; each surface shows the planner's text."""

    EXPECTED = (
        "chat pinned to device 0 needs 4.0 GiB but device 0 has 1.2 GiB usable "
        "(8.0 GiB total, 90% headroom). "
        f"The embed model {EMBED_REF} (6.0 GiB) holds the rest. "
        "Use a smaller embed model, or set the target of one of these models to cpu or "
        "another card."
    )

    @pytest.fixture
    def saved_refusal(self, tmp_path, monkeypatch) -> None:
        roles = (Role(WorkerRole.EMBED, 6), Role(WorkerRole.CHAT, 4), Role(WorkerRole.RERANK, 1))
        prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", PIN_BOTH_AND_CPU_RERANK)
        engine = CardEngine("engine-a", tmp_path / "engine-server", roles, cards_of((8,)))
        reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)

    @pytest.mark.usefixtures("saved_refusal")
    def test_the_cli_prints_the_text(self) -> None:
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert self.EXPECTED in " ".join(result.output.split())

    @pytest.mark.usefixtures("saved_refusal")
    def test_the_rest_route_returns_the_text(self) -> None:
        with create_test_client(route_handlers=[placement_route]) as client:
            response = client.get("/api/placement")
        assert response.status_code == 200
        assert response.json()["refusal"] == self.EXPECTED

    @pytest.mark.usefixtures("saved_refusal")
    def test_the_mcp_tool_returns_the_text(self) -> None:
        assert get_placement_tool()["refusal"] == self.EXPECTED

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("saved_refusal")
    async def test_the_tui_drawer_renders_the_text(self, monkeypatch) -> None:
        monkeypatch.setattr(gpu_fleet_panel, "probe_gpu_stats", lambda devices: {})
        app = _DrawerApp()
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.press("ctrl+g")

            def shown() -> str:
                found = app.screen.query("#placement-skipped")
                return " ".join(str(found.first().render()).split()) if found else ""

            assert await wait_until(pilot, lambda: self.EXPECTED in shown())


class TestWhoHoldsTheRoom:
    def test_only_one_entry_per_other_role_resident_on_that_card_is_named(self) -> None:
        chat = InstancePlan(WorkerRole.CHAT, (0,))
        placement = Placement(
            instances=(
                chat,
                InstancePlan(WorkerRole.VISION, (0,)),
                InstancePlan(WorkerRole.EMBED, (0,)),
                InstancePlan(WorkerRole.EMBED, (0,), replica=1),
                InstancePlan(WorkerRole.RERANK, (1,)),
            ),
            unplaceable_roles=(),
            co_tenants=frozenset({WorkerRole.CHAT, WorkerRole.VISION}),
        )
        charged = dict.fromkeys(WorkerRole, GIB)
        refs = {role: role.value for role in WorkerRole}
        assert planning._room_holders(chat, 0, placement, charged, refs) == (
            RoomHolder(WorkerRole.EMBED, "embed", GIB),
        )

    def test_a_role_outside_the_swap_group_is_named_though_a_group_exists(self) -> None:
        chat = InstancePlan(WorkerRole.CHAT, (0,))
        placement = Placement(
            instances=(chat, InstancePlan(WorkerRole.VISION, (0,))),
            unplaceable_roles=(),
            co_tenants=frozenset({WorkerRole.EMBED, WorkerRole.VISION}),
        )
        charged = dict.fromkeys(WorkerRole, GIB)
        refs = {role: role.value for role in WorkerRole}
        assert planning._room_holders(chat, 0, placement, charged, refs) == (
            RoomHolder(WorkerRole.VISION, "vision", GIB),
        )
