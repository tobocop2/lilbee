"""A chat role on a card and an embed role on the CPU, planned in one fleet plan."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, prepare_host

GIB = 1 << 30
CHAT_GIB = 10
EMBED_GIB = 6
ROLES = (Role(WorkerRole.CHAT, CHAT_GIB), Role(WorkerRole.EMBED, EMBED_GIB))
CHAT_ON_SECOND_CARD = '{"chat": {"devices": [1]}, "embed": {"target": "cpu"}}'
CUDA_VISIBLE = "CUDA_VISIBLE_DEVICES"


class TeiTargetEngine(CardEngine):
    """A sized test engine whose CPU target launches as the real TEI adapter does."""

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return TEI.target_env(target, cards)


def _plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, total_gib: int):
    prepare_host(monkeypatch, tmp_path, ROLES, total_gib * GIB)
    monkeypatch.setattr(cfg, "placement", CHAT_ON_SECOND_CARD)
    cards = cards_of((24, 24))
    chat_engine = CardEngine("chat-engine", tmp_path / "chat-server", ROLES, cards)
    embed_engine = TeiTargetEngine("embed-engine", tmp_path / "embed-server", ROLES, cards)
    engines = {WorkerRole.CHAT: chat_engine, WorkerRole.EMBED: embed_engine}
    return planning.plan_all_launches(engines=engines)


class TestChatOnACardBesideATeiCpuEmbed:
    def test_the_cpu_role_sees_no_card_and_chat_gets_its_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan = _plan(monkeypatch, tmp_path, 64)
        by_role = {launch.role: launch for launch in plan.launches}
        assert set(by_role) == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert by_role[WorkerRole.EMBED].env_overrides[CUDA_VISIBLE] == ""
        assert by_role[WorkerRole.CHAT].env_overrides[CUDA_VISIBLE] == "1"
        assert plan.refused_roles == {}

    def test_the_card_is_charged_for_chat_alone_and_the_cpu_role_for_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan = _plan(monkeypatch, tmp_path, 64)
        by_role = {launch.role: launch for launch in plan.launches}
        assert by_role[WorkerRole.CHAT].est_vram_bytes == CHAT_GIB * GIB
        assert by_role[WorkerRole.CHAT].est_vram_by_device == {"CUDA1": CHAT_GIB * GIB}
        assert by_role[WorkerRole.EMBED].est_vram_bytes == 0
        assert by_role[WorkerRole.EMBED].est_vram_by_device == {}

    def test_the_cpu_role_is_held_to_system_memory_and_chat_still_launches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The embed model needs 6 GiB of system memory; chat's 10 GiB sits on the card."""
        plan = _plan(monkeypatch, tmp_path, EMBED_GIB - 1)
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
        assert list(plan.refused_roles) == [WorkerRole.EMBED]
        assert "the CPU needs 6.0 GiB of system memory" in plan.refused_roles[WorkerRole.EMBED]
