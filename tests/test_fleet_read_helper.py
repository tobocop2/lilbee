"""The fleet reads each role's engine from the engine book; only a missing engine is absorbed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import placement
from lilbee.app.engine_choice import EngineBook
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import GIB, SizedEngine, one_machine

_MLX_REF = "org/repo#mlx"


class _BundledLlama(SizedEngine):
    """A bundled engine that loads gguf files and sizes every model at two GiB."""

    bundled = True
    formats = frozenset({"gguf"})
    served_roles = frozenset(WorkerRole)


@pytest.fixture
def chat_without_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _BundledLlama:
    """A host where only the gguf engine is installed and the chat model is an mlx model."""
    device = one_machine(monkeypatch, tmp_path, unified=False, total=24 * GIB)
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"a")
    engine = _BundledLlama("llama.cpp", binary, MemoryKind.SIZED, 2 * GIB, device)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, engine)  # type: ignore[arg-type]
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX, raising=False)
    monkeypatch.setattr(
        runnable,
        "session_host",
        lambda: collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None),
    )
    monkeypatch.setattr(cfg, "chat_model", _MLX_REF)
    return engine


class TestARoleWithNoEngineIsLeftOut:
    def test_the_provider_plans_the_role_that_has_an_engine_and_skips_the_other(
        self, chat_without_engine: _BundledLlama
    ) -> None:
        engines = FleetProvider()._engines()
        assert engines[WorkerRole.EMBED] is chat_without_engine
        assert WorkerRole.CHAT not in engines

        plan = planning.plan_all_launches(engines=engines)

        launched = {launch.role for launch in plan.launches}
        assert WorkerRole.EMBED in launched
        assert WorkerRole.CHAT not in launched
        assert plan.skipped_no_engine == {WorkerRole.CHAT: _MLX_REF}

    def test_the_placement_read_reports_the_role_that_has_no_engine(
        self, chat_without_engine: _BundledLlama
    ) -> None:
        view = placement.get_placement()

        assert [(skipped.role, skipped.model) for skipped in view.skipped_no_engine] == [
            (WorkerRole.CHAT, _MLX_REF)
        ]
        assert "No installed engine loads the mlx format" in view.skipped_no_engine[0].notice
        placed = {role.role for role in view.roles}
        assert WorkerRole.EMBED in placed
        assert WorkerRole.CHAT not in placed


class TestEveryPlacementReaderLeavesARoleOutInsteadOfFailing:
    """Each reader of the engine book reports the role with no engine; none raises for it."""

    def test_a_preview_reports_the_role_that_has_no_engine(
        self, chat_without_engine: _BundledLlama
    ) -> None:
        view = placement.preview_placement()

        assert [skipped.role for skipped in view.skipped_no_engine] == [WorkerRole.CHAT]
        assert WorkerRole.EMBED in {role.role for role in view.roles}

    def test_a_saved_placement_reports_the_role_that_has_no_engine(
        self, chat_without_engine: _BundledLlama, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(placement, "peek_services", lambda: None)

        view = placement.set_placement(None)

        assert [skipped.role for skipped in view.skipped_no_engine] == [WorkerRole.CHAT]
        assert WorkerRole.EMBED in {role.role for role in view.roles}

    def test_the_size_notice_of_the_role_that_has_an_engine_is_kept(
        self, chat_without_engine: _BundledLlama, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sized = chat_without_engine.estimate_memory
        monkeypatch.setattr(
            chat_without_engine,
            "estimate_memory",
            lambda request: replace(sized(request), size_known=False),
        )

        notices = placement.size_unknown_notices(WorkerRole.EMBED)

        assert len(notices) == 1
        assert "embed model" in notices[0]

    def test_the_host_engine_is_the_engine_of_the_role_that_has_one(
        self, chat_without_engine: _BundledLlama
    ) -> None:
        provider = FleetProvider()

        assert provider._host_engine() is chat_without_engine
        assert provider._unusable_engine_reason() is None


def _broken_engine(error: Exception) -> property:
    """An engine whose adapter cannot be built, for every role."""

    def adapter(self: Any) -> None:
        raise error

    return property(adapter)


def _read_book() -> object:
    return EngineBook(cfg).read()


def _read_provider() -> object:
    return FleetProvider()._engines()


def _read_placement() -> object:
    return placement.get_placement()


class TestAFailureOtherThanAMissingEngineSurfaces:
    @pytest.mark.parametrize(
        "error",
        [
            ProviderError("the engine module failed to load"),
            RuntimeError("the engine module failed to load"),
            OSError("the engine module failed to load"),
        ],
        ids=["provider-error", "runtime-error", "os-error"],
    )
    @pytest.mark.parametrize(
        "read",
        [_read_book, _read_provider, _read_placement],
        ids=["book", "provider", "placement"],
    )
    def test_a_broken_engine_is_not_read_as_a_missing_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        chat_without_engine: _BundledLlama,
        read: Callable[[], object],
        error: Exception,
    ) -> None:
        monkeypatch.setattr(choice.RoleEngine, "adapter", _broken_engine(error))

        with pytest.raises(type(error), match="the engine module failed to load"):
            read()


class TestThePlanSearchKeepsItsEstimates:
    def test_every_packing_pass_reuses_the_size_of_a_split_it_has_asked_for(self) -> None:
        asked: Counter[tuple[WorkerRole, tuple[int, ...]]] = Counter()

        def estimate(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
            asked[(role, ratio)] += 1
            return tuple(6 * GIB for _ in ratio)

        placed = plan_placement(
            [
                ModelPlacementInput(role=role, est_vram_bytes=12 * GIB)
                for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION)
            ],
            [(0, 8 * GIB), (1, 8 * GIB)],
            estimate_peak=estimate,
            refuse_oversize=True,
        )

        assert placed.unplaceable_roles
        assert len(asked) > 0
        assert max(asked.values()) == 1
