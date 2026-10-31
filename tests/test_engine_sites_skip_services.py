"""Tests that placement, the launcher and self-check pick the engine without services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lilbee.app import placement
from lilbee.app import services as services_mod
from lilbee.cli.commands import setup as setup_cmd
from lilbee.cli.launchers import server as launcher
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.placement_spec import PlacementError
from lilbee.providers.roles import WorkerRole


@pytest.fixture(autouse=True)
def _opening_a_store_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test when the services container is built, since that opens the store.

    With no singleton set, every ``get_services`` call reaches ``build_services``.
    """

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the services container was built to pick an engine")

    monkeypatch.setattr(services_mod, "build_services", _forbidden)
    monkeypatch.setattr(services_mod._state, "singleton", None)


def test_placement_views_plan_with_the_selected_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def _resolve(spec: object, *, engine: Any, **kwargs: object) -> Any:
        seen.append(engine)
        raise PlacementError("stop after the engine is chosen")

    monkeypatch.setattr(placement, "resolve_placement_plan", _resolve)
    for call in (placement.get_placement, placement.preview_placement):
        with pytest.raises(PlacementError, match="stop after the engine is chosen"):
            call()
    with pytest.raises(PlacementError, match="stop after the engine is chosen"):
        placement.set_placement(None)
    assert seen == [LLAMA, LLAMA, LLAMA]


def test_launcher_sizes_the_chat_window_with_the_selected_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lilbee.providers.engine_params as engine_params
    import lilbee.providers.fleet.planning as planning
    import lilbee.providers.gguf_meta as gguf_meta

    seen: list[Any] = []
    monkeypatch.setattr(cfg, "chat_model", "local/model.gguf")
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: tmp_path / "m.gguf")
    monkeypatch.setattr(gguf_meta, "read_gguf_metadata", lambda path: None)
    monkeypatch.setattr(planning, "plan_sizing_budget", lambda engine: 1)

    def _ctx(path: Path, meta: object, *, engine: Any, available_bytes: int) -> int:
        seen.append(engine)
        return 4096

    monkeypatch.setattr(engine_params, "resolve_chat_ctx", _ctx)
    assert launcher.planned_chat_ctx() == 4096
    assert seen == [LLAMA]


def test_self_check_kv_flag_uses_the_selected_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    import lilbee.providers.fleet.planning as planning

    seen: list[Any] = []

    def _flag(engine: Any) -> str:
        seen.append(engine)
        return "flag-from-the-spy"

    monkeypatch.setattr(planning, "chat_cache_type_flag", _flag)
    kwargs = setup_cmd._resolved_provider_kwargs()
    assert kwargs["kv_cache_type"] == "flag-from-the-spy"
    assert seen == [LLAMA]


def test_self_check_server_plans_with_the_selected_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lilbee.providers.fleet.planning as planning
    import lilbee.providers.fleet.swap_manager as swap_manager

    seen: list[Any] = []

    class _Launch:
        model_id = "m"

    def _plan(role: object, path: object, *, engine: Any) -> _Launch:
        seen.append(engine)
        return _Launch()

    class _Swap:
        def __init__(self, work_dir: Path, group: object, engine: Any) -> None:
            seen.append(engine)

        def start(self, launches: list[Any]) -> None:
            return None

        def endpoint(self) -> str:
            return "http://127.0.0.1:1"

    monkeypatch.setattr(planning, "build_single_role_launch", _plan)
    monkeypatch.setattr(swap_manager, "SwapManager", _Swap)
    swap, _client, work_dir = setup_cmd._self_check_server(WorkerRole.EMBED, tmp_path / "m.gguf")
    try:
        assert isinstance(swap, _Swap)
        assert seen == [LLAMA, LLAMA]
    finally:
        work_dir.rmdir()
