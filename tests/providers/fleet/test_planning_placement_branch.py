import logging
from pathlib import Path

import pytest

from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.placement import InstancePlan, Placement
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import EngineBackend, WorkerRole
from tests._fleet_engines import engines_for_all_roles

GIB = 1024**3


def test_read_device_cache_collapses_repeat_probes(monkeypatch):
    """A burst of reads within the TTL probes the engine once."""
    cache = planning._ReadDeviceCache(ttl_s=1000, failure_ttl_s=1000)
    calls = {"n": 0}

    def fake(_binary, engine):
        calls["n"] += 1
        return planning.DeviceReading([FleetDevice("CUDA", 0, "A", GIB, GIB)], EngineBackend.CUDA)

    monkeypatch.setattr(planning, "_read_devices", fake)
    first = cache.get(Path("/x"), engine=LLAMA)
    second = cache.get(Path("/x"), engine=LLAMA)
    assert calls["n"] == 1
    assert first == second
    cache.clear()
    cache.get(Path("/x"), engine=LLAMA)
    assert calls["n"] == 2  # cleared -> re-probe


def test_read_device_cache_ttl_zero_always_probes(monkeypatch):
    """A zero TTL disables caching (every read re-probes)."""
    cache = planning._ReadDeviceCache(ttl_s=0, failure_ttl_s=0)
    calls = {"n": 0}

    def fake(_binary, engine):
        calls["n"] += 1
        return planning.DeviceReading([], EngineBackend.CPU)

    monkeypatch.setattr(planning, "_read_devices", fake)
    cache.get(Path("/x"), engine=LLAMA)
    cache.get(Path("/x"), engine=LLAMA)
    assert calls["n"] == 2


def test_resolve_placement_plan_uses_read_cache(monkeypatch):
    """The read/view path serves repeat plans from the device cache (one probe)."""

    planning.clear_read_device_cache()
    calls = {"n": 0}

    def counting(_binary, engine):
        calls["n"] += 1
        return planning.DeviceReading([], EngineBackend.CPU)

    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: Path("/fake")
    )
    monkeypatch.setattr(planning, "_read_devices", counting)
    monkeypatch.setattr(
        planning,
        "_server_model_inputs",
        lambda roles, *, engines, **_k: (
            [],
            {},
            0,
            {},
        ),
    )
    monkeypatch.setattr(
        planning,
        "_resolve_placement",
        lambda *a, **k: Placement(instances=(), unplaceable_roles=()),
    )
    planning.resolve_placement_plan(None, engines=engines_for_all_roles(LLAMA))
    planning.resolve_placement_plan(None, engines=engines_for_all_roles(LLAMA))
    assert calls["n"] == 1
    planning.clear_read_device_cache()


def test_spec_branch_calls_placement_from_spec(monkeypatch):
    # free_bytes is far below total_bytes (a model is resident): placement must
    # still be charged against total capacity, not the warm free VRAM (bb-a8f).
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 10 * GIB)]
    captured = {}

    def fake_from_spec(
        spec,
        active_roles,
        device_capacity,
        *,
        estimate_peak,
        plan_auto,
        model_refs,
        lost_card,
        outside,
    ):
        captured["spec"] = spec
        captured["roles"] = active_roles
        captured["capacity"] = device_capacity
        return Placement(
            instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0,)),), unplaceable_roles=()
        )

    monkeypatch.setattr(planning, "placement_from_spec", fake_from_spec)
    monkeypatch.setattr(
        planning,
        "_server_model_inputs",
        lambda roles, *, unified_budget=None, engine: ([], {WorkerRole.CHAT: "ref"}, 0, {}),
    )
    monkeypatch.setattr(
        planning, "_peak_estimator", lambda refs, engine: lambda role, ratio: (GIB,)
    )
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})

    placement = planning._resolve_placement(
        spec,
        [],
        {WorkerRole.CHAT: "ref"},
        devices,
        unified_budget=None,
        host_cards=planning.HostCards(devices),
        engines=engines_for_all_roles(LLAMA),
    )

    assert captured["spec"] is spec
    assert captured["roles"] == (WorkerRole.CHAT,)
    assert captured["capacity"] == {0: 80 * GIB}  # total, not the 10 GiB free
    assert placement.instances[0].role is WorkerRole.CHAT


def test_auto_planner_charged_against_total_capacity(monkeypatch):
    # The auto planner is fed total capacity too, so a warm fleet's resident
    # models are not double-counted when it re-plans (bb-a8f).
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 5 * GIB)]
    captured = {}

    def fake_plan(inputs, devs, *, estimate_peak, unified_budget, **_kw):
        captured["devs"] = devs
        return Placement(instances=(), unplaceable_roles=())

    monkeypatch.setattr(planning, "plan_placement", fake_plan)
    planning._resolve_placement(
        None,
        [],
        {},
        devices,
        unified_budget=None,
        host_cards=planning.HostCards(devices),
        engines=engines_for_all_roles(LLAMA),
    )
    assert captured["devs"] == [(0, 80 * GIB)]  # total, not the 5 GiB free


def test_saved_spec_that_no_longer_fits_falls_back_to_auto(monkeypatch, caplog):
    """A pin naming a GPU this host no longer has must not take the fleet down."""
    devices = [FleetDevice("Vulkan", 0, "Iris Xe", 8 * GIB, 8 * GIB)]
    auto = Placement(
        instances=(InstancePlan(role=WorkerRole.EMBED, devices=(0,)),), unplaceable_roles=()
    )

    def refuse(spec, *_a, **_kw):
        if spec is None:
            return auto
        raise PlacementError("embed is pinned to device 1, but this machine lists only device 0.")

    monkeypatch.setattr(planning, "_resolve_placement", refuse)
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(1,))})

    with caplog.at_level(logging.WARNING):
        placement, spec_applied, why = planning._placement_or_auto(
            spec,
            [],
            {WorkerRole.EMBED: "ref"},
            devices,
            unified_budget=None,
            host_cards=planning.HostCards(devices),
            engines=engines_for_all_roles(LLAMA),
        )

    assert placement is auto
    assert spec_applied is False
    assert why == "embed is pinned to device 1, but this machine lists only device 0."
    assert "does not fit this hardware" in caplog.text


def test_saved_spec_that_fits_is_applied(monkeypatch):
    """A pin the hardware still satisfies is used, and reports itself as applied."""
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    manual = Placement(
        instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0,)),), unplaceable_roles=()
    )
    monkeypatch.setattr(planning, "_resolve_placement", lambda spec, *_a, **_kw: manual)
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})

    placement, spec_applied, why = planning._placement_or_auto(
        spec,
        [],
        {WorkerRole.CHAT: "ref"},
        devices,
        unified_budget=None,
        host_cards=planning.HostCards(devices),
        engines=engines_for_all_roles(LLAMA),
    )

    assert placement is manual
    assert spec_applied is True
    assert why == ""


def test_no_saved_spec_is_not_reported_as_applied(monkeypatch):
    """With nothing saved the plan is the auto planner's, so spec_applied is False."""
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    auto = Placement(instances=(), unplaceable_roles=())
    monkeypatch.setattr(planning, "_resolve_placement", lambda *_a, **_kw: auto)

    placement, spec_applied, why = planning._placement_or_auto(
        None,
        [],
        {},
        devices,
        unified_budget=None,
        host_cards=planning.HostCards(devices),
        engines=engines_for_all_roles(LLAMA),
    )

    assert placement is auto
    assert spec_applied is False
    assert why == ""


def test_interactive_resolve_still_raises(monkeypatch):
    """Without the fallback opt-in an unfittable spec raises, so an apply fails loud."""

    planning.clear_read_device_cache()
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: Path("/fake")
    )
    monkeypatch.setattr(
        planning, "_read_devices", lambda _b, engine: planning.DeviceReading([], EngineBackend.CPU)
    )
    monkeypatch.setattr(
        planning,
        "_server_model_inputs",
        lambda roles, *, engines, **_k: (
            [],
            {},
            0,
            {},
        ),
    )

    def refuse(*_a, **_kw):
        raise PlacementError("embed is pinned to device 0, but this machine lists no devices.")

    monkeypatch.setattr(planning, "_resolve_placement", refuse)
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))})

    with pytest.raises(PlacementError):
        planning.resolve_placement_plan(spec, engines=engines_for_all_roles(LLAMA))
    planning.clear_read_device_cache()


def test_read_path_reports_the_plan_it_fell_back_to(monkeypatch):
    """The view path opts into the fallback and marks the spec as not applied."""

    planning.clear_read_device_cache()
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: Path("/fake")
    )
    monkeypatch.setattr(
        planning, "_read_devices", lambda _b, engine: planning.DeviceReading([], EngineBackend.CPU)
    )
    monkeypatch.setattr(
        planning,
        "_server_model_inputs",
        lambda roles, *, engines, **_k: (
            [],
            {},
            0,
            {},
        ),
    )

    def refuse(spec, *_a, **_kw):
        if spec is None:
            return Placement(instances=(), unplaceable_roles=())
        raise PlacementError("embed is pinned to device 0, but this machine lists no devices.")

    monkeypatch.setattr(planning, "_resolve_placement", refuse)
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))})

    resolved = planning.resolve_placement_plan(
        spec, fall_back_to_auto=True, engines=engines_for_all_roles(LLAMA)
    )

    assert resolved.spec_applied is False
    planning.clear_read_device_cache()


def test_no_spec_uses_auto_planner(monkeypatch):
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    called = {}

    def fake_plan(inputs, devs, *, estimate_peak, unified_budget, **_kw):
        called["auto"] = True
        return Placement(instances=(), unplaceable_roles=())

    monkeypatch.setattr(planning, "plan_placement", fake_plan)
    planning._resolve_placement(
        None,
        [],
        {},
        devices,
        unified_budget=None,
        host_cards=planning.HostCards(devices),
        engines=engines_for_all_roles(LLAMA),
    )
    assert called.get("auto") is True


def test_a_tight_role_reaches_the_placement_view(monkeypatch, tmp_path) -> None:
    """The planner has always known a role does not fit and only logged it.

    A surface reading ResolvedPlacement saw it as comfortably placed, so the one
    number that predicts a failed load never left the log.
    """
    from lilbee.app.placement import _view
    from lilbee.providers.fleet.devices import FleetDevice
    from lilbee.providers.fleet.placement import InstancePlan
    from lilbee.providers.fleet.planning import ResolvedPlacement
    from lilbee.providers.roles import WorkerRole

    resolved = ResolvedPlacement(
        devices=(FleetDevice("CUDA", 0, "gpu", 24 * 1024**3, 24 * 1024**3),),
        instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0,)),),
        unplaceable_roles=(),
        model_refs={WorkerRole.CHAT: "org/chat.gguf"},
        tight_roles={WorkerRole.CHAT: 3 * 1024**3},
    )
    view = _view(resolved, engines_for_all_roles(LLAMA), manual=False, spec_json=None)
    assert [(t.role, t.shortfall_bytes) for t in view.tight] == [(WorkerRole.CHAT, 3 * 1024**3)]


def test_the_tight_sentence_scales_by_a_binary_gib_and_never_shows_zero():
    from lilbee.providers.fleet.model_memory import tight_memory_text

    assert "estimated to need 1000.0 GiB more GPU memory" in tight_memory_text(
        "the chat model", 1000 * 1024**3
    )
    assert "estimated to need 0.1 GiB more GPU memory" in tight_memory_text("the chat model", 1)


def test_memory_size_text_uses_mib_below_one_gib_and_gib_from_one() -> None:
    from lilbee.providers.fleet.model_memory import memory_size_text

    assert memory_size_text(40 * 1024**2) == "40 MiB"
    assert memory_size_text(1024**3 - 1) == "1024 MiB"
    assert memory_size_text(1024**3) == "1.0 GiB"
    assert memory_size_text(int(7.25 * 1024**3)) == "7.2 GiB"


def test_memory_size_text_never_reads_zero_for_a_size_above_zero() -> None:
    from lilbee.providers.fleet.model_memory import memory_size_text

    assert memory_size_text(0) == "0 MiB"
    assert memory_size_text(1) == "under 1 MiB"
    assert memory_size_text(1024**2 // 2) == "under 1 MiB"
    assert memory_size_text(1024**2 - 1) == "under 1 MiB"
    assert memory_size_text(1024**2) == "1 MiB"


def test_the_planner_logs_the_tight_sentence_and_the_co_tenant_line_as_written(caplog):
    from lilbee.providers.fleet.model_memory import tight_memory_text

    placement = Placement(
        instances=(),
        unplaceable_roles=(),
        swap_classes=(frozenset({WorkerRole.EMBED, WorkerRole.CHAT}),),
        tight_roles={WorkerRole.CHAT: 2 * 1024**3},
    )

    with caplog.at_level(logging.INFO, logger="lilbee.providers.fleet.planning"):
        planning._log_placement_findings(placement, {WorkerRole.CHAT: "org/chat.gguf"})

    assert [r.getMessage() for r in caplog.records] == [
        tight_memory_text("the chat model org/chat.gguf", 2 * 1024**3),
        "chat + embed: share memory, one group loaded at a time",
    ]
