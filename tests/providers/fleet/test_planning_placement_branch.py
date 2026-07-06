from pathlib import Path

from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan, Placement
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def test_read_device_cache_collapses_repeat_probes(monkeypatch):
    """A burst of reads within the TTL probes the engine once."""
    cache = planning._ReadDeviceCache(ttl_s=1000)
    calls = {"n": 0}

    def fake(_binary):
        calls["n"] += 1
        return [FleetDevice("CUDA", 0, "A", GIB, GIB)]

    monkeypatch.setattr(planning, "resolve_devices", fake)
    first = cache.get(Path("/x"))
    second = cache.get(Path("/x"))
    assert calls["n"] == 1
    assert first == second
    cache.clear()
    cache.get(Path("/x"))
    assert calls["n"] == 2  # cleared -> re-probe


def test_read_device_cache_ttl_zero_always_probes(monkeypatch):
    """A zero TTL disables caching (every read re-probes)."""
    cache = planning._ReadDeviceCache(ttl_s=0)
    calls = {"n": 0}

    def fake(_binary):
        calls["n"] += 1
        return []

    monkeypatch.setattr(planning, "resolve_devices", fake)
    cache.get(Path("/x"))
    cache.get(Path("/x"))
    assert calls["n"] == 2


def test_resolve_placement_plan_uses_read_cache(monkeypatch):
    """The read/view path serves repeat plans from the device cache (one probe)."""
    import lilbee.providers.fleet.cuda_runtime as cuda_runtime
    import lilbee.providers.fleet.gpu_env as gpu_env

    planning.clear_read_device_cache()
    calls = {"n": 0}

    def counting(_binary):
        calls["n"] += 1
        return []

    monkeypatch.setattr(planning, "resolve_llama_server", lambda: Path("/fake"))
    monkeypatch.setattr(gpu_env, "apply_fleet_gpu_env", lambda: None)
    monkeypatch.setattr(cuda_runtime, "apply_cuda_runtime_env", lambda: None)
    monkeypatch.setattr(planning, "resolve_devices", counting)
    monkeypatch.setattr(
        planning, "_server_model_inputs", lambda roles, *, unified_budget=None: ([], {}, 0)
    )
    monkeypatch.setattr(
        planning,
        "_resolve_placement",
        lambda *a, **k: Placement(instances=(), unplaceable_roles=()),
    )
    planning.resolve_placement_plan(None)
    planning.resolve_placement_plan(None)
    assert calls["n"] == 1
    planning.clear_read_device_cache()


def test_spec_branch_calls_placement_from_spec(monkeypatch):
    # free_bytes is far below total_bytes (a model is resident): placement must
    # still be charged against total capacity, not the warm free VRAM (bb-a8f).
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 10 * GIB)]
    captured = {}

    def fake_from_spec(spec, active_roles, device_capacity, *, estimate_peak):
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
        lambda roles, *, unified_budget=None: ([], {WorkerRole.CHAT: "ref"}, 0),
    )
    monkeypatch.setattr(planning, "_peak_estimator", lambda refs: lambda role, ratio: (GIB,))
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})

    placement = planning._resolve_placement(
        spec, [], {WorkerRole.CHAT: "ref"}, devices, unified_budget=None
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
    planning._resolve_placement(None, [], {}, devices, unified_budget=None)
    assert captured["devs"] == [(0, 80 * GIB)]  # total, not the 5 GiB free


def test_no_spec_uses_auto_planner(monkeypatch):
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    called = {}
    monkeypatch.setattr(
        planning,
        "plan_placement",
        lambda inputs, devs, *, estimate_peak, unified_budget, **_kw: (
            called.setdefault("auto", True) or Placement(instances=(), unplaceable_roles=())
        ),
    )
    planning._resolve_placement(None, [], {}, devices, unified_budget=None)
    assert called.get("auto") is True
