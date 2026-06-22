from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan, Placement
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def test_spec_branch_calls_placement_from_spec(monkeypatch):
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    captured = {}

    def fake_from_spec(spec, active_roles, device_free, *, estimate_peak):
        captured["spec"] = spec
        captured["roles"] = active_roles
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
    assert placement.instances[0].role is WorkerRole.CHAT


def test_no_spec_uses_auto_planner(monkeypatch):
    devices = [FleetDevice("CUDA", 0, "A", 80 * GIB, 80 * GIB)]
    called = {}
    monkeypatch.setattr(
        planning,
        "plan_placement",
        lambda inputs, devs, *, estimate_peak, unified_budget: (
            called.setdefault("auto", True) or Placement(instances=(), unplaceable_roles=())
        ),
    )
    planning._resolve_placement(None, [], {}, devices, unified_budget=None)
    assert called.get("auto") is True
