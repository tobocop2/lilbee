import pytest

from lilbee.app import placement as app_placement
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def _resolved():
    return ResolvedPlacement(
        devices=(
            FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),
            FleetDevice("CUDA", 1, "NVIDIA A100", 80 * GIB, 80 * GIB),
        ),
        instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1)),),
        unplaceable_roles=(),
        model_refs={WorkerRole.CHAT: "org/chat.gguf"},
    )


def test_active_spec_reads_cfg(monkeypatch):
    monkeypatch.setattr(app_placement.cfg, "placement", None)
    assert app_placement._active_spec() is None


def test_get_placement_renders_view(monkeypatch):
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    view = app_placement.get_placement()
    assert view.manual is False
    assert view.gpus[0].label == "CUDA0"
    assert view.gpus[0].free_bytes == 72 * GIB
    assert view.roles[0].role is WorkerRole.CHAT
    assert view.roles[0].devices == (0, 1)


def test_preview_passes_candidate_spec(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        app_placement,
        "resolve_placement_plan",
        lambda spec: seen.update({"spec": spec}) or _resolved(),
    )
    cand = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
    app_placement.preview_placement(cand)
    assert seen["spec"] is cand


def test_preview_has_no_side_effects(monkeypatch):
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    reset = {"called": False}
    monkeypatch.setattr(app_placement, "reset_services", lambda: reset.__setitem__("called", True))
    app_placement.preview_placement(None)
    assert reset["called"] is False


def test_set_persists_and_resets(monkeypatch):
    writes = {}
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(app_placement.settings, "update_values", lambda root, d: writes.update(d))
    monkeypatch.setattr(app_placement, "reset_services", lambda: writes.setdefault("reset", True))
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0, 1), tensor_split=(1, 1))})
    app_placement.set_placement(spec)
    assert writes["placement"] == spec.to_json()
    assert writes["reset"] is True


def test_set_none_clears(monkeypatch):
    deletes = {}
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(
        app_placement.settings, "delete_values", lambda root, keys: deletes.setdefault("keys", keys)
    )
    monkeypatch.setattr(app_placement, "reset_services", lambda: None)
    app_placement.set_placement(None)
    assert deletes["keys"] == ["placement"]


def test_set_validates_before_persist(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    def boom(spec):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(app_placement, "resolve_placement_plan", boom)
    wrote = {"any": False}
    monkeypatch.setattr(
        app_placement.settings, "update_values", lambda root, d: wrote.__setitem__("any", True)
    )
    with pytest.raises(PlacementError):
        app_placement.set_placement(PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))}))
    assert wrote["any"] is False
