import pytest

from lilbee.providers.fleet.placement import InstancePlan, placement_from_spec
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def _peak(per_device_gib):
    # estimate_peak(role, ratio) -> per-device byte vector aligned to ratio length.
    def estimate(role, ratio):
        return tuple(per_device_gib[role][i] * GIB for i in range(len(ratio)))

    return estimate


def test_builds_plans_from_spec():
    spec = PlacementSpec(
        {
            WorkerRole.CHAT: RolePlacement(devices=(0, 1), tensor_split=(1, 1)),
            WorkerRole.EMBED: RolePlacement(devices=(2,)),
        }
    )
    est = _peak({WorkerRole.CHAT: [30, 30], WorkerRole.EMBED: [4]})
    placement = placement_from_spec(
        spec,
        (WorkerRole.CHAT, WorkerRole.EMBED),
        {0: 80 * GIB, 1: 80 * GIB, 2: 80 * GIB},
        estimate_peak=est,
    )
    assert placement.unplaceable_roles == ()
    assert (
        InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        in placement.instances
    )
    assert InstancePlan(role=WorkerRole.EMBED, devices=(2,)) in placement.instances


def test_errors_when_active_role_missing_from_spec():
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
    with pytest.raises(PlacementError, match="embed has a model but no placement entry"):
        placement_from_spec(
            spec,
            (WorkerRole.CHAT, WorkerRole.EMBED),
            {0: 80 * GIB},
            estimate_peak=_peak({WorkerRole.CHAT: [4]}),
        )


def test_errors_on_nonexistent_device():
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(5,))})
    with pytest.raises(PlacementError, match="chat pinned to device 5 but only 1 GPU"):
        placement_from_spec(
            spec, (WorkerRole.CHAT,), {0: 80 * GIB}, estimate_peak=_peak({WorkerRole.CHAT: [4]})
        )


def test_errors_when_does_not_fit_naming_card():
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
    est = _peak({WorkerRole.CHAT: [70]})
    with pytest.raises(
        PlacementError,
        match=r"chat .* device 0 .* 70.0 GiB .* 36.0 GiB usable .*40.0 GiB total",
    ):
        placement_from_spec(spec, (WorkerRole.CHAT,), {0: 40 * GIB}, estimate_peak=est)


def test_errors_when_two_roles_overbook_one_card():
    spec = PlacementSpec(
        {
            WorkerRole.CHAT: RolePlacement(devices=(0,)),
            WorkerRole.EMBED: RolePlacement(devices=(0,)),
        }
    )
    est = _peak({WorkerRole.CHAT: [50], WorkerRole.EMBED: [40]})
    with pytest.raises(PlacementError, match="device 0"):
        placement_from_spec(
            spec, (WorkerRole.CHAT, WorkerRole.EMBED), {0: 80 * GIB}, estimate_peak=est
        )
