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


def _proportional_peak(totals_gib):
    # Like the real estimator: a role's total bytes split proportionally to ratio.
    def estimate(role, ratio):
        total = totals_gib[role] * GIB
        s = sum(ratio)
        return tuple(int(total * r / s) for r in ratio)

    return estimate


def test_derives_planner_style_split_when_spec_has_none():
    # bb-lt7: three 48 GiB cards with the embedder resident on card 0. An even
    # chat split (~36.7 GiB per card) overflows card 0 (43.2 usable minus 8.5
    # embed = 34.7), so the even-split validator rejects the exact layout the
    # auto planner serves. A remaining-proportional split shrinks card 0's
    # shard and the same layout fits.
    spec = PlacementSpec(
        {
            WorkerRole.EMBED: RolePlacement(devices=(0,)),
            WorkerRole.CHAT: RolePlacement(devices=(0, 1, 2)),
        }
    )
    est = _proportional_peak({WorkerRole.CHAT: 110, WorkerRole.EMBED: 8.5})
    placement = placement_from_spec(
        spec,
        (WorkerRole.EMBED, WorkerRole.CHAT),  # planning registers non-chat roles first
        {0: 48 * GIB, 1: 48 * GIB, 2: 48 * GIB},
        estimate_peak=est,
    )
    assert placement.unplaceable_roles == ()
    chat = next(i for i in placement.instances if i.role is WorkerRole.CHAT)
    assert len(chat.tensor_split) == 3
    assert chat.tensor_split[0] < chat.tensor_split[1]  # co-resident card takes the smaller shard


def test_explicit_tensor_split_still_wins():
    # A spec that names its own split is honored verbatim, even when uneven.
    spec = PlacementSpec(
        {WorkerRole.CHAT: RolePlacement(devices=(0, 1), tensor_split=(3, 1))}
    )
    est = _proportional_peak({WorkerRole.CHAT: 40})
    placement = placement_from_spec(
        spec, (WorkerRole.CHAT,), {0: 80 * GIB, 1: 80 * GIB}, estimate_peak=est
    )
    chat = next(i for i in placement.instances if i.role is WorkerRole.CHAT)
    assert chat.tensor_split == (3, 1)
