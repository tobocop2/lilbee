import pytest

from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole


def test_round_trips_through_json():
    spec = PlacementSpec(
        {
            WorkerRole.CHAT: RolePlacement(devices=(0, 1), tensor_split=(60, 40)),
            WorkerRole.EMBED: RolePlacement(devices=(2,)),
        }
    )
    restored = PlacementSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.roles[WorkerRole.CHAT].tensor_split == (60, 40)
    assert restored.roles[WorkerRole.EMBED].replicas == 1


def test_str_is_json():
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))})
    assert str(spec) == spec.to_json()


def test_rejects_unknown_role():
    with pytest.raises(PlacementError, match="unknown role 'planner'"):
        PlacementSpec.from_json('{"planner": {"devices": [0]}}')


def test_rejects_empty_devices():
    with pytest.raises(PlacementError, match="chat: at least one device"):
        PlacementSpec.from_json('{"chat": {"devices": []}}')


def test_rejects_tensor_split_length_mismatch():
    with pytest.raises(PlacementError, match="chat: tensor_split has 3 weights for 2 devices"):
        PlacementSpec.from_json('{"chat": {"devices": [0, 1], "tensor_split": [1, 2, 3]}}')


def test_rejects_non_positive_replicas():
    with pytest.raises(PlacementError, match="embed: replicas must be >= 1"):
        PlacementSpec.from_json('{"embed": {"devices": [0], "replicas": 0}}')


def test_rejects_malformed_json():
    with pytest.raises(PlacementError, match="not valid JSON"):
        PlacementSpec.from_json("{not json")
