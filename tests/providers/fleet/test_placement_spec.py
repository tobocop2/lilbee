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


def test_to_json_includes_replicas_when_not_one():
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,), replicas=2)})
    import json

    data = json.loads(spec.to_json())
    assert data["embed"]["replicas"] == 2


def test_rejects_json_array_at_top_level():
    with pytest.raises(PlacementError, match="must be a JSON object keyed by role"):
        PlacementSpec.from_json("[1, 2, 3]")


def test_rejects_non_dict_role_entry():
    with pytest.raises(PlacementError, match="placement entry must be an object"):
        PlacementSpec.from_json('{"chat": [0, 1]}')


def test_rejects_duplicate_devices():
    with pytest.raises(PlacementError, match="duplicate device indices"):
        PlacementSpec.from_json('{"chat": {"devices": [0, 0]}}')


def test_rejects_negative_device_index():
    with pytest.raises(PlacementError, match="device indices must be >= 0"):
        PlacementSpec.from_json('{"chat": {"devices": [-1]}}')


def test_rejects_non_positive_tensor_split_weight():
    with pytest.raises(PlacementError, match="tensor_split weights must be > 0"):
        PlacementSpec.from_json('{"chat": {"devices": [0, 1], "tensor_split": [0, 1]}}')


def test_rejects_negative_tensor_split_weight():
    with pytest.raises(PlacementError, match="tensor_split weights must be > 0"):
        PlacementSpec.from_json('{"chat": {"devices": [0, 1], "tensor_split": [1, -2]}}')


def test_rejects_unknown_entry_key():
    with pytest.raises(PlacementError, match="unknown placement key"):
        PlacementSpec.from_json('{"chat": {"devices": [0], "tensor-split": [1]}}')


def test_rejects_non_integer_device():
    with pytest.raises(PlacementError, match="devices must be integers"):
        PlacementSpec.from_json('{"chat": {"devices": ["bad"]}}')


def test_rejects_non_list_devices():
    with pytest.raises(PlacementError, match="devices must be a list"):
        PlacementSpec.from_json('{"chat": {"devices": 5}}')


def test_rejects_fractional_float_device():
    with pytest.raises(PlacementError, match="devices must be integers"):
        PlacementSpec.from_json('{"chat": {"devices": [1.9]}}')


def test_accepts_integral_float_device():
    spec = PlacementSpec.from_json('{"chat": {"devices": [2.0]}}')
    assert spec.roles[WorkerRole.CHAT].devices == (2,)


def test_rejects_boolean_device():
    # bool is an int subclass; without the guard ``true`` would silently pin device 1.
    with pytest.raises(PlacementError, match="devices must be integers"):
        PlacementSpec.from_json('{"chat": {"devices": [true]}}')


def test_rejects_boolean_replicas():
    with pytest.raises(PlacementError, match="replicas must be integers"):
        PlacementSpec.from_json('{"embed": {"devices": [0], "replicas": true}}')
