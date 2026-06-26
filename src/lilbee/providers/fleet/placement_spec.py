"""User-authored manual placement spec for the multi-GPU fleet."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from lilbee.providers.roles import WorkerRole

_KEY_DEVICES = "devices"
_KEY_TENSOR_SPLIT = "tensor_split"
_KEY_REPLICAS = "replicas"


class PlacementError(ValueError):
    """A placement spec is malformed or does not fit the hardware."""


@dataclass(frozen=True)
class RolePlacement:
    """One role's manual placement: device pins, optional split, replica count."""

    devices: tuple[int, ...]
    tensor_split: tuple[int, ...] | None = None
    replicas: int = 1


@dataclass(frozen=True)
class PlacementSpec:
    """A manual placement for every active role, keyed by role."""

    roles: Mapping[WorkerRole, RolePlacement] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a compact JSON string keyed by role value."""
        out: dict[str, dict[str, object]] = {}
        for role, rp in self.roles.items():
            entry: dict[str, object] = {_KEY_DEVICES: list(rp.devices)}
            if rp.tensor_split is not None:
                entry[_KEY_TENSOR_SPLIT] = list(rp.tensor_split)
            if rp.replicas != 1:
                entry[_KEY_REPLICAS] = rp.replicas
            out[role.value] = entry
        return json.dumps(out, sort_keys=True)

    def __str__(self) -> str:
        return self.to_json()

    @classmethod
    def from_json(cls, raw: str) -> PlacementSpec:
        """Parse a JSON string produced by ``to_json`` into a ``PlacementSpec``."""
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise PlacementError(f"placement is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise PlacementError("placement must be a JSON object keyed by role")
        roles: dict[WorkerRole, RolePlacement] = {}
        for key, entry in data.items():
            role = _role_for(key)
            roles[role] = _role_placement(role, entry)
        return cls(roles=roles)


_ALLOWED_KEYS = frozenset({_KEY_DEVICES, _KEY_TENSOR_SPLIT, _KEY_REPLICAS})


def _role_for(key: str) -> WorkerRole:
    try:
        return WorkerRole(key)
    except ValueError as exc:
        raise PlacementError(f"unknown role {key!r} in placement") from exc


def _coerce_int(role: WorkerRole, field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PlacementError(f"{role.value}: {field} must be integers, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise PlacementError(f"{role.value}: {field} must be integers, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlacementError(f"{role.value}: {field} must be integers, got {value!r}") from exc


def _int_list(role: WorkerRole, field: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise PlacementError(f"{role.value}: {field} must be a list, got {type(value).__name__}")
    return tuple(_coerce_int(role, field, item) for item in value)


def _role_placement(role: WorkerRole, entry: object) -> RolePlacement:
    if not isinstance(entry, dict):
        raise PlacementError(f"{role.value}: placement entry must be an object")
    unknown = set(entry) - _ALLOWED_KEYS
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_KEYS))
        raise PlacementError(
            f"{role.value}: unknown placement key(s) {sorted(unknown)}; allowed: {allowed}"
        )
    devices = _int_list(role, _KEY_DEVICES, entry.get(_KEY_DEVICES, []))
    if not devices:
        raise PlacementError(f"{role.value}: at least one device is required")
    if any(d < 0 for d in devices):
        raise PlacementError(f"{role.value}: device indices must be >= 0, got {list(devices)}")
    if len(set(devices)) != len(devices):
        raise PlacementError(f"{role.value}: duplicate device indices in {list(devices)}")
    raw_split = entry.get(_KEY_TENSOR_SPLIT)
    tensor_split: tuple[int, ...] | None = None
    if raw_split is not None:
        tensor_split = _int_list(role, _KEY_TENSOR_SPLIT, raw_split)
        if len(tensor_split) != len(devices):
            raise PlacementError(
                f"{role.value}: tensor_split has {len(tensor_split)} weights "
                f"for {len(devices)} devices"
            )
        if any(w <= 0 for w in tensor_split):
            raise PlacementError(
                f"{role.value}: tensor_split weights must be > 0, got {list(tensor_split)}"
            )
    replicas = _coerce_int(role, _KEY_REPLICAS, entry.get(_KEY_REPLICAS, 1))
    if replicas < 1:
        raise PlacementError(f"{role.value}: replicas must be >= 1")
    return RolePlacement(devices=devices, tensor_split=tensor_split, replicas=replicas)
