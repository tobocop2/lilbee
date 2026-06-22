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


def _role_for(key: str) -> WorkerRole:
    try:
        return WorkerRole(key)
    except ValueError as exc:
        raise PlacementError(f"unknown role {key!r} in placement") from exc


def _role_placement(role: WorkerRole, entry: object) -> RolePlacement:
    if not isinstance(entry, dict):
        raise PlacementError(f"{role.value}: placement entry must be an object")
    devices = tuple(int(d) for d in entry.get(_KEY_DEVICES, []))
    if not devices:
        raise PlacementError(f"{role.value}: at least one device is required")
    raw_split = entry.get(_KEY_TENSOR_SPLIT)
    tensor_split = tuple(int(w) for w in raw_split) if raw_split is not None else None
    if tensor_split is not None and len(tensor_split) != len(devices):
        raise PlacementError(
            f"{role.value}: tensor_split has {len(tensor_split)} weights for {len(devices)} devices"
        )
    replicas = int(entry.get(_KEY_REPLICAS, 1))
    if replicas < 1:
        raise PlacementError(f"{role.value}: replicas must be >= 1")
    return RolePlacement(devices=devices, tensor_split=tensor_split, replicas=replicas)
