"""The launch spec for one llama-server instance, shared by planning and llama-swap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lilbee.providers.roles import WorkerRole

# Separator between a role and its replica index in a llama-swap model id.
_REPLICA_SEP = "-"


def role_model_prefix(role: WorkerRole) -> str:
    """Prefix shared by every replica model id of *role* (``<role>-``)."""
    return f"{role.value}{_REPLICA_SEP}"


@dataclass
class InstanceLaunch:
    """Everything needed to run one llama-server, minus the port (claimed at spawn)."""

    role: WorkerRole
    argv: list[str]  # llama-server command WITHOUT --port; the runner appends it
    env_overrides: dict[str, str]  # backend-specific device-pinning env
    model: str
    port_file: Path
    token_cap: int | None = None  # per-slot ctx for embed/rerank input truncation
    weights_bytes: int = 0  # model file size on disk; scales the cold-load timeout
    slots: int = 1  # --parallel continuous-batching slots; chat concurrency capacity
    ctx: int = 0  # per-slot context the server runs with; what a client should fit to
    replica: int = 0  # index within the role's data-parallel pool (0 = single server)

    @property
    def model_id(self) -> str:
        """The llama-swap model id for this instance: ``<role>-<replica>``."""
        return f"{role_model_prefix(self.role)}{self.replica}"
