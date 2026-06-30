"""The launch spec for one llama-server instance, shared by planning and llama-swap."""

from __future__ import annotations

from dataclasses import dataclass, field

from lilbee.providers.roles import RerankMode, WorkerRole

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
    token_cap: int | None = None  # per-slot ctx for embed/rerank input truncation
    weights_bytes: int = 0  # model file size on disk; scales the cold-load timeout
    slots: int = 1  # --parallel continuous-batching slots; chat concurrency capacity
    ctx: int = 0  # per-slot context the server runs with; what a client should fit to
    replica: int = 0  # index within the role's data-parallel pool (0 = single server)
    rerank_mode: RerankMode | None = None  # set only for RERANK; picks the client scoring path
    # Estimated VRAM (bytes) this instance occupies per device index; empty off-GPU.
    # The provider sums these across the resident fleet to credit its own residency
    # back to the device probe on a reload (so the chat split is sized cold).
    device_vram: dict[int, int] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """The llama-swap model id for this instance: ``<role>-<replica>``."""
        return f"{role_model_prefix(self.role)}{self.replica}"
