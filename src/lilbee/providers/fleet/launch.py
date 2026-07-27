"""The launch spec for one llama-server instance, shared by planning and llama-swap."""

from __future__ import annotations

from dataclasses import dataclass

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
    # GPU bytes placement charged this instance. Carried so the engine's own
    # startup report can be checked against it once the server is up; 0 for a
    # model the estimator could not size, where there is nothing to compare.
    est_vram_bytes: int = 0

    def to_state(self) -> dict:
        """JSON-safe form written into every engine state file so a guest lilbee
        can read the serving contract and bind."""
        return {
            "role": self.role.value,
            "argv": list(self.argv),
            "env_overrides": dict(self.env_overrides),
            "model": self.model,
            "token_cap": self.token_cap,
            "weights_bytes": self.weights_bytes,
            "slots": self.slots,
            "ctx": self.ctx,
            "replica": self.replica,
            "rerank_mode": self.rerank_mode.value if self.rerank_mode else None,
            "est_vram_bytes": self.est_vram_bytes,
        }

    @classmethod
    def from_state(cls, payload: dict) -> InstanceLaunch:
        """Rebuild a launch from :meth:`to_state` output; raises on a foreign shape."""
        raw_mode = payload.get("rerank_mode")
        return cls(
            role=WorkerRole(payload["role"]),
            argv=list(payload["argv"]),
            env_overrides=dict(payload.get("env_overrides") or {}),
            model=str(payload["model"]),
            token_cap=payload.get("token_cap"),
            weights_bytes=int(payload.get("weights_bytes") or 0),
            slots=int(payload.get("slots") or 1),
            est_vram_bytes=int(payload.get("est_vram_bytes") or 0),
            ctx=int(payload.get("ctx") or 0),
            replica=int(payload.get("replica") or 0),
            rerank_mode=RerankMode(raw_mode) if raw_mode else None,
        )

    @property
    def model_id(self) -> str:
        """The llama-swap model id for this instance: ``<role>-<replica>``."""
        return f"{role_model_prefix(self.role)}{self.replica}"
