"""The launch spec for one llama-server instance, shared by planning and llama-swap."""

from __future__ import annotations

from dataclasses import dataclass, field

from lilbee.providers.roles import RerankMode, WorkerRole

# Separator between a role and its replica index in a llama-swap model id.
_REPLICA_SEP = "-"
# Stand-in for the binary of a record whose argv arrived empty (a foreign state file).
_UNKNOWN_BINARY = "an unrecorded binary"


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
    # The chat ctx target the builder planned against (num_ctx pin else
    # chat_n_ctx_target); 0 for non-chat roles and pre-field records. A
    # co-tenant whose target this covers adopts the engine even when the
    # served window is smaller: the same planner aiming at least as high
    # already achieved this window, so a rebuild cannot beat it.
    built_ctx_target: int = 0
    replica: int = 0  # index within the role's data-parallel pool (0 = single server)
    rerank_mode: RerankMode | None = None  # set only for RERANK; picks the client scoring path
    # GPU bytes placement charged this instance. Carried so the engine's own
    # startup report can be checked against it once the server is up; 0 for a
    # model the estimator could not size, where there is nothing to compare.
    est_vram_bytes: int = 0
    # What placement charged each card this instance runs on, keyed by the name
    # the engine prints for it. The scalar above cannot distinguish a split that
    # landed 50/50 from one that landed 80/20, and the second is the one that
    # overruns a card.
    est_vram_by_device: dict[str, int] = field(default_factory=dict)
    est_unreported_bytes: int = 0
    """Estimated bytes the engine allocates but never reports in its buffer lines.

    A vision projector's weights are the case: llama.cpp allocates them without
    emitting a "buffer size" line, so the readback total is short by exactly this
    much and the self-check would warn on a load that was sized correctly.

    Log-mode readback only. An engine serving GET /memory reports the projector
    per device in its mmproj field, so that path compares the full estimate and
    ignores this.
    """

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
            "built_ctx_target": self.built_ctx_target,
            "replica": self.replica,
            "rerank_mode": self.rerank_mode.value if self.rerank_mode else None,
            "est_vram_bytes": self.est_vram_bytes,
            "est_vram_by_device": dict(self.est_vram_by_device),
            "est_unreported_bytes": self.est_unreported_bytes,
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
            est_unreported_bytes=int(payload.get("est_unreported_bytes") or 0),
            est_vram_by_device={
                str(k): int(v) for k, v in (payload.get("est_vram_by_device") or {}).items()
            },
            ctx=int(payload.get("ctx") or 0),
            built_ctx_target=int(payload.get("built_ctx_target") or 0),
            replica=int(payload.get("replica") or 0),
            rerank_mode=RerankMode(raw_mode) if raw_mode else None,
        )

    @property
    def binary(self) -> str:
        """The llama-server this instance runs: the argv's first word."""
        return next(iter(self.argv), _UNKNOWN_BINARY)

    @property
    def model_id(self) -> str:
        """The llama-swap model id for this instance: ``<role>-<replica>``."""
        return f"{role_model_prefix(self.role)}{self.replica}"
