"""The contract an inference engine implements for the fleet."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, assert_never, runtime_checkable

from lilbee.catalog.hf_client import HfToken
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

if TYPE_CHECKING:
    from lilbee.providers.fleet.adapters import RoleServerSpec
    from lilbee.providers.fleet.devices import DeviceProbe
    from lilbee.providers.fleet.engines.host_gate import HostGate
    from lilbee.providers.fleet.host_caps import HostCaps
    from lilbee.providers.fleet.launch import InstanceLaunch


# Stands for the port value in an argv whose port flag must sit before a subcommand.
PORT_SLOT = "{port}"


class EngineName(StrEnum):
    """The engines the fleet can launch."""

    LLAMA = "llama.cpp"
    IK_LLAMA = "ik_llama.cpp"
    MISTRAL = "mistral.rs"
    TEI = "tei"
    MLX = "mlx"
    OPENVINO = "openvino"
    LEMONADE = "lemonade"
    OVMS = "ovms"


class RerankShape(StrEnum):
    """The request and answer shape of an engine's rerank route."""

    TEXTS = "texts"
    """The request holds ``query`` and ``texts``; the answer lists ``index`` and ``score``."""
    DOCUMENTS = "documents"
    """The Cohere shape: ``query`` and ``documents``; the answer holds ``results`` of ``index``
    and ``relevance_score``."""


@dataclass(frozen=True)
class ServerProcess:
    """How a running server of an engine shows in the process table.

    ``name`` is the executable name, matched whole. ``folder`` returns a directory that the
    server's command line points into, or is ``None`` when the server is found by its
    ``--port`` only. ``interpreted`` is true for a script server: its process name is the
    interpreter's, and the first argument that is no flag names the script. That argument
    matches ``name``, or one of ``scripts`` when a launcher hands over to a script or module
    of another name.
    """

    name: str
    folder: Callable[[], Path] | None = None
    interpreted: bool = False
    scripts: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchRequest:
    """Placement and load choices for one instance; the adapter turns them into argv."""

    binary: Path
    model_path: Path
    devices: tuple[int, ...]
    n_gpu_layers: int
    slots: int
    ctx_per_slot: int
    device_names: tuple[str, ...] = ()
    tensor_split: tuple[int, ...] = ()
    mmproj: Path | None = None
    flash_attn: str | None = None
    cache_type: str | None = None
    batch_size: int | None = None
    no_mmap: bool = False
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    memory_endpoint: bool = False
    fit_target_mib: tuple[int, ...] = ()
    kv_unified_per_slot: int | None = None
    target: RoleTarget = RoleTarget.GPU
    replica: int = 0


@dataclass(frozen=True)
class UpstreamCheck:
    """A route of the engine's own server that the client reads, and how to read its reply."""

    path: str
    """The route on the engine's server; the client reaches it through llama-swap."""
    refusal: Callable[[Any], str | None]
    """The user-facing error that the decoded reply shows, or None when it shows none."""


@dataclass(frozen=True)
class RequestProfile:
    """The request fields and paths the client uses against an engine's server."""

    embed_fields: Mapping[str, Any]
    """Extra fields on every embeddings request body."""
    stream_fields: Mapping[str, Any]
    """Extra fields on a streamed chat request body."""
    tokenize_path: str | None
    """The tokenize route, or None when the engine has none (the client then estimates)."""
    detokenize_path: str | None
    """The detokenize route, or None when the engine has none."""
    apply_template_path: str
    label: str
    """The name of the engine's server in error messages."""
    context_overflow: re.Pattern[str]
    """Matches an error body that reports a prompt past the context window."""
    batch_overflow: re.Pattern[str] | None
    """Matches a 500 body that reports an input past the physical batch, or None."""
    unreadable_embedding_hint: str
    """What to tell the user when embedding vectors do not decode."""
    reasoning_field: str
    """The response field that carries reasoning text apart from the answer."""
    structured_output: bool = True
    """Whether the server enforces ``response_format``; if not, the client drops or refuses it."""
    works_on_after_disconnect: bool = False
    """Whether the server keeps working on a request after its client goes away.

    Such a server makes every later request wait, so a timed-out request restarts it.
    """
    rerank_path: str | None = None
    """The route that scores a query against texts, or None to rank through embeddings."""
    batch_limit: int | None = None
    """The most inputs one embed or rerank request may hold, or None for the client default.

    The launch argv of a managed server is built from this figure.
    """
    request_tokens: int | None = None
    """The most tokens one embed request may hold across its inputs. Rerank does not read it.

    ``None`` keeps a request within the per-input token cap, so each request holds one
    full input. An engine sets it when it queues the inputs of a request one by one and
    so takes a request larger than its per-input window.
    """
    batch_count_refusal: re.Pattern[str] | None = None
    """Matches an error body that refuses a request for holding too many inputs, or None."""
    takes_cache_prompt: bool = False
    """True when the engine reads the ``cache_prompt`` field of a chat body; others never get it."""
    takes_seed: bool = False
    """True when the engine reads ``seed``; others never get a seed that lilbee chose."""
    rerank_fields: Mapping[str, Any] = field(default_factory=dict)
    """Extra fields on a rerank route request, such as one that stops silent truncation."""
    context_overflow_status: int = 400
    """The status the server answers an input past its limit with."""
    input_count_path: str | None = None
    """The route that counts the tokens of a list of inputs, or None for an engine that has none."""
    input_limit_path: str | None = None
    """The route that reports ``max_input_length``, or None. An engine with both routes cuts an
    over-limit embed input by its own tokens without a sign, so the client sends such an input
    whole, then counts the cuts and warns."""
    rerank_shape: RerankShape = RerankShape.TEXTS
    """The shape of the rerank route; it matters only when ``rerank_path`` is set."""
    rerank_model: str | None = None
    """The model name the rerank request body carries, or None for a route that takes none."""
    device_check: UpstreamCheck | None = None
    """Read once after a warm loads the model; an error fails the warm, because the engine
    does not run the model on the device that the role was placed on. None skips the read."""
    load_failure_check: UpstreamCheck | None = None
    """Read once after a warm fails; an error replaces the engine's own failure with a reason
    the user can act on. None keeps the engine's failure."""

    def rerank_body(
        self, query: str, texts: list[str], extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The rerank request body for *texts*; *extra* holds fields such as ``truncate``."""
        extra = self.rerank_fields if extra is None else extra
        match self.rerank_shape:
            case RerankShape.TEXTS:
                return {"query": query, "texts": texts, **extra}
            case RerankShape.DOCUMENTS:
                body: dict[str, Any] = {"query": query, "documents": texts, "top_n": len(texts)}
                if self.rerank_model is not None:
                    body["model"] = self.rerank_model
                return {**body, **extra}
        assert_never(self.rerank_shape)  # pragma: no cover

    def rerank_scores(self, payload: Any) -> dict[int, float]:
        """The score of each candidate by index, from the answer of the rerank route."""
        match self.rerank_shape:
            case RerankShape.TEXTS:
                return {int(item["index"]): float(item["score"]) for item in payload}
            case RerankShape.DOCUMENTS:
                return {
                    int(item["index"]): float(item["relevance_score"])
                    for item in payload["results"]
                }
        assert_never(self.rerank_shape)  # pragma: no cover


class EngineAdapter(Protocol, Hashable):
    """One inference engine behind the fleet; hashable, so per-engine caches can key on it.

    The provider holds one adapter. Call order: ``binary()`` resolves the
    executable, ``discover_devices`` probes it, ``estimate_memory`` sizes a
    plan, ``role_spec`` and ``launch_argv`` build each command line, and
    ``check_readback`` says whether the running instance reports its memory.
    ``device_targets`` runs after ``binary()``, before ``launch_argv``, and only
    for a role that names a target other than gpu.
    """

    @property
    def name(self) -> str: ...

    @property
    def formats(self) -> frozenset[str]: ...

    architectures: frozenset[str]
    served_roles: frozenset[WorkerRole]
    memory_kind: MemoryKind
    """A fixed reserve's figure does not follow the window and has no share per card."""
    bundled: bool
    """Whether the engine ships in the package, so it needs no build in the engine store."""
    takes_device_names: bool
    """Whether ``launch_argv`` puts ``LaunchRequest.device_names`` on the command line."""
    server_process: ServerProcess
    """How the orphan sweep recognises this engine's server after its llama-swap is gone."""
    writes_own_log: bool
    """Whether the engine writes its output to the file the ``LLAMA_LOG_FILE`` variables name.

    An engine that is False gets its output copied into the llama-swap log of its group.
    """

    concatenates_pooling: bool
    """Whether a model that enables several pooling modes gets one vector block per mode.

    An engine that is False serves the first enabled mode only, so the vector
    keeps the width of one block.
    """

    def supports_architecture(self, architecture: str) -> bool: ...

    def host_gate(self, host: HostCaps) -> HostGate:
        """Whether *host* can run this engine; callable at any time, before ``binary()``."""
        ...

    def binary(self) -> Path:
        """The server binary; raises ``ProviderError`` when this host has no install of it."""
        ...

    def identity(self, binary: Path) -> str: ...

    def pin_record(self) -> Mapping[str, Any]: ...

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec: ...

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]: ...

    def discover_devices(self, binary: Path) -> DeviceProbe: ...

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        """The targets *binary* can run a role on; ``launch_argv`` writes their flags."""
        ...

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        """Size one instance; ``per_device`` follows a ratio only if ``launch_argv`` honours it."""
        ...

    def check_readback(self, binary: Path) -> bool:
        """Whether *binary* can report its memory use by endpoint."""
        ...

    def supports_fit_target(self, binary: Path) -> bool:
        """Whether *binary* places layers itself, leaving a stated margin free."""
        ...

    def supports_shared_kv(self, binary: Path) -> bool:
        """Whether *binary* can give its slots one KV pool with a per-slot cap."""
        ...

    def model_meta(self, path: Path) -> dict[str, str] | None:
        """Header metadata of the model file at *path*, or ``None`` when it has none."""
        ...

    def assert_loadable(self, repo: str, filename: str, token: HfToken) -> None:
        """Raise ``UnsupportedQuantError`` when the engine cannot load the remote file.

        *token* is a Hugging Face token. A string is the token to send. ``False`` means
        a token was refused, so the request goes anonymous. ``None`` means none was found.
        The adapter sends no token for ``False`` or ``None``.
        """
        ...

    def runtime_env(self) -> Mapping[str, str]:
        """Extra environment variables for a spawned engine process."""
        ...

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        """Environment variables for a process on *target* with *cards* pinned cards."""
        ...

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]: ...

    def request_profile(self, role: WorkerRole) -> RequestProfile: ...


def engine_label(engine: EngineAdapter) -> str:
    """The name of *engine*'s server in messages, from the profile of its first role."""
    return engine.request_profile(min(engine.served_roles)).label


class SplitRefusedError(ProviderError):
    """An engine's launch refuses a tensor split; the message names the rule."""


@runtime_checkable
class SplitChecked(Protocol):
    """An engine whose launch refuses some tensor splits; placement asks the same rule."""

    def check_split(self, model_path: Path, ratio: tuple[int, ...]) -> None:
        """Raise ``SplitRefusedError`` naming why *ratio* cannot launch over its cards."""
        ...


@runtime_checkable
class RoleFormatChecked(Protocol):
    """An engine that loads a format for some roles and not for others."""

    refused_role_formats: Mapping[WorkerRole, frozenset[str]]
    """The formats the engine cannot load for each role, though it loads them for another."""


@runtime_checkable
class UserInstalled(Protocol):
    """An engine that lilbee has no pack for: the user installs it and lilbee finds it."""

    install_hint: str
    """How to get the engine, for a message that says it is missing."""

    install_docs_url: str
    """The page that tells how to install the engine; empty when there is none."""

    def user_binary(self) -> Path | None:
        """The executable the user installed, or ``None`` when there is none."""
        ...
