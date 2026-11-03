"""The Lemonade adapter: the ``lemond`` server, which starts its own llama.cpp children."""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engine_store import engines_root
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
)
from lilbee.providers.fleet.engines.host_gate import PLATFORM_UNREAD, HostGate, npu_gate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, NpuVendor
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

log = logging.getLogger(__name__)

_VERSION = "2026.44.1"
_TOOL = "lemond"
_BACKEND = "cpu"
_HOST = "127.0.0.1"
_LICENCES = ["Apache-2.0"]
_XDNA2 = "XDNA2"
# Lemonade starts one llama-server per role on these ports and opens a WebSocket on
# the last one. The recorded run did not set the first three, so lilbee cannot move them.
CHILD_PORTS = (8001, 8002, 8003)
WEBSOCKET_PORT = 9000
_RESERVE_BYTES = 2 * 1024**3
"""The figure charged for one role; the recorded chat, embed and rerank models fit under it."""
_ENDPOINTS = {
    WorkerRole.CHAT: "/v1/chat/completions",
    WorkerRole.EMBED: "/v1/embeddings",
    WorkerRole.RERANK: "/v1/reranking",
}
_DATA_DIRNAME = "lemonade-data"
_BACKEND_DOWNLOAD = (
    "Lemonade downloads its llama.cpp backend (about 17 MB) the first time a model loads. "
    "The first request waits for that download."
)

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="Lemonade",
    context_overflow=re.compile("exceed_context_size|context size", re.IGNORECASE),
    batch_overflow=None,
    unreadable_embedding_hint="Lemonade answers embeddings as lists of floats.",
    reasoning_field="reasoning_content",
    rerank_path="/v1/reranking",
)


def port_is_busy(port: int) -> bool:
    """True when something on this computer already listens on *port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((_HOST, port)) == 0


def busy_ports() -> tuple[int, ...]:
    """The ports Lemonade needs that something already holds."""
    return tuple(p for p in (*CHILD_PORTS, WEBSOCKET_PORT) if port_is_busy(p))


def data_dirs() -> tuple[Path, Path]:
    """The cache and config folders of Lemonade, apart from any Lemonade of the user."""
    root = engines_root(cfg.models_dir) / _DATA_DIRNAME
    return root / "cache", root / "config"


class LemonadeAdapter:
    """Lemonade behind the engine contract; chat, embed and rerank, on an AMD XDNA2 host."""

    name = EngineName.LEMONADE
    formats = frozenset({ModelFormat.GGUF})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset(_ENDPOINTS)
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        """Linux on x86_64 with an AMD XDNA2 NPU; an unread NPU list is unknown, not absent."""
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64):
            return HostGate(False, f"Lemonade is not offered on {host.os.value} {host.arch.value}.")
        gate = npu_gate(host, NpuVendor.AMD, _XDNA2)
        return (
            gate if gate.ok else HostGate(False, f"{gate.reason} Lemonade needs an AMD XDNA2 NPU.")
        )

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(
                "The Lemonade engine is not installed. Run lilbee engine install lemonade."
            )
        return tool

    def identity(self, binary: Path) -> str:
        return f"{self.name.value} {_VERSION} {binary}"

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": _BACKEND,
            "sha256": pack_digest(self.name.value, _VERSION, host, _BACKEND),
            "tool": _TOOL,
            "launch": [_TOOL, "--port", PORT_SLOT],
            "platforms": pack_platforms(self.name.value, _VERSION, host, _BACKEND),
            "memory": {"sizing": None, "rule": MemoryKind.FIXED_RESERVE.value},
            "licenses": _LICENCES,
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, _ENDPOINTS[role], (), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The ``lemond`` command, after a check that its fixed ports are free.

        Lemonade picks the ports of its llama.cpp children itself, so lilbee refuses
        to start it while another program holds one of them.
        """
        held = busy_ports()
        if held:
            ports = ", ".join(str(p) for p in held)
            raise ProviderError(
                f"Lemonade needs the ports {', '.join(map(str, CHILD_PORTS))} and "
                f"{WEBSOCKET_PORT}, and port {ports} is in use. Stop the program that uses it."
            )
        cache, config = data_dirs()
        if not (cache / "bin" / "llamacpp" / _BACKEND).is_dir():
            log.warning(_BACKEND_DOWNLOAD)
        return [
            str(request.binary),
            "--port",
            PORT_SLOT,
            "--host",
            _HOST,
            "--no-broadcast",
            str(cache),
            str(config),
        ]

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        return frozenset({RoleTarget.CPU})

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=_RESERVE_BYTES,
            host_bytes=0,
            shared_bytes=_RESERVE_BYTES,
        )

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return None

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"checkEndpoint": "/live"}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LEMONADE = LemonadeAdapter()
ADAPTER = LEMONADE
