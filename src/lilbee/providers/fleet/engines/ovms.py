"""The OpenVINO Model Server adapter: chat, embed and rerank of OpenVINO IR models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lilbee.catalog.hf_client import HfToken
from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
    RerankShape,
    ServerProcess,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_VERSION = "2026.4.1"
_BACKEND = "openvino"
_TOOL = "ovms"
_LICENCES = ["Apache-2.0", "BSD-3-Clause", "GPL-2.0-only WITH GCC-exception-2.0"]
"""The server is Apache-2.0. It links libgit2, which is GPL-2.0 with a linking exception.

The wheel takes ittnotify under its BSD-3-Clause option."""
_HOST = "127.0.0.1"
_INTEL_PCI_VENDOR = 0x8086
_SERVED_NAME = "default_model"
"""The one name OVMS serves the model under; llama-swap rewrites the model of a request to it."""
_BATCH_LIMIT = 100
"""The most documents one rerank request holds; the server default refuses more."""
_TASKS = {
    WorkerRole.CHAT: "text_generation",
    WorkerRole.EMBED: "embeddings",
    WorkerRole.RERANK: "rerank",
}
_ENDPOINTS = {
    WorkerRole.CHAT: "/v3/chat/completions",
    WorkerRole.EMBED: "/v3/embeddings",
    WorkerRole.RERANK: "/v3/rerank",
}
_DEVICES = {RoleTarget.CPU: "CPU", RoleTarget.GPU: "GPU"}
"""The ``--target_device`` value of each target a launch accepts."""
_PIN_LAUNCH = [
    "{binary}",
    "--model_path",
    "{model}",
    "--model_name",
    _SERVED_NAME,
    "--rest_port",
    PORT_SLOT,
]

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="OpenVINO Model Server",
    context_overflow=re.compile(r"(?!)"),
    batch_overflow=None,
    unreadable_embedding_hint="OpenVINO Model Server returned vectors that could not be decoded.",
    reasoning_field="reasoning_content",
    structured_output=False,
)
_RERANK_PROFILE = replace(
    _PROFILE,
    rerank_path="/v3/rerank",
    rerank_shape=RerankShape.DOCUMENTS,
    rerank_model=_SERVED_NAME,
    batch_limit=_BATCH_LIMIT,
)


class OvmsAdapter:
    """OpenVINO Model Server behind the engine contract.

    It loads OpenVINO IR folders only, one model for each process. Its HTTP port is
    ``--rest_port``; ``--port`` is its gRPC port, which lilbee leaves off. Its routes sit
    under ``/v3``, and its rerank route takes the Cohere request shape.
    """

    name = EngineName.OVMS
    formats = frozenset({ModelFormat.OPENVINO})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset(_TASKS)
    memory_kind = MemoryKind.SIZED
    takes_device_names = False
    concatenates_pooling = False
    bundled = False
    server_process = ServerProcess(name=_TOOL)

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        """Linux on x86_64 only. The CPU device runs on any x86 CPU."""
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64):
            return HostGate(
                False, f"OpenVINO Model Server is not offered on {host.os.value} {host.arch.value}."
            )
        return PASS

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import install_advice, store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(
                f"The OpenVINO Model Server engine is not installed. {install_advice(self)}"
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
            "launch": _PIN_LAUNCH,
            "platforms": pack_platforms(self.name.value, _VERSION, host, _BACKEND),
            "memory": {"sizing": None, "rule": None},
            "licenses": _LICENCES,
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, _ENDPOINTS[role], ("--task", _TASKS[role]), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The server command; the HTTP port is ``--rest_port`` and the gRPC port stays off."""
        device = _DEVICES.get(request.target, _DEVICES[RoleTarget.CPU])
        return [
            str(request.binary),
            "--model_path",
            str(request.model_path),
            "--model_name",
            _SERVED_NAME,
            "--rest_bind_address",
            _HOST,
            "--rest_port",
            PORT_SLOT,
            "--target_device",
            device,
            *spec.extra_args,
        ]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False)

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        from lilbee.catalog.runnable import session_host

        if _INTEL_PCI_VENDOR in session_host().display_vendors:
            return frozenset({RoleTarget.CPU, RoleTarget.GPU})
        return frozenset({RoleTarget.CPU})

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return estimate_memory(ModelFormat.OPENVINO, request)

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return None

    def assert_loadable(self, repo: str, filename: str, token: HfToken) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"checkEndpoint": "/v2/health/ready", "useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _RERANK_PROFILE if role is WorkerRole.RERANK else _PROFILE


OVMS = OvmsAdapter()
ADAPTER = OVMS
