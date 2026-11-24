"""The OpenVINO Model Server adapter: chat, embed and rerank of OpenVINO IR models."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lilbee.catalog.hf_client import HfToken
from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
    RerankShape,
    ServerProcess,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate, npu_gate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, NpuVendor
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_VERSION = "2026.4.1"
_BACKEND = "openvino"
_TOOL = "ovms"
_INSTALL_DOCS_URL = (
    "https://docs.openvino.ai/2026/model-server/ovms_docs_deploying_server_baremetal.html"
)
_LIBRARY_PATH_VAR = "LD_LIBRARY_PATH"
_INSTALL_HINT = (
    "lilbee has no OpenVINO Model Server pack. "
    f"Install OpenVINO Model Server {_VERSION} as {_INSTALL_DOCS_URL} tells, "
    f"then put the folder with {_TOOL} on PATH and set {_LIBRARY_PATH_VAR} to its lib folder."
)
_HOST = "127.0.0.1"
_INTEL_PCI_VENDOR = 0x8086
_SERVED_NAME = "default_model"
"""The one name OVMS serves the model under; llama-swap rewrites the model of a request to it."""
_BATCH_LIMIT = 100
"""The most documents lilbee sends in one rerank request. The server allows 10000 by default."""
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
_POSITION_KEY = "max_position_embeddings"
_DEVICES = {RoleTarget.CPU: "CPU", RoleTarget.GPU: "GPU", RoleTarget.NPU: "NPU"}
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


def _position_limit(model_dir: Path) -> int | None:
    """The input length that the model's ``config.json`` names, or ``None``."""
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    limit = config.get(_POSITION_KEY) if isinstance(config, dict) else None
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return limit
    return None


class OvmsAdapter:
    """OpenVINO Model Server behind the engine contract.

    It loads OpenVINO IR folders only, one model for each process. Its HTTP port is
    ``--rest_port``; ``--port`` is its gRPC port, which lilbee leaves off. Its routes sit
    under ``/v3``, and its rerank route takes the Cohere request shape.

    lilbee has no pack of it: the user installs the server, and lilbee runs the
    ``ovms`` on PATH.
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
    install_hint = _INSTALL_HINT
    install_docs_url = _INSTALL_DOCS_URL

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
        tool = self.user_binary()
        if tool is None:
            raise ProviderError(
                f"The OpenVINO Model Server engine is not installed. {_INSTALL_HINT}",
                kind=ProviderErrorKind.NOT_FOUND,
            )
        return tool

    def user_binary(self) -> Path | None:
        """The ``ovms`` that the user put on PATH, or ``None``."""
        found = shutil.which(_TOOL)
        return None if found is None else Path(found)

    def identity(self, binary: Path) -> str:
        return f"{self.name.value} {_VERSION} {binary}"

    def pin_record(self) -> Mapping[str, Any]:
        """The record of an engine with no pack: no digest, no platforms and no licences."""
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": _BACKEND,
            "sha256": None,
            "tool": _TOOL,
            "launch": _PIN_LAUNCH,
            "platforms": {},
            "memory": {"sizing": None, "rule": None},
            "licenses": [],
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, _ENDPOINTS[role], ("--task", _TASKS[role]), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The server command; the HTTP port is ``--rest_port`` and the gRPC port stays off.

        An embed launch on the NPU bounds the input and cuts a longer one: the NPU compiles for
        one fixed length. The bound is the smaller of the window and the model's position limit.
        """
        device = _DEVICES.get(request.target, _DEVICES[RoleTarget.CPU])
        bound: list[str] = []
        if spec.role is WorkerRole.EMBED and request.target is RoleTarget.NPU:
            limit = _position_limit(request.model_path) or request.ctx_per_slot
            limit = min(limit, request.ctx_per_slot)
            bound = ["--max_length", str(limit), "--truncate", "true"]
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
            *bound,
        ]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False, has_listing=False)

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        targets = {RoleTarget.CPU}
        if _INTEL_PCI_VENDOR in host.display_vendors:
            targets.add(RoleTarget.GPU)
        npu = npu_gate(host, NpuVendor.INTEL)
        if npu.ok and npu.known:
            targets.add(RoleTarget.NPU)
        return frozenset(targets)

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
        """``ovms/lib`` on ``LD_LIBRARY_PATH``: the binary finds ``libtbb`` only through it.

        The release archive holds ``ovms/bin/ovms`` beside ``ovms/lib``. A value already set
        stays after the new folder. Nothing is added when there is no ``lib`` folder.
        """
        tool = self.user_binary()
        if tool is None:
            return {}
        lib = tool.resolve().parent.parent / "lib"
        if not lib.is_dir():
            return {}
        existing = os.environ.get(_LIBRARY_PATH_VAR, "")
        return {_LIBRARY_PATH_VAR: os.pathsep.join(p for p in (str(lib), existing) if p)}

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"checkEndpoint": "/v2/health/ready", "useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _RERANK_PROFILE if role is WorkerRole.RERANK else _PROFILE


OVMS = OvmsAdapter()
ADAPTER = OVMS
