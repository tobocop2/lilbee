"""The Lemonade adapter: the ``lemond`` server, which runs NPU models through its flm recipe."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.catalog.xdna import read_checkpoint
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

_VERSION = "2026.44.1"
_TOOL = "lemond"
_BACKEND = "npu"
_HOST = "127.0.0.1"
_LICENCES = ["Apache-2.0"]
_XDNA2 = "XDNA2"
_NPU_SYSTEMS = frozenset({HostOS.LINUX, HostOS.WINDOWS})
_RESERVE_BYTES = 2 * 1024**3
"""The figure charged for one role; not measured on an NPU."""
_ENDPOINTS = {
    WorkerRole.CHAT: "/v1/chat/completions",
    WorkerRole.EMBED: "/v1/embeddings",
}
_DATA_DIRNAME = "lemonade-data"
_RECIPE = "flm"
_USER_PREFIX = "user."
_KEY_CHECKPOINT = "checkpoint"
_KEY_RECIPE = "recipe"
_KEY_LABELS = "labels"
_KEY_CTX = "ctx_size"
_LABEL_EMBEDDINGS = "embeddings"
_TMP_SUFFIX = ".tmp"
_INSTALL_HINT = (
    "lilbee has no Lemonade pack. Install Lemonade from "
    "https://lemonade-server.ai/docs/guide/install/ or unpack the lemonade-embeddable "
    f"archive from https://github.com/lemonade-sdk/lemonade/releases/tag/v{_VERSION}, "
    f"then put the folder with {_TOOL} on PATH."
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
)


def data_root() -> Path:
    """The folder that holds every Lemonade role folder, apart from any Lemonade of the user."""
    return engines_root(cfg.models_dir) / _DATA_DIRNAME


def role_dir(role: WorkerRole, replica: int) -> Path:
    """The private cache, config and model view of one role instance."""
    return data_root() / f"{role.value}-{replica}"


def _entry_name(role: WorkerRole) -> str:
    """The key of the model of *role* in ``user_models.json``."""
    return f"lilbee-{role.value}"


def model_name(role: WorkerRole) -> str:
    """The name Lemonade gives the model of *role* in requests and in ``recipe_options.json``."""
    return f"{_USER_PREFIX}{_entry_name(role)}"


def _write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* through a temporary file, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _registration(role: WorkerRole, tag: str) -> dict[str, dict[str, Any]]:
    """The ``user_models.json`` content: one NPU-recipe model named by its tag."""
    entry: dict[str, Any] = {_KEY_CHECKPOINT: tag, _KEY_RECIPE: _RECIPE}
    if role is WorkerRole.EMBED:
        entry[_KEY_LABELS] = [_LABEL_EMBEDDINGS]
    return {_entry_name(role): entry}


def _prepare(spec: RoleServerSpec, request: LaunchRequest) -> Path:
    """Write the folders, the config and the model registration of one role; safe to repeat."""
    root = role_dir(spec.role, request.replica)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    registry = _registration(spec.role, read_checkpoint(request.model_path))
    options = {model_name(spec.role): {_KEY_CTX: request.ctx_per_slot * request.slots}}
    config = root / "config"
    _write_atomic(config / "config.json", json.dumps({"websocket_port": "auto"}, indent=2))
    _write_atomic(config / "user_models.json", json.dumps(registry, indent=2))
    _write_atomic(config / "recipe_options.json", json.dumps(options, indent=2))
    return root


class LemonadeAdapter:
    """Lemonade behind the engine contract; chat and embed on the AMD XDNA2 NPU."""

    name = EngineName.LEMONADE
    formats = frozenset({ModelFormat.XDNA})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset(_ENDPOINTS)
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False
    bundled = False
    install_hint = _INSTALL_HINT

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        """Linux or Windows on x86_64 with an AMD XDNA2 NPU; an unread NPU list is unknown."""
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if host.os not in _NPU_SYSTEMS or host.arch is not HostArch.X86_64:
            return HostGate(False, f"Lemonade is not offered on {host.os.value} {host.arch.value}.")
        gate = npu_gate(host, NpuVendor.AMD, _XDNA2)
        return (
            gate if gate.ok else HostGate(False, f"{gate.reason} Lemonade needs an AMD XDNA2 NPU.")
        )

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        tool = tool or self.user_binary()
        if tool is None:
            raise ProviderError(f"The Lemonade engine is not installed. {_INSTALL_HINT}")
        return tool

    def user_binary(self) -> Path | None:
        """The ``lemond`` that the user put on PATH, or ``None``."""
        found = shutil.which(_TOOL)
        return None if found is None else Path(found)

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
        """The ``lemond`` command, after the role folder holds its config and model entry.

        ``lemond`` downloads the model on its first load and picks the ports of
        its NPU server and WebSocket itself.
        """
        root = _prepare(spec, request)
        return [
            str(request.binary),
            "--port",
            PORT_SLOT,
            "--host",
            _HOST,
            "--no-broadcast",
            str(root / "cache"),
            str(root / "config"),
        ]

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        return frozenset({RoleTarget.NPU})

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

    def target_env(self, target: RoleTarget) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"checkEndpoint": "/live", "useModelName": model_name(launch.role)}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LEMONADE = LemonadeAdapter()
ADAPTER = LEMONADE
