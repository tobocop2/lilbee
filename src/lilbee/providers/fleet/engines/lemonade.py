"""The Lemonade adapter: the ``lemond`` server, which starts its own llama.cpp children."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.binary import resolve_llama_server
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
_BACKEND = "cpu"
_HOST = "127.0.0.1"
_LICENCES = ["Apache-2.0"]
_XDNA2 = "XDNA2"
_RESERVE_BYTES = 2 * 1024**3
"""The figure charged for one role; the recorded chat, embed and rerank models fit under it."""
_ENDPOINTS = {
    WorkerRole.CHAT: "/v1/chat/completions",
    WorkerRole.EMBED: "/v1/embeddings",
    WorkerRole.RERANK: "/v1/reranking",
}
_DATA_DIRNAME = "lemonade-data"
_MODES = {
    WorkerRole.CHAT: "chat",
    WorkerRole.EMBED: "embeddings",
    WorkerRole.RERANK: "reranking",
}
_TMP_SUFFIX = ".tmp"

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


def data_root() -> Path:
    """The folder that holds every Lemonade role folder, apart from any Lemonade of the user."""
    return engines_root(cfg.models_dir) / _DATA_DIRNAME


def role_dir(role: WorkerRole, replica: int) -> Path:
    """The private cache, config and model view of one role instance."""
    return data_root() / f"{role.value}-{replica}"


def model_name(role: WorkerRole) -> str:
    """The name Lemonade gives the model of *role*, taken from the link file name."""
    return f"lilbee-{role.value}"


def _write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* through a temporary file, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _links_to(link: Path, target: Path) -> bool:
    """True when *link* already serves *target*, as a link or as a hard link or copy."""
    if link.is_symlink():
        return link.readlink() == target
    try:
        return link.is_file() and os.path.samefile(link, target)
    except OSError:
        return False


def _place_model(link: Path, target: Path) -> None:
    """Make *link* serve *target*: a symlink, else a hard link, else a copy."""
    if _links_to(link, target):
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    tmp = link.with_name(link.name + _TMP_SUFFIX)
    tmp.unlink(missing_ok=True)
    for make in (os.symlink, os.link, shutil.copyfile):
        try:
            make(target, tmp)
        except OSError:
            continue
        os.replace(tmp, link)
        return
    raise ProviderError(f"Lemonade cannot link the model {target} into {link.parent}.")


def _prepare(spec: RoleServerSpec, request: LaunchRequest) -> Path:
    """Write the folders, the config and the model link of one role; safe to repeat."""
    root = role_dir(spec.role, request.replica)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    view = root / "view"
    _place_model(view / _MODES[spec.role] / f"{model_name(spec.role)}.gguf", request.model_path)
    config = {
        "extra_models_dir": str(view),
        "websocket_port": "auto",
        "llamacpp": {"backend": _BACKEND, "cpu_bin": str(resolve_llama_server().parent)},
    }
    recipe = {
        f"extra.{model_name(spec.role)}": {
            "ctx_size": request.ctx_per_slot * request.slots,
            "llamacpp_args": f"--parallel {request.slots}",
        }
    }
    _write_atomic(root / "config" / "config.json", json.dumps(config, indent=2))
    _write_atomic(root / "config" / "recipe_options.json", json.dumps(recipe, indent=2))
    return root


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
        """The ``lemond`` command, after the role folder holds its config and model link.

        Lemonade reads the model from the view folder in place, runs lilbee's own
        llama-server, and picks the ports of its children and WebSocket itself.
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
        return {"checkEndpoint": "/live", "useModelName": model_name(launch.role)}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LEMONADE = LemonadeAdapter()
ADAPTER = LEMONADE
