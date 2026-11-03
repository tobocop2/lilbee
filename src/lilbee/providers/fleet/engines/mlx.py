"""The mlx-lm adapter; the only module that names mlx-lm's server and its flags."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_VERSION = "0.32.2"
_TOOL = "mlx-lm-server"
_HOST = "127.0.0.1"
_METAL = "metal"
_LICENCES = ["MIT", "Apache-2.0", "BSD-3-Clause"]
MACOS_FLOOR = (14, 0)
"""The oldest macOS the pack runs on; the ``min_os`` of the published row states the same."""
_SERVED_NAME = "default_model"
"""The model name that selects the model the server loaded at start.

Any other name makes mlx-lm load that name as a Hugging Face repo.
"""
_PROMPT_CACHE_BYTES = 1024**3
"""The byte bound on the prompt cache; the server default holds ten caches with no bound.

Not measured on a Mac yet.
"""
_CONFIG = "config.json"
_WINDOW_KEY = "max_position_embeddings"

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="mlx-lm",
    # The server has no prompt-length check and no overflow body; lilbee bounds the prompt itself.
    context_overflow=re.compile(r"(?!)"),
    batch_overflow=None,
    unreadable_embedding_hint="mlx-lm serves no embeddings.",
    reasoning_field="reasoning",
)


def _floor_text() -> str:
    return ".".join(map(str, MACOS_FLOOR))


class MlxAdapter:
    """mlx-lm behind the engine contract; chat only, on Apple Silicon with macOS 14 or newer."""

    name = EngineName.MLX
    formats = frozenset({ModelFormat.MLX})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset({WorkerRole.CHAT})
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if host.os is not HostOS.MACOS:
            return HostGate(False, f"mlx is not offered on {host.os.value} {host.arch.value}.")
        if host.arch is not HostArch.ARM64:
            return HostGate(
                False,
                f"mlx needs Apple Silicon. This process reports {host.arch.value}. "
                "If this Mac has Apple Silicon, run lilbee natively, not under Rosetta.",
            )
        if not host.macos_known or host.macos_version is None:
            return HostGate(True, host.macos_reason, known=False)
        if (*host.macos_version, 0, 0)[:2] < MACOS_FLOOR:
            have = ".".join(map(str, host.macos_version))
            return HostGate(
                False,
                f"mlx needs macOS {_floor_text()} or later; this computer has {have}.",
            )
        return PASS

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(
                f"The mlx engine is not installed. Run lilbee engine install {self.name}."
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
            "backend": _METAL,
            "sha256": pack_digest(self.name.value, _VERSION, host, _METAL),
            "tool": _TOOL,
            "launch": ["{binary}", "--model", "{model}", "--host", _HOST, "--port", PORT_SLOT],
            "platforms": pack_platforms(self.name.value, _VERSION, host, _METAL),
            "memory": {"sizing": None, "rule": MemoryKind.FIXED_RESERVE.value},
            "licenses": _LICENCES,
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, ROLE_SPECS[role].endpoint_path, (), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The server command; mlx-lm places the model itself, so layers and context stay unset."""
        return [
            str(request.binary),
            "--model",
            str(request.model_path),
            "--host",
            _HOST,
            "--port",
            PORT_SLOT,
            "--prompt-cache-bytes",
            str(_PROMPT_CACHE_BYTES),
            *spec.extra_args,
        ]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="")

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        return frozenset({RoleTarget.GPU})

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        """The weights of the model folder, charged as a fixed reserve; the cache is not priced."""
        return estimate_memory(ModelFormat.MLX, request)

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        """The window of the model, from the ``config.json`` in its folder, or ``None``.

        A vision-language model keeps the value under ``text_config``.
        """
        try:
            config = json.loads((path / _CONFIG).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(config, dict):
            return None
        for section in (config, config.get("text_config")):
            window = section.get(_WINDOW_KEY) if isinstance(section, dict) else None
            if isinstance(window, int) and not isinstance(window, bool) and window > 0:
                return {"context_length": str(window)}
        return None

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


MLX = MlxAdapter()
ADAPTER = MLX
