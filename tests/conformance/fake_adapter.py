"""A second engine adapter that shares nothing with llama.cpp: argv, formats, readback, fields."""

from __future__ import annotations

import functools
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engines.base import LaunchRequest, RequestProfile
from lilbee.providers.fleet.engines.host_gate import HostGate, gate_on
from lilbee.providers.fleet.host_caps import HostCaps, nvidia_feature
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, WorkerRole

_FAKE_BINARY = Path(__file__).with_name("fake_binary.py")
_SHA = "b" * 64
_BINARY_CFG = (
    '{"flags": ["--port", "--weights"], "devices": "usage", "estimate": true,'
    ' "term": "exit", "help_rc": 0}'
)
_RESERVE_BYTES = 512 * 1024 * 1024


@functools.cache
def _binary_path() -> Path:
    """An executable stand-in for the engine, written once per process."""
    script = (
        f"#!{sys.executable}\n"
        "import os, runpy\n"
        f"os.environ['FAKE_ENGINE_CFG'] = {_BINARY_CFG!r}\n"
        f"runpy.run_path({str(_FAKE_BINARY)!r}, run_name='__main__')\n"
    )
    path = Path(tempfile.mkdtemp(prefix="fake-adapter-")) / "fake-engine"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


class FakeAdapter:
    """An engine unlike llama.cpp in every field the planner could assume."""

    name = "fake-engine"
    formats = frozenset({"safetensors"})
    architectures = frozenset({"fakearch"})
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED})
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures

    def host_gate(self, host: HostCaps) -> HostGate:
        return gate_on(host.avx2, nvidia_feature(host))

    def binary(self) -> Path:
        return _binary_path()

    def identity(self, binary: Path) -> str:
        return f"fake-engine@{binary.name}"

    def pin_record(self) -> Mapping[str, Any]:
        return {
            "engine": self.name,
            "version": "1.2.3",
            "backend": "cpu",
            "launch": ["{binary}", "--port", "{port}"],
            "platforms": {
                "linux-x86_64": {
                    "url": "https://example.org/fake-engine-linux.tar.gz",
                    "sha256": _SHA,
                    "min_os": None,
                },
                "macos-arm64": {
                    "url": "https://example.org/fake-engine-mac.tar.gz",
                    "sha256": _SHA,
                    "min_os": "14.0",
                },
            },
            "memory": {"sizing": ["{binary}", "--estimate", "{model}"], "rule": None},
            "licenses": ["MIT"],
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        return RoleServerSpec(
            role=role,
            endpoint_path="/api/generate",
            extra_args=("--chat-mode",),
            server_capable=True,
        )

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        return [
            str(request.binary),
            "--weights",
            str(request.model_path),
            "--window",
            str(request.ctx_per_slot * request.slots),
            *spec.extra_args,
        ]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=0,
            host_bytes=_RESERVE_BYTES,
            shared_bytes=_RESERVE_BYTES,
        )

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return None

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {
            "checkEndpoint": "/live",
            "useModelName": "default",
            "filters": {"stripParams": "embd_normalize, return_progress"},
        }

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return RequestProfile(
            embed_fields={},
            stream_fields={},
            tokenize_path="/v1/tokenize",
            detokenize_path="/v1/detokenize",
            apply_template_path="/v1/render",
            label="fake-engine",
            context_overflow=re.compile("FAKE_CTX_FULL"),
            batch_overflow=re.compile("FAKE_BATCH_FULL"),
            unreadable_embedding_hint="the fake engine returns vectors in its own format.",
            reasoning_field="thoughts",
        )


FAKE = FakeAdapter()
