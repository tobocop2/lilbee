"""Text a user reads names the engine that answered, never a fixed llama-server or llama.cpp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import httpx
import pytest

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engines.base import EngineAdapter, RequestProfile, engine_label
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.openvino import OPENVINO
from lilbee.providers.roles import WorkerRole

_STALE_NAME = "llama-server"


def _mistral_client(status: int = 500) -> LlamaServerClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="boom")

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient(
        "http://gpu0",
        "m",
        http=http,
        profile=MISTRAL.request_profile(WorkerRole.CHAT),
    )


def test_a_mistral_http_error_names_mistral() -> None:
    with pytest.raises(ProviderError, match="HTTP 500") as caught:
        _mistral_client().chat([{"role": "user", "content": "hi"}])
    assert str(caught.value).startswith("mistral.rs returned HTTP 500")
    assert _STALE_NAME not in str(caught.value)


class _SilentEngine:
    """An adapter whose device probe lists nothing, over a real profile."""

    def __init__(self, real: EngineAdapter) -> None:
        self._real = real
        self.served_roles = real.served_roles

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return self._real.request_profile(role)

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe([], "", spoke_protocol=True)


def _warning_for(
    engine: EngineAdapter,
    binary: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> list[str]:
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
    )
    monkeypatch.setattr(
        "lilbee.providers.fleet.rocm_runtime.assert_rocm_devices_usable", lambda *_a: None
    )
    monkeypatch.setattr(
        "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
        lambda: frozenset({0x10DE}),
    )
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
        planning_mod.resolve_devices(binary, engine=cast(EngineAdapter, _SilentEngine(engine)))
    return [m for m in caplog.messages if "reported none" in m]


@pytest.mark.parametrize(
    ("engine", "name"),
    [(MISTRAL, "mistral.rs"), (OPENVINO, "llama-server (OpenVINO)")],
    ids=["mistral", "openvino"],
)
def test_the_unlisted_gpu_warning_names_the_engine_that_probed(
    engine: EngineAdapter,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    binary = Path("/bin/engine-bin")
    assert _warning_for(engine, binary, monkeypatch, caplog) == [
        f"This host has a NVIDIA GPU but the engine's device probe ({binary} --list-devices) "
        "reported none; placement is falling back to shared-memory mode with unpinned "
        f"GPUs. Check the GPU driver, CUDA_VISIBLE_DEVICES, and that this {name} build "
        "supports that GPU."
    ]


class _AskedEngine:
    """An engine that records the role its profile was asked for."""

    served_roles = frozenset({WorkerRole.EMBED, WorkerRole.CHAT})

    def __init__(self) -> None:
        self.asked: list[WorkerRole] = []

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        self.asked.append(role)
        return MISTRAL.request_profile(role)


def test_the_engine_label_reads_the_profile_of_its_first_role() -> None:
    engine = _AskedEngine()
    assert engine_label(cast(EngineAdapter, engine)) == "mistral.rs"
    assert engine.asked == [min(engine.served_roles)]


def test_the_provider_setting_help_does_not_name_one_engine() -> None:
    help_text = SETTINGS_MAP["llm_provider"].help_text
    assert _STALE_NAME not in help_text
    assert "managed engine" in help_text


def test_the_gpu_devices_setting_help_does_not_name_one_engine() -> None:
    help_text = SETTINGS_MAP["gpu_devices"].help_text
    assert "llama.cpp" not in help_text
    assert help_text.startswith("Restrict the engine to specific GPU indexes")
