"""The contract methods of the OpenVINO and Lemonade adapters answer what a launch reads."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.openvino import OPENVINO
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.roles import WorkerRole

ADAPTERS = [OPENVINO, LEMONADE]
BINARY = Path("/engines/bin/tool")


def _unread() -> HostCaps:
    from lilbee.providers.fleet import host_caps

    return host_caps.collect(system="Plan9", machine="mips", cpu_info={}, nvidia=None)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
def test_an_unread_platform_is_unknown_not_refused(adapter) -> None:
    gate = adapter.host_gate(_unread())
    assert (gate.ok, gate.known) == (True, False)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
def test_binary_comes_from_the_store_or_says_how_to_install(
    adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(choice, "store_tool", lambda *_a: BINARY)
    assert adapter.binary() == BINARY
    monkeypatch.setattr(choice, "store_tool", lambda *_a: None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    expected = (
        "lemonade-server.ai"
        if hasattr(adapter, "user_binary")
        else f"lilbee engine install {adapter.name.value}"
    )
    with pytest.raises(ProviderError, match=expected):
        adapter.binary()


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
def test_the_identity_names_the_engine_and_the_binary(adapter) -> None:
    assert adapter.identity(BINARY).startswith(adapter.name.value)
    assert str(BINARY) in adapter.identity(BINARY)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
def test_the_pin_names_a_licence_and_no_pack_is_published(adapter) -> None:
    pin = adapter.pin_record()
    assert pin["licenses"]
    assert (pin["engine"], pin["sha256"], pin["platforms"]) == (adapter.name.value, None, {})


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
def test_the_fixed_answers_of_the_contract(adapter) -> None:
    assert adapter.supports_architecture("llama")
    assert adapter.discover_devices(BINARY).devices == []
    assert adapter.check_readback(BINARY) is False
    assert adapter.supports_fit_target(BINARY) is False
    assert adapter.supports_shared_kv(BINARY) is False
    assert adapter.assert_loadable("repo", "file.gguf", None) is None
    assert adapter.runtime_env() == {}
    assert adapter.request_profile(WorkerRole.CHAT).label


def test_lemonade_charges_a_fixed_reserve(tmp_path: Path) -> None:
    request = MemoryRequest(
        model_path=tmp_path / "m.gguf",
        ctx=2048,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    estimate = LEMONADE.estimate_memory(request)
    assert estimate.kind is MemoryKind.FIXED_RESERVE
    assert estimate.device_bytes > 0


def test_openvino_sizes_a_gguf_model_with_the_gguf_estimator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[ModelFormat] = []

    def fake(fmt: ModelFormat, request: MemoryRequest) -> str:
        seen.append(fmt)
        return "sized"

    monkeypatch.setattr("lilbee.providers.fleet.engines.openvino.estimate_memory", fake)
    request = MemoryRequest(
        model_path=tmp_path / "m.gguf",
        ctx=2048,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    assert OPENVINO.estimate_memory(request) == "sized"
    assert seen == [ModelFormat.GGUF]


def test_lemonade_does_not_read_a_model_header(tmp_path: Path) -> None:
    assert LEMONADE.model_meta(tmp_path / "m.gguf") is None


def test_lemonade_health_check_uses_the_recorded_live_route() -> None:
    launch = InstanceLaunch(role=WorkerRole.CHAT, argv=[], env_overrides={}, model="m")
    assert LEMONADE.swap_extras(launch) == {
        "checkEndpoint": "/live",
        "useModelName": "user.lilbee-chat",
    }
    assert OPENVINO.swap_extras(launch) == {}


def test_lemonade_targets_the_npu_only() -> None:
    from lilbee.providers.roles import RoleTarget

    assert LEMONADE.device_targets(BINARY) == {RoleTarget.NPU}


def test_openvino_reads_the_model_header_with_the_gguf_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.openvino.read_gguf_metadata", lambda p: {"arch": "x"}
    )
    assert OPENVINO.model_meta(BINARY) == {"arch": "x"}
