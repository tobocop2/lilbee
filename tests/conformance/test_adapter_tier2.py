"""Tier 2 through a second adapter, and the planner pieces that take its fields."""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest, host_caps
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA, EngineAdapter, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import PinError, pin_lock
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole
from tests._published import render_table, wheel_name
from tests.conformance.fake_adapter import FAKE
from tests.conformance.harness import Verdict, _check_pin, run_managed

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in binary is a POSIX executable script"
)


def _host(system: str, machine: str) -> host_caps.HostCaps:
    return host_caps.collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvml=None,
    )


_ROWS = ("M1", "M2", "M3", "M4", "M5", "M6")


@_POSIX_ONLY
def test_fake_adapter_passes_the_managed_rows_through_its_own_pin() -> None:
    verdicts = run_managed(FAKE.binary(), pin=FAKE.pin_record(), timeout=15.0)
    assert set(verdicts) == set(_ROWS)
    assert verdicts["M2"] is Verdict.FALLBACK
    assert all(v in {Verdict.PASS, Verdict.FALLBACK} for v in verdicts.values())


def test_fake_adapter_is_a_second_engine() -> None:
    assert FAKE.name != EngineName.LLAMA
    assert FAKE.formats.isdisjoint(LLAMA.formats)


def test_fake_argv_shares_no_flag_with_llama(tmp_path: Path) -> None:
    request = LaunchRequest(
        binary=Path("/b/engine"),
        model_path=tmp_path / "m.bin",
        devices=(),
        n_gpu_layers=0,
        slots=2,
        ctx_per_slot=1024,
    )
    fake = FAKE.launch_argv(FAKE.role_spec(WorkerRole.CHAT, None, None), request)
    llama = LLAMA.launch_argv(LLAMA.role_spec(WorkerRole.CHAT, None, None), request)
    flags = {arg for arg in fake if arg.startswith("--")}
    assert flags
    assert flags.isdisjoint({arg for arg in llama if arg.startswith("--")})
    assert "2048" in fake


def test_extras_of_the_second_engine_reach_its_swap_entry() -> None:
    plain = InstanceLaunch(role=WorkerRole.CHAT, argv=["/b"], env_overrides={}, model="m")
    launch = InstanceLaunch(
        role=WorkerRole.CHAT,
        argv=["/b"],
        env_overrides={},
        model="m",
        swap_extras=FAKE.swap_extras(plain),
    )
    entry = json.loads(build_swap_config([launch], {launch.model_id: 9}))["models"][launch.model_id]
    assert entry["checkEndpoint"] == "/live"
    assert entry["useModelName"] == "default"
    assert entry["cmd"].startswith("/b ")
    bare = json.loads(build_swap_config([plain], {plain.model_id: 9}))["models"][plain.model_id]
    assert "checkEndpoint" not in bare


def test_swap_extras_survive_the_state_file() -> None:
    launch = InstanceLaunch(
        role=WorkerRole.CHAT,
        argv=["/b"],
        env_overrides={},
        model="m",
        swap_extras={"checkEndpoint": "/live"},
    )
    assert InstanceLaunch.from_state(launch.to_state()).swap_extras == {"checkEndpoint": "/live"}


def _client_posting(seen: list[tuple[str, dict]], adapter: EngineAdapter) -> LlamaServerClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("embeddings"):
            vector = base64.b64encode(struct.pack("<2f", 0.5, 0.25)).decode()
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector}]})
        return httpx.Response(200, json={"content": "text"})

    http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return LlamaServerClient(
        "http://x", "m", http=http, profile=adapter.request_profile(WorkerRole.EMBED)
    )


def test_client_sends_the_fields_of_the_adapter_it_was_given() -> None:
    seen: list[tuple[str, dict]] = []
    client = _client_posting(seen, FAKE)
    assert len(client.embed(["a"])) == 1
    assert seen
    body = seen[0][1]
    assert "embd_normalize" not in body
    assert body["encoding_format"] == "base64"


def test_client_sends_llama_fields_for_llama() -> None:
    seen: list[tuple[str, dict]] = []
    client = _client_posting(seen, LLAMA)
    assert len(client.embed(["a"])) == 1
    assert seen[0][1]["embd_normalize"] == -1


def test_client_detokenize_uses_the_adapter_path() -> None:
    seen: list[tuple[str, dict]] = []
    client = _client_posting(seen, FAKE)
    assert client._detokenize([1]) == "text"
    assert seen[0][0].endswith("/v1/detokenize")


def _llama_table(*keys: str, version: str | None = None) -> dict[str, Any]:
    """The table the generator writes for llama.cpp wheels in cells *keys*.

    The engines repo names the wheel 1.<upstream build number>.<release build>, so the default
    is the real name of the b12032 wheel, whatever the lilbee version is.
    """
    wheel_version = version or "1.12032.1"
    wheels = [
        (wheel_name(EngineName.LLAMA, wheel_version, key), f"{n}" * 64, n)
        for n, key in enumerate(keys, 1)
    ]
    return render_table(wheels)


def test_llama_pin_record_lists_release_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host("Darwin", "arm64"))
    table = _llama_table("macos-arm64-cpu", "linux-x86_64-cpu")
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)
    pin = LLAMA.pin_record()
    assert _check_pin(pin) is Verdict.PASS
    row = pin["platforms"]["macos-arm64-cpu"]
    assert (row["sha256"], row["size"], row["min_os"]) == ("1" * 64, 1, "14.0")
    assert row["url"].startswith("https://")
    assert set(pin["platforms"]) == {"macos-arm64-cpu"}


def test_llama_pin_record_takes_the_cell_of_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host("Linux", "x86_64"))
    monkeypatch.setattr(
        engine_manifest, "MANIFESTS", _llama_table("macos-arm64-cpu", "linux-x86_64-cpu")
    )
    pin = LLAMA.pin_record()
    assert _check_pin(pin) is Verdict.PASS
    assert pin["platforms"]["linux-x86_64-cpu"]["sha256"] == "2" * 64
    assert set(pin["platforms"]) == {"linux-x86_64-cpu"}


def test_llama_pin_record_skips_the_wheels_of_another_engine_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host("Linux", "x86_64"))
    table = _llama_table("linux-x86_64-cpu", version="1.12033.1")
    assert table
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)
    assert LLAMA.pin_record()["platforms"] == {}


def test_llama_pin_record_with_an_empty_table_lists_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    pin = LLAMA.pin_record()
    assert pin["platforms"] == {}
    assert _check_pin(pin) is Verdict.FAIL
    with pytest.raises(PinError, match="the pin record is not usable"):
        pin_lock(pin)
