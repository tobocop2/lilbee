"""Tier 2 through a second adapter, and the planner pieces that take its fields."""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

import httpx
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA, EngineAdapter, EngineName, LaunchRequest
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole
from tests.conformance.fake_adapter import FAKE
from tests.conformance.harness import Verdict, _check_pin, run_managed

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in binary is a POSIX executable script"
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


@pytest.mark.xfail(
    strict=True,
    reason="no per-platform url, sha256 and macOS floor exist in the repo: the engine is built "
    "from source, so the pin record cannot list release artifacts yet",
)
def test_llama_pin_record_lists_release_artifacts() -> None:
    assert _check_pin(LLAMA.pin_record()) is Verdict.PASS
