"""The TEI adapter: host gate, builds, roles, memory, argv, and its rerank route."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import re
import stat
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response

from lilbee.app import engine_service
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.engine_params import EMBED_FALLBACK_CTX
from lilbee.providers.fleet import client as fleet_client
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engine_store import EngineNotInstalledError, EngineStore, PinLock
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines import tei as tei_module
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import ENCODER_RUNTIME_BYTES, UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.roles import RoleTarget, WorkerRole
from lilbee.providers.truncation import note_truncated, truncation_scope

FIXTURES = Path(__file__).parent / "conformance" / "fixtures" / "engines"
L4 = ("NVIDIA L4", (8, 9))
GTX = ("NVIDIA GeForce GTX 1070 Ti", (6, 1))


def _caps(
    system: str = "Linux",
    machine: str = "x86_64",
    gpu: tuple[str, tuple[int, int]] | None = None,
    cuda: int = 12080,
) -> HostCaps:
    nvidia = None
    if gpu is not None:
        nvidia = {
            "driver": "570.0",
            "cuda_driver": cuda,
            "gpus": [{"name": gpu[0], "cc": list(gpu[1])}],
        }
    return collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx", "avx2"], "brand_raw": "Test CPU"},
        nvidia=nvidia,
    )


@pytest.mark.parametrize(
    ("compute", "backend"),
    [
        ((7, 5), "cuda-sm75"),
        ((8, 0), "cuda-sm80"),
        ((8, 6), "cuda-sm86"),
        ((8, 9), "cuda-sm89"),
        ((9, 0), "cuda-sm90"),
        ((10, 0), "cuda-sm100"),
        ((12, 0), "cuda-sm120"),
        ((8, 7), "cuda-sm86"),
    ],
)
def test_cuda_host_gets_the_build_of_its_compute_capability(
    compute: tuple[int, int], backend: str
) -> None:
    pick = TEI.pick_build(_caps(gpu=("GPU", compute)))
    assert pick.backend == backend


def test_gpu_older_than_turing_gets_the_cpu_build_with_the_floor_in_the_reason() -> None:
    pick = TEI.pick_build(_caps(gpu=GTX))
    assert pick.backend == "cpu"
    assert "7.5" in pick.reason
    assert "6.1" in pick.reason


def test_gpu_with_no_build_for_its_major_version_gets_the_cpu_build() -> None:
    pick = TEI.pick_build(_caps(gpu=("GPU", (11, 0))))
    assert pick.backend == "cpu"
    assert "11.0" in pick.reason


def test_host_with_no_gpu_gets_the_cpu_build() -> None:
    pick = TEI.pick_build(_caps())
    assert pick.backend == "cpu"
    assert "NVIDIA" in pick.reason


def test_apple_silicon_gets_the_metal_build() -> None:
    assert TEI.pick_build(_caps("Darwin", "arm64")).backend == "metal"


@pytest.mark.parametrize(
    ("system", "machine", "named"),
    [
        ("Windows", "AMD64", "Windows"),
        ("Darwin", "x86_64", "macOS"),
        ("Linux", "aarch64", "architecture"),
    ],
)
def test_host_with_no_build_is_refused_naming_the_cause(
    system: str, machine: str, named: str
) -> None:
    caps = _caps(system, machine, gpu=("GPU", (8, 6)))
    pick = TEI.pick_build(caps)
    gate = TEI.host_gate(caps)
    assert pick.backend is None
    assert named in pick.reason
    assert (gate.ok, gate.reason) == (False, pick.reason)


def test_host_gate_passes_a_host_with_a_build() -> None:
    assert TEI.host_gate(_caps(gpu=L4)).ok


def test_unknown_operating_system_has_no_build() -> None:
    assert TEI.pick_build(_caps("Plan9", "x86_64")).backend is None


def test_serves_embed_and_rerank_only() -> None:
    assert TEI.served_roles == frozenset({WorkerRole.EMBED, WorkerRole.RERANK})
    assert TEI.role_spec(WorkerRole.RERANK, None, None).endpoint_path == "/rerank"
    assert TEI.role_spec(WorkerRole.EMBED, None, None).endpoint_path == "/v1/embeddings"
    with pytest.raises(ValueError, match="serves embed and rerank, not chat"):
        TEI.role_spec(WorkerRole.CHAT, None, None)


def test_choosing_tei_for_chat_is_refused_naming_chat_and_the_roles_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "org/repo/file.gguf")
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.use_for_role(WorkerRole.CHAT, "org/repo/file.gguf", "tei")
    assert str(raised.value) == "Engine tei does not serve chat. It serves embed and rerank."


def test_choosing_auto_for_a_role_skips_the_served_roles_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "chat_model", "org/repo/file.gguf")
    assert engine_service.use_for_role(WorkerRole.CHAT, "org/repo/file.gguf", "auto") == "auto"


def _encoder_folder(root: Path, weight_bytes: int, dtype: str | None) -> Path:
    folder = root / "encoder"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    config = {"torch_dtype": dtype} if dtype is not None else {}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def _encoder_request(model_path: Path, ctx: int = 512) -> MemoryRequest:
    return MemoryRequest(
        model_path=model_path,
        ctx=ctx,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


def test_memory_is_the_weights_plus_the_runtime_allowance_on_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    estimate = TEI.estimate_memory(_encoder_request(folder))
    assert estimate.kind is MemoryKind.FIXED_RESERVE
    assert estimate.device_bytes == estimate.shared_bytes == 4_000 + ENCODER_RUNTIME_BYTES
    assert estimate.size_known is True


def test_a_cuda_build_holds_float32_weights_as_float16(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        2_000 + ENCODER_RUNTIME_BYTES
    )


def test_a_cuda_build_leaves_bfloat16_weights_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    folder = _encoder_folder(tmp_path, 4_000, "bfloat16")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 + ENCODER_RUNTIME_BYTES
    )


def test_a_metal_build_holds_float32_weights_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps("Darwin", "arm64"))
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 + ENCODER_RUNTIME_BYTES
    )


def test_a_host_with_no_build_holds_float32_weights_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps("Windows"))
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 + ENCODER_RUNTIME_BYTES
    )


def test_a_longer_context_leaves_the_charge_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder, 512)) == TEI.estimate_memory(
        _encoder_request(folder, 32768)
    )


def test_a_model_with_no_weights_is_the_unknown_size_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    estimate = TEI.estimate_memory(_encoder_request(tmp_path / "absent"))
    assert estimate.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES
    assert estimate.size_known is False


def test_launch_raises_the_server_batch_limit_to_the_client_batch(tmp_path: Path) -> None:
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert argv == [
        str(tmp_path / "router"),
        "--model-id",
        str(tmp_path / "model"),
        "--hostname",
        "127.0.0.1",
        "--max-client-batch-size",
        "64",
        "--max-batch-tokens",
        "640",
    ]


def test_pin_names_the_picked_build_and_refuses_a_lock_with_no_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    pin = TEI.pin_record()
    assert (pin["engine"], pin["version"], pin["backend"]) == ("tei", "1.9.4", "cuda-sm89")
    assert pin["licenses"] == ["Apache-2.0"]
    with pytest.raises(choice.NoPublishedPackError, match=r"No pack of tei is published yet\."):
        choice.pin_lock(pin)


def test_binary_with_no_published_pack_says_there_is_nothing_to_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    with pytest.raises(
        EngineNotInstalledError,
        match=r"Engine tei is not installed\. No pack of tei is published yet, so there is "
        r"nothing to install\.",
    ):
        TEI.binary()


def test_binary_comes_from_the_store_record_of_the_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    with pytest.raises(EngineNotInstalledError, match="Run lilbee engine install tei"):
        TEI.binary()
    wheel = tmp_path / "tei.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("tei_fx/bin/text-embeddings-router")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("tei_fx-1.dist-info/METADATA", "License-Expression: Apache-2.0\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = PinLock("tei", "1.9.4", "cpu", {}, ("Apache-2.0",), digest, "text-embeddings-router")
    EngineStore(tmp_path / "engines").install_from_wheel(lock, wheel)
    want = tmp_path / "engines" / "tei" / "1.9.4-cpu" / "tei_fx" / "bin"
    assert TEI.binary() == want / "text-embeddings-router"


def test_other_probes_of_the_adapter_answer_without_a_server(tmp_path: Path) -> None:
    probe = TEI.discover_devices(tmp_path / "router")
    assert probe.devices == []
    assert TEI.identity(tmp_path / "router") == "tei@1.9.4"
    assert TEI.name is EngineName.TEI
    assert TEI.supports_architecture("bert")
    assert TEI.model_meta(tmp_path) is None
    assert TEI.assert_loadable("org/repo", "file", None) is None
    assert TEI.runtime_env() == {}
    assert not TEI.check_readback(tmp_path)
    assert not TEI.supports_fit_target(tmp_path)
    assert not TEI.supports_shared_kv(tmp_path)
    assert TEI.swap_extras(None) == {"checkEndpoint": "/health"}  # type: ignore[arg-type]


def test_profile_has_no_tokenize_route_and_names_the_rerank_route() -> None:
    profile = TEI.request_profile(WorkerRole.RERANK)
    assert (profile.tokenize_path, profile.detokenize_path) == (None, None)
    assert (profile.rerank_path, profile.batch_limit) == ("/rerank", 64)
    assert profile.embed_fields == {}
    assert profile.batch_overflow is None
    assert profile.batch_count_refusal is not None
    assert profile.batch_count_refusal.search("batch size 40 > maximum allowed batch size 32")
    assert profile.context_overflow.search("`inputs` must have less than 512 tokens. Given: 600")
    assert profile.context_overflow.search(
        "`inputs` must have less than 2048 characters. Given: 3000"
    )
    assert not profile.context_overflow.search("`inputs` cannot be empty")
    assert (profile.rerank_fields, profile.context_overflow_status) == ({"truncate": False}, 422)


def _recorded_rerank(name: str) -> tuple[dict, str]:
    recording = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    exchange = next(e for e in recording["exchanges"] if e["path"] == "/rerank")
    return exchange["request"], exchange["body"]


@pytest.mark.parametrize("recording", ["tei-cpu", "tei-cuda"])
def test_recorded_rerank_answer_gives_scores_in_candidate_order(recording: str) -> None:
    request, body = _recorded_rerank(recording)
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        request["model"],
        http=http,
        profile=TEI.request_profile(WorkerRole.RERANK),
    )
    scores = client.rerank(request["query"], request["texts"])
    by_index = {item["index"]: item["score"] for item in json.loads(body)}
    assert scores == [by_index[0], by_index[1], by_index[2]]
    assert max(range(3), key=scores.__getitem__) == 1
    assert [r.url.path for r in seen] == [f"/upstream/{request['model']}/rerank"]
    assert json.loads(seen[0].content) == {
        "query": request["query"],
        "texts": request["texts"],
        "truncate": False,
    }


def _base64_vectors(vectors: list[list[float]]) -> dict[str, object]:
    return {
        "data": [
            {
                "embedding": base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode(),
                "index": index,
            }
            for index, vec in enumerate(vectors)
        ]
    }


def _tei_client(handler, role: WorkerRole = WorkerRole.RERANK, **changes) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    profile = dataclasses.replace(TEI.request_profile(role), **changes)
    return LlamaServerClient("http://gpu0", "m", http=http, profile=profile)


def _refuse_past(
    limit: int, seen: list[int], req: httpx.Request, key: str
) -> httpx.Response | None:
    count = len(json.loads(req.content)[key])
    seen.append(count)
    if count > limit:
        return httpx.Response(
            422, json={"error": f"batch size {count} > maximum allowed batch size {limit}"}
        )
    return None


def test_two_hundred_candidates_go_out_in_requests_the_managed_server_accepts() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(64, sizes, req, "texts")
        if refused is not None:
            return refused
        texts = json.loads(req.content)["texts"]
        scored = [{"index": i, "score": float(text)} for i, text in enumerate(texts)]
        return httpx.Response(200, json=scored[::-1])

    client = _tei_client(handler)
    candidates = [str(i) for i in range(200)]
    assert client.rerank("q", candidates) == [float(i) for i in range(200)]
    assert sizes == [64, 64, 64, 8]


def test_two_hundred_texts_embed_in_requests_the_managed_server_accepts() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(64, sizes, req, "input")
        if refused is not None:
            return refused
        texts = json.loads(req.content)["input"]
        vectors = [[float(text)] for text in texts]
        return httpx.Response(200, json=_base64_vectors(vectors))

    client = _tei_client(handler, WorkerRole.EMBED, input_limit_path=None)
    out = client.embed([str(i) for i in range(200)])
    assert [float(vec[0]) for vec in out] == [float(i) for i in range(200)]
    assert sizes == [64, 64, 64, 8]


def test_a_batch_size_refusal_is_a_bad_request_sent_once_not_a_context_overflow() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(32, sizes, req, "input")
        assert refused is not None
        return refused

    client = _tei_client(handler, WorkerRole.EMBED, batch_limit=None, input_limit_path=None)
    with pytest.raises(ProviderError, match=r"too many inputs\. Send fewer inputs") as raised:
        client.embed(["x"] * 33)
    assert raised.value.kind is ProviderErrorKind.BAD_REQUEST
    assert sizes == [33]


def test_the_launch_argv_carries_the_figure_the_profile_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tei_module, "_PROFILE", dataclasses.replace(tei_module._PROFILE, batch_limit=17)
    )
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert argv[argv.index("--max-client-batch-size") + 1] == "17"


def _batch_tokens(tmp_path: Path, ctx: int, target: RoleTarget = RoleTarget.GPU) -> int:
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=ctx,
        target=target,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    return int(argv[argv.index("--max-batch-tokens") + 1])


@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_the_default_chunk_size_bounds_the_warm_up_batch(
    tmp_path: Path, target: RoleTarget
) -> None:
    """The default chunk size of 512 gives 512 chunk tokens plus 128 for a rerank query."""
    assert cfg.chunk_size == 512
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX, target) == 640


def test_a_larger_chunk_size_raises_the_warm_up_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "chunk_size", 1024)
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX) == 1152


@pytest.mark.parametrize("ctx", [512, EMBED_FALLBACK_CTX, 8192, 32768])
def test_the_warm_up_batch_does_not_follow_the_context_per_slot(tmp_path: Path, ctx: int) -> None:
    assert _batch_tokens(tmp_path, ctx) == _batch_tokens(tmp_path, 512)


def test_the_launch_argv_keeps_the_default_truncation_for_long_inputs(tmp_path: Path) -> None:
    """With truncation off, TEI refuses to start when the model limit exceeds the batch."""
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert "--auto-truncate" not in argv
    assert not any(a.startswith("--auto-truncate=") for a in argv)


def test_an_unreadable_platform_stands_and_is_never_auto_selected() -> None:
    gate = TEI.host_gate(_caps("", ""))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_a_cuda_build_runs_a_role_on_the_card_or_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=("GPU", (8, 6))))
    assert TEI.device_targets(Path("/b/router")) == {RoleTarget.GPU, RoleTarget.CPU}


def test_a_cpu_role_hides_every_card_from_the_cuda_build() -> None:
    assert TEI.target_env(RoleTarget.CPU, 1) == {"CUDA_VISIBLE_DEVICES": ""}
    assert TEI.target_env(RoleTarget.GPU, 1) == {}


def test_a_cpu_role_charges_the_fixed_reserve_to_system_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.providers.fleet import planning

    model = tmp_path / "embed.bin"
    model.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda _ref: model)
    sized = planning._estimate_cpu_role(
        WorkerRole.EMBED, "org/embed", engine=TEI, skipped_not_installed={}
    )
    assert sized is not None
    assert sized.est_vram_bytes == 0
    assert sized.est_ram_bytes >= 2 * 1024**3


_OVER_LIMIT = {"error": "Input validation error: `inputs` must have less than 24 tokens. Given: 31"}


def _rerank_exchange(over_limit: bool) -> tuple[list[dict], LlamaServerClient]:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if over_limit and body.get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        return httpx.Response(200, json=[{"index": 0, "score": 0.25}])

    return bodies, _tei_client(handler)


def test_rerank_that_fits_sends_one_request_with_truncation_off_and_no_warning(caplog) -> None:
    bodies, client = _rerank_exchange(over_limit=False)
    with caplog.at_level("WARNING"):
        assert client.rerank("q", ["a"]) == [0.25]
    assert bodies == [{"query": "q", "texts": ["a"], "truncate": False}]
    assert caplog.records == []


def test_rerank_over_the_limit_warns_once_and_is_sent_again_for_the_server_to_truncate(
    caplog,
) -> None:
    bodies, client = _rerank_exchange(over_limit=True)
    with caplog.at_level("WARNING"):
        assert client.rerank("q", ["a"]) == [0.25]
    assert bodies == [
        {"query": "q", "texts": ["a"], "truncate": False},
        {"query": "q", "texts": ["a"]},
    ]
    [record] = caplog.records
    assert record.getMessage().startswith("Rerank query and candidate exceed the reranker's token")
    assert "the server truncates the input" in record.getMessage()
    assert "must have less than 24 tokens. Given: 31" in record.getMessage()


def test_rerank_over_the_character_limit_is_sent_again_for_the_server_to_truncate() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if body.get("truncate") is False:
            error = "`inputs` must have less than 2048 characters. Given: 3000"
            return httpx.Response(422, json={"error": error})
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    assert _tei_client(handler).rerank("q", ["a"]) == [0.5]
    assert len(bodies) == 2


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, "`inputs` cannot be empty"),
        (422, "`texts` must be a list"),
        (400, _OVER_LIMIT["error"]),
    ],
    ids=["empty-input", "other-validation-error", "overflow-body-wrong-status"],
)
def test_rerank_answer_that_is_not_the_over_limit_answer_is_an_error_sent_once(
    status: int, error: str, caplog
) -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(status, json={"error": error})

    with caplog.at_level("WARNING"), pytest.raises(ProviderError) as raised:
        _tei_client(handler).rerank("q", ["a"])
    assert raised.value.kind is not ProviderErrorKind.CONTEXT_OVERFLOW
    assert len(bodies) == 1
    assert caplog.records == []


def test_rerank_gateway_retry_after_an_overflow_still_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    resends = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal resends
        if json.loads(req.content).get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        resends += 1
        if resends == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=[{"index": 0, "score": 0.75}])

    with caplog.at_level("WARNING"):
        assert _tei_client(handler).rerank("q", ["a"]) == [0.75]
    assert resends == 2
    assert len(caplog.records) == 1


def _long_candidates(count: int, long_at: set[int]) -> list[str]:
    return [("long" if i in long_at else "short") + str(i) for i in range(count)]


def _batched_rerank_exchange() -> tuple[list[dict], LlamaServerClient]:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        over = any(text.startswith("long") for text in body["texts"])
        if over and body.get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        scores = [{"index": i, "score": 0.5} for i in range(len(body["texts"]))]
        return httpx.Response(200, json=scores)

    return bodies, _tei_client(handler)


@pytest.mark.parametrize(
    ("count", "long_at", "overflowing"),
    [(200, set(range(200)), 4), (65, {0, 64}, 2), (200, {70, 130}, 2), (200, set(), 0)],
    ids=["every-batch", "both-batches", "some-batches", "no-batch"],
)
def test_rerank_in_batches_warns_once_for_the_whole_call(
    count: int, long_at: set[int], overflowing: int, caplog
) -> None:
    bodies, client = _batched_rerank_exchange()
    with caplog.at_level("WARNING"):
        scores = client.rerank("q", _long_candidates(count, long_at))
    assert scores == [0.5] * count
    batches = -(-count // 64)
    assert len(bodies) == batches + overflowing
    if overflowing == 0:
        assert caplog.records == []
        return
    [record] = caplog.records
    assert f"in {overflowing} of {batches} batches" in record.getMessage()
    assert "must have less than 24 tokens. Given: 31" in record.getMessage()


def test_rerank_overflow_from_an_engine_with_no_extra_fields_is_not_sent_again() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(422, json=_OVER_LIMIT)

    client = _tei_client(handler, rerank_fields={})
    with pytest.raises(ProviderError) as raised:
        client.rerank("q", ["a"])
    assert raised.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert bodies == [{"query": "q", "texts": ["a"]}]


def test_the_422_over_limit_answer_is_a_context_overflow() -> None:
    client = _tei_client(lambda req: httpx.Response(422, json=_OVER_LIMIT), WorkerRole.EMBED)
    with pytest.raises(ProviderError, match="context window") as raised:
        client._raise_for_status(client._http.post("/v1/embeddings", json={}))
    assert raised.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW


def test_the_400_status_is_not_the_over_limit_answer_of_this_engine() -> None:
    client = _tei_client(lambda req: httpx.Response(400, json=_OVER_LIMIT), WorkerRole.EMBED)
    with pytest.raises(ProviderError) as raised:
        client._raise_for_status(client._http.post("/v1/embeddings", json={}))
    assert raised.value.kind is not ProviderErrorKind.CONTEXT_OVERFLOW


_ROUTER_LIMIT = 8


class _Embedder:
    """A fake router as measured: it embeds any input, cuts it silently, counts a word a token.

    ``special`` tokens flank each input, flagged as the router flags them. With ``by_char``
    a character is a token.
    """

    def __init__(self, limit: int = _ROUTER_LIMIT, special: int = 0, by_char: bool = False) -> None:
        self.limit = limit
        self.special = special
        self.by_char = by_char
        self.paths: list[str] = []
        self.timeouts: dict[str, float | None] = {}
        self.tokenized: list[list[str]] = []

    def _tokens(self, text: str) -> list[dict[str, Any]]:
        pieces = list(text) if self.by_char else text.split()
        tokens = [{"id": i, "special": False} for i, _ in enumerate(pieces)]
        specials = [{"id": 0, "special": True} for _ in range(self.special)]
        return specials[:1] + tokens + specials[1:]

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.paths.append(req.url.path)
        self.timeouts[req.url.path] = req.extensions["timeout"]["read"]
        body = json.loads(req.content) if req.content else {}
        if req.url.path.endswith("/info"):
            return httpx.Response(200, json={"max_input_length": self.limit})
        if req.url.path.endswith("/tokenize"):
            self.tokenized.append(body["inputs"])
            return httpx.Response(200, json=[self._tokens(t) for t in body["inputs"]])
        vector = base64.b64encode(np.array([0.5, 0.5], dtype="<f4").tobytes()).decode()
        data = [{"index": i, "embedding": vector} for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})


def _words(count: int) -> str:
    return " ".join(["w"] * count)


def test_embed_inputs_past_the_router_limit_give_one_warning_for_the_call(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        vectors = client.embed([_words(30), _words(3), _words(20)])
    assert len(vectors) == 3
    [record] = caplog.records
    assert "2 of 3" in record.getMessage()
    assert "8-token limit" in record.getMessage()
    assert "the cut text is not in the vectors" in record.getMessage()


def test_embed_inputs_within_the_router_limit_give_no_warning(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(3), _words(4)])
    assert caplog.records == []
    assert router.tokenized == [[_words(3), _words(4)]]


def test_embed_counts_every_input_in_one_request() -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    short, long = "a b", _words(30)
    client.embed([short, long])
    assert router.tokenized == [[short, long]]


class _ByteLevelEmbedder(_Embedder):
    """A router whose tokenizer makes one token per UTF-8 byte, so a character can be many."""

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/tokenize"):
            body = json.loads(req.content)
            return httpx.Response(
                200, json=[[{"id": b} for b in t.encode("utf-8")] for t in body["inputs"]]
            )
        return super().__call__(req)


def test_embed_warns_for_a_text_with_few_characters_and_many_tokens(caplog) -> None:
    """Six characters are twelve bytes: past the limit of eight, though not in characters."""
    client = _tei_client(_ByteLevelEmbedder(), WorkerRole.EMBED)
    text = "é" * 6
    assert len(text) < _ROUTER_LIMIT
    with caplog.at_level("WARNING"):
        client.embed([text])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_embed_asks_the_router_for_its_limit_once() -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_embeds_in_one_scope_give_one_warning_that_adds_their_counts(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"), truncation_scope():
        client.embed([_words(30), _words(2)])
        client.embed([_words(40)])
    [record] = caplog.records
    assert record.getMessage().startswith("2 of 3")
    assert "8-token limit" in record.getMessage()


def test_the_count_of_a_call_with_several_batches_adds_the_cut_inputs_of_each(caplog) -> None:
    """The cut input is in the first batch and the last batch holds none."""
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)] + [_words(2)] * 70)
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 71")


@pytest.mark.parametrize("special", [0, 1, 2, 3])
def test_an_input_the_router_cuts_only_with_its_special_tokens_gives_a_warning(
    special, caplog
) -> None:
    """The router adds *special* tokens, so one character a token is cut at limit - special + 1."""
    router = _Embedder(special=special, by_char=True)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["x" * (_ROUTER_LIMIT - special + 1)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


@pytest.mark.parametrize("special", [0, 1, 2, 3])
def test_an_input_that_fits_with_the_special_tokens_is_not_counted(special, caplog) -> None:
    router = _Embedder(special=special, by_char=True)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["x" * (_ROUTER_LIMIT - special)])
    assert router.tokenized == [["x" * (_ROUTER_LIMIT - special)]]
    assert caplog.records == []


@pytest.mark.parametrize("route", ["/info", "/tokenize"])
def test_a_side_route_request_asks_for_an_answer_within_five_seconds(route) -> None:
    router = _Embedder(special=2, by_char=True)
    http = httpx.Client(transport=httpx.MockTransport(router), base_url="http://gpu0", timeout=60.0)
    client = LlamaServerClient(
        "http://gpu0", "m", http=http, profile=TEI.request_profile(WorkerRole.EMBED)
    )
    client.embed(["x" * 7])
    assert [t for path, t in router.timeouts.items() if path.endswith(route)] == [5.0]


class _RecoveringRouter(_Embedder):
    """A router whose /info answers with the status in ``info_status`` until the test changes it."""

    def __init__(self, info_status: int) -> None:
        super().__init__()
        self.info_status = info_status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/info") and self.info_status != 200:
            self.paths.append(req.url.path)
            return httpx.Response(self.info_status, json={"error": "down"})
        return super().__call__(req)


def test_a_busy_router_pauses_the_check_for_a_cool_down(caplog) -> None:
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1
    assert caplog.records == []


def test_the_check_runs_again_when_the_cool_down_is_over(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


class _FrozenClock:
    """Stands in for the ``time`` module of the client with a clock that never moves."""

    def __init__(self, now: float) -> None:
        self.now = now
        self.sleep = time.sleep

    def monotonic(self) -> float:
        return self.now


def test_a_client_checks_on_its_first_embed_whatever_the_clock_reads(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "time", _FrozenClock(0.5))
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_a_cool_down_of_no_time_ends_at_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "time", _FrozenClock(100.0))
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_a_count_route_answer_with_entries_that_are_not_lists_is_named_as_unexpected(
    caplog,
) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json=[1]))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /tokenize route gave an unexpected "
        "answer (the body is not a list of token lists: [1]). "
        "Embed inputs the router cuts are not reported."
    )


def test_a_router_that_cannot_be_reached_is_asked_again_after_the_cool_down(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/tokenize", httpx.ConnectError("refused"))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.answer = httpx.Response(200, json=[[{"id": 0, "special": False}] * 30])
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert [r.getMessage()[:6] for r in caplog.records] == ["1 of 1"]


def test_a_route_the_router_does_not_have_stops_the_check_for_the_client_after_a_cool_down(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_an_input_of_exactly_the_limit_in_characters_is_counted_and_not_cut(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["a" * _ROUTER_LIMIT])
    assert router.tokenized == [["a" * _ROUTER_LIMIT]]
    assert caplog.records == []


def test_an_input_of_exactly_the_limit_in_tokens_gives_no_warning(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(_ROUTER_LIMIT)])
    assert router.tokenized == [[_words(_ROUTER_LIMIT)]]
    assert caplog.records == []


def test_two_scopes_on_one_client_each_give_one_warning(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        for _ in range(2):
            with truncation_scope():
                client.embed([_words(30)])
                client.embed([_words(40)])
    assert len(caplog.records) == 2


def test_embeds_outside_a_scope_each_warn(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
        client.embed([_words(40)])
    assert len(caplog.records) == 2


def test_a_scope_with_nothing_cut_logs_nothing(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"), truncation_scope():
        client.embed([_words(3)])
    assert caplog.records == []


def test_embed_of_a_dense_text_the_count_finds_within_the_limit_gives_no_warning(caplog) -> None:
    """Thirty-nine characters could be thirty-nine tokens; the router count says twenty."""
    router = _Embedder(limit=38)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(20)])
    assert router.tokenized == [[_words(20)]]
    assert caplog.records == []


@pytest.mark.parametrize(
    "routes",
    [
        {"input_count_path": None, "input_limit_path": None},
        {"input_count_path": None},
    ],
    ids=["no-routes", "limit-route-only"],
)
def test_an_engine_with_no_count_route_gives_no_warning(routes, caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED, **routes)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert router.tokenized == []
    assert caplog.records == []
    assert all(path.endswith(("/info", "/embeddings")) for path in router.paths)


class _BrokenSideRoute(_Embedder):
    """A router whose embeddings work while one side route answers as the test sets."""

    def __init__(self, route: str, answer: httpx.Response | Exception) -> None:
        super().__init__()
        self.route = route
        self.answer = answer

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith(self.route):
            self.paths.append(req.url.path)
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
        return super().__call__(req)


_BROKEN_ANSWERS = {
    "404": httpx.Response(404, json={"error": "no route"}),
    "500": httpx.Response(500, text="boom"),
    "503": httpx.Response(503, text="restarting"),
    "not-json": httpx.Response(200, text="<html>"),
    "connection": httpx.ConnectError("refused"),
}
_BROKEN_INFO = {
    **_BROKEN_ANSWERS,
    "no-limit-key": httpx.Response(200, json={"model_id": "x"}),
    "limit-not-a-number": httpx.Response(200, json={"max_input_length": "many"}),
    "limit-null": httpx.Response(200, json={"max_input_length": None}),
    "body-a-list": httpx.Response(200, json=[1, 2]),
}
_BROKEN_TOKENIZE = {
    **_BROKEN_ANSWERS,
    "body-an-object": httpx.Response(200, json={"tokens": 1}),
    "entries-not-lists": httpx.Response(200, json=[1]),
}


_DEAD_ROUTER_KINDS = {"503", "connection"}


def _skipped_level(kind: str) -> str:
    """A router that is down costs a debug line; an answer the check cannot use is a warning."""
    return "DEBUG" if kind in _DEAD_ROUTER_KINDS else "WARNING"


@pytest.mark.parametrize("kind", list(_BROKEN_INFO))
def test_embed_returns_its_vectors_when_the_limit_route_fails(kind, caplog) -> None:
    router = _BrokenSideRoute("/info", _BROKEN_INFO[kind])
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == _skipped_level(kind)


@pytest.mark.parametrize("kind", list(_BROKEN_TOKENIZE))
def test_embed_returns_its_vectors_when_the_count_route_fails(kind, caplog) -> None:
    router = _BrokenSideRoute("/tokenize", _BROKEN_TOKENIZE[kind])
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == _skipped_level(kind)


def test_a_tokenize_body_of_an_empty_object_is_a_failed_check(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    assert sum("check skipped" in r.getMessage() for r in caplog.records) == 1


def test_an_unexpected_count_route_answer_is_a_warning_naming_the_route_and_the_body(
    caplog,
) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, text='{"tokens": 1}'))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /tokenize route gave an unexpected "
        'answer (the body is not a list of token lists: {"tokens": 1}). '
        "Embed inputs the router cuts are not reported."
    )


def test_a_missing_limit_key_is_a_warning_naming_the_limit_route(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /info route gave an unexpected "
        "answer ('max_input_length'). Embed inputs the router cuts are not reported."
    )


def test_a_router_that_is_down_costs_a_debug_line_and_no_warning(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(503, text="restarting"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_an_absent_route_is_one_warning_that_names_the_404(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, text="no route"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert "/info route" in record.getMessage()
    assert "(HTTP 404): no route" in record.getMessage()


def test_a_dead_router_debug_line_carries_the_reason(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.ConnectError("refused"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "DEBUG"
    assert record.getMessage().startswith("Embed truncation check skipped: ")
    assert "refused" in record.getMessage()


def test_an_absent_route_warning_quotes_at_most_200_characters_of_the_body(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, text="x" * 300))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert "(HTTP 404): " + "x" * 200 + ")" in record.getMessage()


def test_a_bad_token_list_warning_quotes_at_most_200_characters_of_the_body(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json="y" * 300))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert '"' + "y" * 199 + ")" in record.getMessage()


def test_a_404_on_a_side_route_is_a_not_found_error_that_names_the_router() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with pytest.raises(ProviderError) as raised:
        client._raise_for_side_route(httpx.Response(404, text="gone"))
    assert raised.value.kind is ProviderErrorKind.NOT_FOUND
    assert raised.value.provider == client._profile.label
    assert raised.value.provider != ""


def test_an_unexpected_answer_is_asked_again_after_the_cool_down_and_warns_once(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
        client.embed([_words(30)])
        router.answer = httpx.Response(200, json={"max_input_length": _ROUTER_LIMIT})
        client.embed([_words(30)])
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert sum(p.endswith("/info") for p in router.paths) == 3
    assert len(warnings) == 2
    assert warnings[0].getMessage().startswith("Embed truncation check skipped")
    assert warnings[1].getMessage().startswith("1 of 1")


def test_an_unexpected_answer_is_not_asked_again_inside_the_cool_down() -> None:
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_a_second_embed_after_a_failed_limit_read_logs_no_second_skip(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
        client.embed([_words(30)])
    assert sum("check skipped" in r.getMessage() for r in caplog.records) == 1


def test_a_failed_limit_read_is_not_tried_again_by_the_same_client() -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_a_later_embed_with_no_limit_keeps_the_limit_the_scope_has(caplog) -> None:
    with caplog.at_level("WARNING"), truncation_scope():
        note_truncated(2, 8, 512)
        note_truncated(0, 5, None)
    [record] = caplog.records
    assert "2 of 13" in record.getMessage()
    assert "512-token limit" in record.getMessage()


def test_a_cut_with_no_known_limit_never_prints_none(caplog) -> None:
    with caplog.at_level("WARNING"), truncation_scope():
        note_truncated(1, 4, None)
    [record] = caplog.records
    assert "None" not in record.getMessage()
    assert "1 of 4 embed inputs are past the embedder's limit and" in record.getMessage()


def test_the_warning_names_the_chunk_size_setting_and_a_larger_limit_model(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    message = caplog.records[0].getMessage()
    assert "Lower chunk_size, or use an embedding model with a larger input limit." in message


_HANG_S = 3.0
_CHECK_BOUND_S = 1.5


@pytest.fixture
def hung_side_route(monkeypatch: pytest.MonkeyPatch):
    """A real threaded router whose /info or /tokenize answer waits ``_HANG_S``."""
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 0.2)
    release = threading.Event()
    server = HTTPServer(host="127.0.0.1", port=0, threaded=True)
    server.start()
    paths: list[str] = []

    def serve(hung: str) -> LlamaServerClient:
        def answer(request: Request) -> Response:
            paths.append(request.path)
            if request.path.endswith(hung):
                release.wait(_HANG_S)
            if request.path.endswith("/info"):
                return Response(json.dumps({"max_input_length": _ROUTER_LIMIT}))
            inputs = json.loads(request.data)
            if request.path.endswith("/tokenize"):
                tokens = [
                    [{"id": i, "special": False} for i, _ in enumerate(t.split())]
                    for t in inputs["inputs"]
                ]
                return Response(json.dumps(tokens))
            return Response(json.dumps(_base64_vectors([[0.5, 0.5]] * len(inputs["input"]))))

        server.expect_request(re.compile(".*")).respond_with_handler(answer)
        profile = TEI.request_profile(WorkerRole.EMBED)
        base = f"http://127.0.0.1:{server.port}"
        return LlamaServerClient(base, "m", timeout=_HANG_S, profile=profile)

    yield serve, paths
    release.set()
    server.clear()
    server.stop()


def _timed_embed(client: LlamaServerClient) -> float:
    start = time.monotonic()
    vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    return time.monotonic() - start


@pytest.mark.parametrize("hung", ["/info", "/tokenize"])
def test_a_hung_side_route_never_holds_an_embed_back(hung, hung_side_route) -> None:
    serve, paths = hung_side_route
    client = serve(hung)
    first, second = _timed_embed(client), _timed_embed(client)
    assert first < _CHECK_BOUND_S, f"first embed took {first:.2f} s with {hung} hung"
    assert second < _CHECK_BOUND_S, f"second embed took {second:.2f} s with {hung} hung"
    assert sum(path.endswith(hung) for path in paths) == 1
    assert sum(path.endswith("/embeddings") for path in paths) == 2
