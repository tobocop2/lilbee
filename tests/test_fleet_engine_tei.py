"""The TEI adapter: host gate, builds, roles, memory, argv, and its rerank route."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import stat
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from lilbee.app import engine_service
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.engine_params import EMBED_FALLBACK_CTX
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engine_store import EngineNotInstalledError, EngineStore, PinLock
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines import tei as tei_module
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.roles import RoleTarget, WorkerRole

FIXTURES = Path(__file__).parent / "conformance" / "fixtures" / "engines"
L4 = ("NVIDIA L4", (8, 9))
GTX = ("NVIDIA GeForce GTX 1070 Ti", (6, 1))


def _caps(
    system: str = "Linux",
    machine: str = "x86_64",
    gpu: tuple[str, tuple[int, int]] | None = None,
    cuda: int = 12080,
) -> HostCaps:
    nvml = None
    if gpu is not None:
        nvml = {
            "driver": "570.0",
            "cuda_driver": cuda,
            "gpus": [{"name": gpu[0], "cc": list(gpu[1])}],
        }
    return collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx", "avx2"], "brand_raw": "Test CPU"},
        nvml=nvml,
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


def test_memory_charges_the_fixed_reserve_whatever_the_file_size(tmp_path: Path) -> None:
    estimates = []
    for size in (1_000, 50_000):
        model = tmp_path / f"model-{size}.bin"
        model.write_bytes(b"\0" * size)
        request = MemoryRequest(
            model_path=model,
            ctx=512,
            slots=1,
            gpu_layers=0,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        estimates.append(TEI.estimate_memory(request))
    assert estimates[0] == estimates[1]
    assert estimates[0].kind is MemoryKind.FIXED_RESERVE
    assert estimates[0].device_bytes == estimates[0].shared_bytes == 2 * 1024**3


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
        "2176",
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

    client = _tei_client(handler, WorkerRole.EMBED)
    out = client.embed([str(i) for i in range(200)])
    assert [float(vec[0]) for vec in out] == [float(i) for i in range(200)]
    assert sizes == [64, 64, 64, 8]


def test_a_batch_size_refusal_is_a_bad_request_sent_once_not_a_context_overflow() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(32, sizes, req, "input")
        assert refused is not None
        return refused

    client = _tei_client(handler, WorkerRole.EMBED, batch_limit=None)
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
    """The default chunk size of 512 gives 512 * 4 chunk tokens plus 128 for a rerank query."""
    assert cfg.chunk_size == 512
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX, target) == 2176


def test_a_larger_chunk_size_raises_the_warm_up_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "chunk_size", 1024)
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX) == 4224


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
    assert TEI.target_env(RoleTarget.CPU) == {"CUDA_VISIBLE_DEVICES": ""}
    assert TEI.target_env(RoleTarget.GPU) == {}


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


def test_rerank_422_that_is_not_an_overflow_is_an_error_sent_once() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(422, json={"error": "`inputs` cannot be empty"})

    with pytest.raises(ProviderError) as raised:
        _tei_client(handler).rerank("q", ["a"])
    assert raised.value.kind is not ProviderErrorKind.CONTEXT_OVERFLOW
    assert len(bodies) == 1


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
