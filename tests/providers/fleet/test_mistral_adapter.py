"""The mistral.rs adapter: build pick, host gate, argv, profile and the role refusal."""

from __future__ import annotations

import hashlib
import json
import shlex
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps, swap_config
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines import mistral
from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import (
    NoPublishedPackError,
    installed_engines,
    pin_lock,
)
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.roles import WorkerRole

L4 = ("NVIDIA L4", (8, 9))
A40 = ("NVIDIA A40", (8, 6))
TURING = ("NVIDIA T4", (7, 5))


def caps(
    system: str,
    machine: str,
    *,
    gpus: list[tuple[str, tuple[int, int]]] | None = None,
    cuda: tuple[int, int] = (12, 8),
    nvml: dict[str, Any] | None = None,
) -> host_caps.HostCaps:
    """A HostCaps built by the product's own reader from raw readings."""
    if nvml is None and gpus:
        nvml = {
            "driver": "570.0",
            "cuda_driver": cuda[0] * 1000 + cuda[1] * 10,
            "gpus": [{"name": n, "cc": list(cc)} for n, cc in gpus],
        }
    return host_caps.collect(
        system=system,
        machine=machine,
        mac_ver="15.1" if system == "Darwin" else "",
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvml=nvml,
        npu_devices=[],
    )


def request(tmp_path: Path, **overrides: Any) -> LaunchRequest:
    fields: dict[str, Any] = {
        "binary": tmp_path / "mistralrs",
        "model_path": tmp_path / "model",
        "devices": (),
        "n_gpu_layers": 99,
        "slots": 2,
        "ctx_per_slot": 4096,
    }
    return LaunchRequest(**{**fields, **overrides})


@pytest.mark.parametrize(
    ("gpu", "cuda", "backend"),
    [
        (L4, (12, 8), "cuda128-sm89"),
        (A40, (12, 8), "cuda128-sm86"),
        (L4, (13, 0), "cuda128-sm89"),
    ],
)
def test_linux_with_a_matching_card_and_driver_picks_the_cuda_archive(
    gpu: tuple[str, tuple[int, int]], cuda: tuple[int, int], backend: str
) -> None:
    assert MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[gpu], cuda=cuda)).backend == backend


def test_an_old_driver_picks_the_cpu_archive_and_names_the_floor() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[L4], cuda=(12, 4)))
    assert pick.backend == "cpu"
    assert "12.8" in pick.reason
    assert "12.4" in pick.reason


def test_a_card_without_an_archive_picks_the_cpu_archive_and_names_its_capability() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[TURING]))
    assert pick.backend == "cpu"
    assert "7.5" in pick.reason


def test_cards_of_two_capabilities_pick_the_cpu_archive() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[L4, A40]))
    assert pick.backend == "cpu"
    assert "differ" in pick.reason


def test_two_cards_of_one_capability_pick_their_cuda_archive() -> None:
    assert MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[L4, L4])).backend == "cuda128-sm89"


def test_a_host_without_a_card_picks_the_cpu_archive() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64"))
    assert (pick.backend, "no NVIDIA GPU" in pick.reason) == ("cpu", True)


def test_an_unreadable_driver_picks_the_cpu_archive() -> None:
    nvml = {"driver": None, "cuda_driver": None, "gpus": None}
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", nvml=nvml))
    assert pick.backend == "cpu"
    assert "could not be read" in pick.reason


def test_windows_picks_the_cpu_archive_and_names_windows() -> None:
    pick = MISTRAL.pick_build(caps("Windows", "AMD64", gpus=[L4]))
    assert pick.backend == "cpu"
    assert "Windows" in pick.reason


def test_apple_silicon_picks_the_metal_archive() -> None:
    assert MISTRAL.pick_build(caps("Darwin", "arm64")).backend == "metal"


@pytest.mark.parametrize(
    ("system", "machine", "named"),
    [("Linux", "aarch64", "arm64"), ("Darwin", "x86_64", "x86_64"), ("Windows", "ARM64", "arm64")],
)
def test_a_host_without_an_archive_is_refused_and_offered_nothing(
    system: str, machine: str, named: str
) -> None:
    host = caps(system, machine)
    gate = MISTRAL.host_gate(host)
    assert not gate.ok
    assert named in gate.reason
    pick = MISTRAL.pick_build(host)
    assert (pick.backend, pick.reason) == (None, gate.reason)


def test_a_host_that_cannot_be_read_neither_passes_nor_refuses() -> None:
    host = caps("Plan9", "mips")
    gate = MISTRAL.host_gate(host)
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)
    assert MISTRAL.pick_build(host).backend is None


def test_chat_and_vision_serve_on_the_cpu_with_the_cpu_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    for role in (WorkerRole.CHAT, WorkerRole.VISION):
        spec = MISTRAL.role_spec(role, None, None)
        argv = MISTRAL.launch_argv(spec, request(tmp_path))
        assert argv == [
            str(tmp_path / "mistralrs"),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            PORT_SLOT,
            "--cpu",
            "-m",
            str(tmp_path / "model"),
        ]


def test_embed_names_the_embedding_command_after_the_serve_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.EMBED, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path))
    assert argv[1:8] == ["serve", "--host", "127.0.0.1", "--port", PORT_SLOT, "embedding", "--cpu"]


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION])
def test_the_swap_command_puts_the_port_before_the_subcommand(
    role: WorkerRole, tmp_path: Path
) -> None:
    argv = MISTRAL.launch_argv(MISTRAL.role_spec(role, None, None), request(tmp_path))
    words = shlex.split(swap_config._command_line(argv, 4321))
    assert words.index("--port") < (words.index("embedding") if "embedding" in words else 99)
    assert words[words.index("--port") + 1] == "4321"
    assert words.count("--port") == 1
    assert words[:2] == [str(tmp_path / "mistralrs"), "serve"]


def test_a_gguf_goes_by_its_file_name_with_the_folder_as_the_model(tmp_path: Path) -> None:
    model = tmp_path / "models" / "m-Q4_K_M.gguf"
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=model))
    assert argv[-4:] == ["-m", str(model.parent), "-f", "m-Q4_K_M.gguf"]


@pytest.mark.parametrize(
    "host",
    [
        pytest.param(("Linux", "x86_64", [L4]), id="cuda"),
        pytest.param(("Darwin", "arm64", None), id="metal"),
    ],
)
@pytest.mark.parametrize("devices", [(), (0,)])
def test_a_gpu_build_turns_paged_attention_off_and_never_forces_the_cpu(
    host: tuple[str, str, list[tuple[str, tuple[int, int]]] | None],
    devices: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, machine, gpus = host
    monkeypatch.setattr(runnable, "session_host", lambda: caps(system, machine, gpus=gpus))
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, devices=devices))
    assert "--cpu" not in argv
    assert "--gpu-memory-utilization" not in argv
    assert argv[argv.index("--paged-attn") + 1 : argv.index("--paged-attn") + 2] == ["off"]


def test_a_cpu_build_forces_the_cpu_even_when_a_device_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, devices=(0,)))
    assert "--cpu" in argv
    assert "--paged-attn" not in argv


def test_rerank_has_no_spec_and_is_not_served() -> None:
    assert WorkerRole.RERANK not in MISTRAL.served_roles
    with pytest.raises(ProviderError, match="does not serve rerank"):
        MISTRAL.role_spec(WorkerRole.RERANK, None, None)


def test_the_profile_has_no_tokenize_route_and_matches_the_recorded_overflow() -> None:
    profile = MISTRAL.request_profile(WorkerRole.CHAT)
    assert (profile.tokenize_path, profile.detokenize_path, profile.rerank_path) == (
        None,
        None,
        None,
    )
    recorded = "Prompt sequence length is greater than 40960, perhaps consider using `truncate`?"
    assert profile.context_overflow.search(recorded)
    assert not profile.batch_overflow.search(recorded)


def test_an_oversize_embed_is_cut_by_characters_and_calls_no_tokenize_route() -> None:
    seen: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}")
        seen.append((req.url.path, body))
        if "tokenize" in req.url.path:
            return httpx.Response(404, json={"error": "not found"})
        vector = "AAAAPwAAgD4="  # base64 of the floats 0.5 and 0.25
        return httpx.Response(200, json={"data": [{"embedding": vector}] * len(body["input"])})

    http = httpx.Client(base_url="http://engine.invalid", transport=httpx.MockTransport(handler))
    client = LlamaServerClient(
        "http://engine.invalid",
        "m",
        http=http,
        token_cap=4,
        profile=MISTRAL.request_profile(WorkerRole.EMBED),
    )
    assert len(client.embed(["x" * 100])) == 1
    assert not [path for path, _ in seen if "tokenize" in path]
    assert max(len(body["input"][0]) for _, body in seen) <= 4 * 3


def test_the_memory_estimate_is_a_fixed_reserve_of_the_file_size(tmp_path: Path) -> None:
    model = tmp_path / "m.bin"
    model.write_bytes(b"x" * 1000)
    memory = MemoryRequest(model, 4096, 1, 99, True, cfg.kv_cache_type)
    estimate = MISTRAL.estimate_memory(memory)
    assert (estimate.kind, estimate.device_bytes, estimate.shared_bytes) == (
        MemoryKind.FIXED_RESERVE,
        1000,
        1000,
    )
    missing = MISTRAL.estimate_memory(
        MemoryRequest(tmp_path / "gone", 4096, 1, 99, True, cfg.kv_cache_type)
    )
    assert missing.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES


def test_the_probes_answer_unknown_without_running_the_binary(tmp_path: Path) -> None:
    probe = MISTRAL.discover_devices(tmp_path / "never-run")
    assert (probe.devices, probe.spoke_protocol) == ([], False)
    assert not MISTRAL.check_readback(tmp_path)
    assert not MISTRAL.supports_fit_target(tmp_path)
    assert not MISTRAL.supports_shared_kv(tmp_path)
    assert MISTRAL.model_meta(tmp_path) is None
    assert MISTRAL.assert_loadable("repo", "file", None) is None
    assert MISTRAL.swap_extras(None) == {}  # type: ignore[arg-type]
    assert MISTRAL.supports_architecture("anything")
    assert "mistral.rs" in MISTRAL.identity(tmp_path / "mistralrs")


def test_the_model_cache_sits_under_the_models_folder(tmp_path: Path) -> None:
    saved = cfg.models_dir
    cfg.models_dir = tmp_path / "models"
    try:
        assert MISTRAL.runtime_env() == {"HF_HUB_CACHE": str(tmp_path / "models" / "mistralrs")}
    finally:
        cfg.models_dir = saved


def make_pack(path: Path) -> str:
    """A wheel with one executable under pkg/bin and an MIT licence; returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/mistralrs")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_engine_counts_as_installed_only_with_a_record_and_finds_its_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    assert EngineName.MISTRAL not in installed_engines()
    with pytest.raises(ProviderError, match=r"engine install mistral\.rs"):
        MISTRAL.binary()
    sha = make_pack(tmp_path / "pack.whl")
    monkeypatch.setattr(mistral, "PACK_SHA256", sha)
    pin = MISTRAL.pin_record()
    assert (pin["engine"], pin["backend"]) == ("mistral.rs", "cpu")
    lock = PinLock(pin["engine"], pin["version"], pin["backend"], {}, ("MIT",), sha)
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, tmp_path / "pack.whl")
    assert EngineName.MISTRAL in installed_engines()
    assert MISTRAL.binary().parts[-3:] == ("pkg", "bin", "mistralrs")


def test_a_pin_with_no_published_pack_refuses_a_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    with pytest.raises(NoPublishedPackError, match=r"No pack of mistral\.rs is published yet\."):
        pin_lock(MISTRAL.pin_record())


@pytest.mark.parametrize(
    ("host", "reason"),
    [
        (lambda: caps("Windows", "AMD64"), "The release has no CUDA archive for Windows."),
        (
            lambda: caps("Linux", "x86_64", gpus=[L4], cuda=(12, 4)),
            "this driver offers CUDA 12.4, so the CPU archive runs.",
        ),
        (lambda: caps("Linux", "x86_64"), "so the CPU archive runs."),
    ],
)
def test_the_reason_for_a_cpu_archive_reaches_the_user_through_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: Any, reason: str
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "detect", host)
    sha = make_pack(tmp_path / "pack.whl")
    monkeypatch.setattr(mistral, "PACK_SHA256", sha)
    built = engine_service.install("mistral.rs", tmp_path / "pack.whl")
    assert (built.backend, reason in built.note) == ("cpu", True)


def test_a_host_with_a_matching_card_adds_no_note_to_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    host = lambda: caps("Linux", "x86_64", gpus=[L4])  # noqa: E731
    monkeypatch.setattr(runnable, "detect", host)
    sha = make_pack(tmp_path / "pack.whl")
    monkeypatch.setattr(mistral, "PACK_SHA256", sha)
    built = engine_service.install("mistral.rs", tmp_path / "pack.whl")
    assert (built.backend, built.note) == ("cuda128-sm89", "")


def test_an_install_reads_the_host_now_and_a_list_reads_the_cached_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    reads: list[bool] = []
    has_gpu = [False]

    def detect_now() -> host_caps.HostCaps:
        reads.append(has_gpu[0])
        return caps("Linux", "x86_64", gpus=[L4] if has_gpu[0] else [])

    monkeypatch.setattr(runnable, "detect", detect_now)
    monkeypatch.setattr(host_caps, "detect", detect_now)
    runnable.session_host()
    has_gpu[0] = True
    sha = make_pack(tmp_path / "pack.whl")
    monkeypatch.setattr(mistral, "PACK_SHA256", sha)
    installed_engines()
    assert reads == [False]
    built = engine_service.install("mistral.rs", tmp_path / "pack.whl")
    assert (built.backend, reads) == ("cuda128-sm89", [False, True])
    assert EngineName.MISTRAL in installed_engines()
    assert reads == [False, True]


def _count_processes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Count every ``subprocess.Popen`` started from now on."""
    started: list[object] = []
    real = subprocess.Popen

    def counting(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        started.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", counting)
    return started


def test_a_cold_list_with_nothing_installed_starts_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(mistral, "PACK_SHA256", "a" * 64)
    runnable.session_host.cache_clear()
    started = _count_processes(monkeypatch)
    assert runnable.session_host.cache_info().currsize == 0
    assert EngineName.MISTRAL not in installed_engines()
    assert (started, runnable.session_host.cache_info().currsize) == ([], 0)
    # Control: the same instrument sees the host read when it happens.
    runnable.session_host()
    assert started != []
    runnable.session_host.cache_clear()


def test_a_cold_list_with_a_build_in_the_store_reads_the_host_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    reads: list[bool] = []

    def detect_now() -> host_caps.HostCaps:
        reads.append(True)
        return caps("Linux", "x86_64", gpus=[L4])

    monkeypatch.setattr(runnable, "detect", detect_now)
    monkeypatch.setattr(host_caps, "detect", detect_now)
    sha = make_pack(tmp_path / "pack.whl")
    monkeypatch.setattr(mistral, "PACK_SHA256", sha)
    engine_service.install("mistral.rs", tmp_path / "pack.whl")
    runnable.session_host.cache_clear()
    reads.clear()
    assert EngineName.MISTRAL in installed_engines()
    installed_engines()
    assert reads == [True]
    runnable.session_host.cache_clear()


def test_rerank_is_refused_for_the_engine_before_its_install_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "reranker_model", "org/repo/file.gguf")
    with pytest.raises(engine_service.EngineRefusedError, match="does not serve rerank") as raised:
        engine_service.use_for_role(WorkerRole.RERANK, "org/repo/file.gguf", "mistral.rs")
    assert "chat, embed and vision" in str(raised.value)
