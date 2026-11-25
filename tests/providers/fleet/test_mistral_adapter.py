"""The mistral.rs adapter: build pick, host gate, argv, profile and the role refusal."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.cli import app as cli_app
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import engine_manifest, host_caps, planning, swap_config
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    SplitRefusedError,
)
from lilbee.providers.fleet.engines.choice import (
    NoPublishedPackError,
    installed_engines,
    pin_lock,
)
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.gpu_backends.nvidia import NvidiaCard
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.gguf_meta import snapshot_entry_of_blob
from lilbee.providers.roles import RoleTarget, WorkerRole
from lilbee.server.routes import engines as routes
from tests._gguf_fixture import cut_file_of
from tests._litestar_test_client import create_test_client
from tests._published import publish
from tests._safetensors import safetensors_bytes
from tests._stated_sizes import state_size

L4 = ("NVIDIA L4", (8, 9))
A40 = ("NVIDIA A40", (8, 6))
TURING = ("NVIDIA T4", (7, 5))
GIB = 1 << 30


def caps(
    system: str,
    machine: str,
    *,
    gpus: list[tuple[str, tuple[int, int]]] | None = None,
    cuda: tuple[int, int] = (12, 8),
    nvidia: dict[str, Any] | None = None,
) -> host_caps.HostCaps:
    """A HostCaps built by the product's own reader from raw readings."""
    if nvidia is None and gpus:
        nvidia = {
            "driver": "570.0",
            "cuda_driver": cuda[0] * 1000 + cuda[1] * 10,
            "gpus": [{"name": n, "cc": list(cc)} for n, cc in gpus],
        }
    return host_caps.collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvidia=nvidia,
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


A100 = ("NVIDIA A100", (8, 0))
H100 = ("NVIDIA H100", (9, 0))
B200 = ("NVIDIA B200", (10, 0))
RTX5090 = ("NVIDIA GeForce RTX 5090", (12, 0))
ORIN = ("NVIDIA Orin", (8, 7))
THOR = ("NVIDIA Thor", (11, 0))
OLD_DRIVER_REASON = "Update the driver"
SHIPPED_CELLS = [
    "linux-x86_64-cpu",
    "linux-x86_64-cuda-sm100",
    "linux-x86_64-cuda-sm120",
    "linux-x86_64-cuda-sm80",
    "linux-x86_64-cuda-sm86",
    "linux-x86_64-cuda-sm89",
    "linux-x86_64-cuda-sm90",
    "macos-arm64-metal",
]


@pytest.mark.parametrize(
    ("card", "backend"),
    [
        (A100, "cuda-sm80"),
        (A40, "cuda-sm86"),
        (ORIN, "cuda-sm86"),
        (L4, "cuda-sm89"),
        (H100, "cuda-sm90"),
        (B200, "cuda-sm100"),
        (RTX5090, "cuda-sm120"),
    ],
)
def test_a_card_picks_the_newest_cell_that_its_machine_code_runs_on(
    card: tuple[str, tuple[int, int]], backend: str
) -> None:
    host = caps("Linux", "x86_64", gpus=[card], cuda=(12, 8))
    pick = MISTRAL.pick_build(host)
    assert (pick.backend, pick.reason) == (backend, "")
    assert (MISTRAL.host_gate(host).ok, MISTRAL.host_gate(host).reason) == (True, "")


def test_a_newer_driver_than_the_floor_still_picks_the_cuda_cell() -> None:
    assert MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[L4], cuda=(13, 0))).backend == (
        "cuda-sm89"
    )


@pytest.mark.parametrize(
    ("gpus", "backend"),
    [
        ([L4, A40], "cuda-sm86"),
        ([A40, L4], "cuda-sm86"),
        ([A40, A40], "cuda-sm86"),
        ([A100, A40, L4], "cuda-sm80"),
    ],
)
def test_cards_take_the_cell_that_the_weakest_card_runs(
    gpus: list[tuple[str, tuple[int, int]]], backend: str
) -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=gpus, cuda=(12, 8)))
    assert pick.backend == backend


@pytest.mark.parametrize("gpus", [[L4, H100], [H100, L4], [A100, B200]])
def test_cards_of_different_generations_pick_cpu_and_say_so(
    gpus: list[tuple[str, tuple[int, int]]],
) -> None:
    """Machine code of one major version does not run on another."""
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=gpus, cuda=(12, 8)))
    assert pick.backend == "cpu"
    assert "different generations" in pick.reason
    assert all(name in pick.reason for name, _ in gpus)


def test_a_card_below_every_cell_picks_cpu_and_says_why() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[TURING], cuda=(12, 8)))
    assert pick.backend == "cpu"
    assert "NVIDIA T4" in pick.reason and "7.5" in pick.reason and "8.0" in pick.reason


def test_a_turing_card_beside_a_newer_one_picks_cpu() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[TURING, L4], cuda=(12, 8)))
    assert pick.backend == "cpu"
    assert "NVIDIA T4" in pick.reason


def test_a_card_newer_than_every_cell_picks_cpu_and_names_its_compute() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", gpus=[THOR], cuda=(12, 8)))
    assert pick.backend == "cpu"
    assert "NVIDIA Thor" in pick.reason and "11.0" in pick.reason


def test_a_driver_below_the_build_floor_picks_cpu_and_says_to_update() -> None:
    host = caps("Linux", "x86_64", gpus=[L4], cuda=(12, 4))
    pick = MISTRAL.pick_build(host)
    assert pick.backend == "cpu"
    assert "NVIDIA L4" in pick.reason and "12.4" in pick.reason and "12.8" in pick.reason
    assert OLD_DRIVER_REASON in pick.reason
    gate = MISTRAL.host_gate(host)
    assert (gate.ok, gate.reason) == (True, pick.reason)


def test_the_pack_table_has_a_cell_for_every_published_card_family() -> None:
    assert sorted(engine_manifest.MANIFESTS["mistral.rs"]["0.9.5"]) == SHIPPED_CELLS


def test_the_pick_of_a_cuda_host_resolves_to_a_pack_digest() -> None:
    host = caps("Linux", "x86_64", gpus=[L4], cuda=(12, 8))
    backend = MISTRAL.pick_build(host).backend
    assert engine_manifest.pack_digest("mistral.rs", "0.9.5", host, backend)


def test_a_host_without_a_card_picks_the_cpu_archive() -> None:
    pick = MISTRAL.pick_build(caps("Linux", "x86_64"))
    assert (pick.backend, pick.reason) == (
        "cpu",
        "This host has no NVIDIA GPU, so the CPU archive runs.",
    )


def test_an_unreadable_driver_picks_the_cpu_archive() -> None:
    nvidia = {"driver": None, "cuda_driver": None, "gpus": None}
    pick = MISTRAL.pick_build(caps("Linux", "x86_64", nvidia=nvidia))
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
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    assert MISTRAL.launch_argv(spec, request(tmp_path)) == [
        str(tmp_path / "mistralrs"),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        PORT_SLOT,
        "text",
        "--cpu",
        "--max-model-len",
        "4096",
        "-m",
        str(tmp_path / "model"),
    ]


def test_vision_passes_no_context_because_the_multimodal_loader_refuses_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.VISION, None, None)
    assert MISTRAL.launch_argv(spec, request(tmp_path)) == [
        str(tmp_path / "mistralrs"),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        PORT_SLOT,
        "multimodal",
        "--cpu",
        "-m",
        str(tmp_path / "model"),
    ]


def test_chat_passes_the_planned_context_of_each_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, ctx_per_slot=2048, slots=4))
    assert argv[argv.index("--max-model-len") + 1] == "2048"
    assert argv.index("--max-model-len") > argv.index("text")


def test_a_gguf_chat_launch_passes_the_context_before_the_file_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=tmp_path / "m.gguf"))
    assert argv[-6:] == ["--max-model-len", "4096", "-m", str(tmp_path), "-f", "m.gguf"]


def test_embed_names_the_embedding_command_after_the_serve_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(WorkerRole.EMBED, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path))
    assert argv[1:8] == ["serve", "--host", "127.0.0.1", "--port", PORT_SLOT, "embedding", "--cpu"]
    assert "--max-model-len" not in argv


def _split_like_the_swap(command: str) -> list[str]:
    """Split *command* back into argv by the rules the swap proxy uses on this platform."""
    if sys.platform != "win32":
        return shlex.split(command)
    import ctypes
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    count = ctypes.c_int()
    found = shell32.CommandLineToArgvW(command, ctypes.byref(count))
    assert found, "CommandLineToArgvW refused the command"
    try:
        return [found[i] for i in range(count.value)]
    finally:
        kernel32.LocalFree(found)


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION])
def test_the_swap_command_puts_the_port_before_the_subcommand(
    role: WorkerRole, tmp_path: Path
) -> None:
    argv = MISTRAL.launch_argv(MISTRAL.role_spec(role, None, None), request(tmp_path))
    words = _split_like_the_swap(swap_config._command_line(argv, 4321))
    assert words.index("--port") < words.index(
        {WorkerRole.EMBED: "embedding", WorkerRole.VISION: "multimodal"}.get(role, "text")
    )
    assert words[words.index("--port") + 1] == "4321"
    assert words.count("--port") == 1
    assert words[:2] == [str(tmp_path / "mistralrs"), "serve"]


def test_a_gguf_goes_by_its_file_name_with_the_folder_as_the_model(tmp_path: Path) -> None:
    model = tmp_path / "models" / "m-Q4_K_M.gguf"
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=model))
    assert argv[-4:] == ["-m", str(model.parent), "-f", "m-Q4_K_M.gguf"]


@pytest.mark.parametrize("blob_dir", ["blobs", "blobs/93"])
def test_a_cached_gguf_goes_by_its_snapshot_folder_and_file_name(
    tmp_path: Path, blob_dir: str
) -> None:
    repo = tmp_path / "models" / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / blob_dir / "9375c1fd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "Qwen3-4B-Q4_K_M.gguf").symlink_to(blob)
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=blob))
    assert argv[-4:] == ["-m", str(snapshot), "-f", "Qwen3-4B-Q4_K_M.gguf"]


def test_a_snapshot_that_links_into_the_shared_blob_store_still_names_its_blob(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    shared = models / "blobs" / "93" / "9375c1fd"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"gguf")
    repo = models / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / "blobs" / "93" / "9375c1fd"
    blob.parent.mkdir(parents=True)
    os.link(shared, blob)
    alias = repo / "blobs" / "7485fe6f"
    alias.symlink_to(Path("..") / ".." / "blobs" / "93" / "9375c1fd")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "Qwen3-4B-Q4_K_M.gguf").symlink_to(Path("..") / ".." / "blobs" / "7485fe6f")
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=blob))
    assert argv[-4:] == ["-m", str(snapshot), "-f", "Qwen3-4B-Q4_K_M.gguf"]


def test_a_repo_blob_that_links_into_the_shared_blob_store_names_its_snapshot_file(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    shared = models / "blobs" / "93" / "9375c1fd"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"gguf")
    repo = models / "models--Qwen--Qwen3-4B-GGUF"
    entry = repo / "blobs" / "7485fe6f"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(Path("..") / ".." / "blobs" / "93" / "9375c1fd")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "Qwen3-4B-Q4_K_M.gguf").symlink_to(Path("..") / ".." / "blobs" / "7485fe6f")
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=entry))
    assert argv[-4:] == ["-m", str(snapshot), "-f", "Qwen3-4B-Q4_K_M.gguf"]


def test_a_gguf_in_a_snapshot_subfolder_goes_by_that_folder_and_file_name(tmp_path: Path) -> None:
    repo = tmp_path / "models" / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / "blobs" / "93" / "9375c1fd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    folder = repo / "snapshots" / "rev1" / "Q4_K_M"
    folder.mkdir(parents=True)
    (folder / "Qwen3-4B-Q4_K_M.gguf").symlink_to(blob)
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=blob))
    assert argv[-4:] == ["-m", str(folder), "-f", "Qwen3-4B-Q4_K_M.gguf"]


@pytest.mark.parametrize("quant", ["Q2_K", "Q4_K_M", "Q8_0"])
def test_each_blob_of_a_snapshot_with_several_quants_goes_by_its_own_file(
    tmp_path: Path, quant: str
) -> None:
    repo = tmp_path / "models" / "models--Qwen--Qwen3-4B-GGUF"
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    blobs = {}
    for name in ("Q2_K", "Q4_K_M", "Q8_0"):
        blob = repo / "blobs" / f"hash-{name}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(name.encode())
        (snapshot / f"Qwen3-4B-{name}.gguf").symlink_to(blob)
        blobs[name] = blob
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=blobs[quant]))
    assert argv[-4:] == ["-m", str(snapshot), "-f", f"Qwen3-4B-{quant}.gguf"]


def test_a_blob_that_no_snapshot_names_stays_as_given(tmp_path: Path) -> None:
    repo = tmp_path / "models--org--repo"
    blob = repo / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    other = repo / "blobs" / "def"
    other.write_bytes(b"other")
    (snapshot / "other.gguf").symlink_to(other)
    assert snapshot_entry_of_blob(blob) == blob


def _split_snapshot(tmp_path: Path) -> tuple[Path, list[Path]]:
    """A cached three-shard GGUF: the snapshot folder and its shard links, first shard first."""
    repo = tmp_path / "models" / "models--Qwen--Qwen3-4B-GGUF"
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    links = []
    for number in (1, 2, 3):
        blob = repo / "blobs" / f"hash-shard-{number}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"gguf")
        link = snapshot / f"Qwen3-4B-Q4_K_M-0000{number}-of-00003.gguf"
        link.symlink_to(blob)
        links.append(link)
    return snapshot, links


def test_a_split_gguf_launches_by_its_first_shard_snapshot_link(tmp_path: Path) -> None:
    snapshot, shards = _split_snapshot(tmp_path)
    spec = MISTRAL.role_spec(WorkerRole.CHAT, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=shards[0]))
    assert argv[-4:] == ["-m", str(snapshot), "-f", "Qwen3-4B-Q4_K_M-00001-of-00003.gguf"]


def test_the_first_shard_snapshot_link_of_a_split_gguf_stays_as_given(tmp_path: Path) -> None:
    _, shards = _split_snapshot(tmp_path)
    assert snapshot_entry_of_blob(shards[0]) == shards[0]


def test_a_blob_without_a_snapshot_folder_stays_as_given(tmp_path: Path) -> None:
    blob = tmp_path / "models--org--repo" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    assert snapshot_entry_of_blob(blob) == blob


@pytest.mark.parametrize("devices", [(), (0,)])
def test_the_metal_build_turns_paged_attention_off_and_never_forces_the_cpu(
    devices: tuple[int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Darwin", "arm64"))
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
    assert estimate.size_known is True
    missing = MISTRAL.estimate_memory(
        MemoryRequest(tmp_path / "gone", 4096, 1, 99, True, cfg.kv_cache_type)
    )
    assert missing.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES
    assert missing.size_known is False


@pytest.mark.usefixtures("stated_sizes")
def test_a_cut_off_gguf_is_an_unknown_size_not_its_partial_bytes(tmp_path: Path) -> None:
    model = cut_file_of(tmp_path / "cut.gguf", declared_bytes=4096, size=60)
    estimate = MISTRAL.estimate_memory(MemoryRequest(model, 4096, 1, 99, True, cfg.kv_cache_type))
    assert estimate.size_known is False
    assert estimate.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES


def test_the_memory_estimate_of_a_folder_prices_its_weights_and_its_cache(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(safetensors_bytes(1000))
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    estimate = MISTRAL.estimate_memory(MemoryRequest(folder, 100, 2, 99, True, cfg.kv_cache_type))
    cache = 2 * 2 * 1 * 4 * 2 * 100 * 2
    assert estimate.device_bytes == 1000 + cache + 512 * 1024**2
    assert estimate.size_known is True


def _vision_folder(tmp_path: Path, config: dict[str, Any]) -> Path:
    folder = tmp_path / "vlm"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(safetensors_bytes(1000))
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


_VLM_TEXT = {"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1}


def test_a_vision_folder_with_a_large_window_is_priced_at_the_planned_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planner asks for the vision role, so the request carries whatever it carries."""
    priced: dict[str, int] = {}
    for name, window in (("small", 4096), ("large", 262144)):
        text = {**_VLM_TEXT, "head_dim": 4, "max_position_embeddings": window}
        (tmp_path / name).mkdir()
        folder = _vision_folder(tmp_path / name, {"text_config": text})
        monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda _ref, f=folder: f)
        monkeypatch.setattr(planning, "vision_mmproj", lambda _ref: None)
        placed = planning._estimate_role(WorkerRole.VISION, "org/vlm", engine=MISTRAL, slots=2)
        priced[name] = placed.est_vram_bytes
    assert priced["large"] == priced["small"]


def _vision_launch(
    tmp_path: Path, architectures: list[str], window: int = 8192, nested: bool = False
) -> list[str]:
    config: dict[str, Any] = {"architectures": architectures}
    config.update({"text_config": {_WINDOW: window}} if nested else {_WINDOW: window})
    folder = _vision_folder(tmp_path, config)
    spec = MISTRAL.role_spec(WorkerRole.VISION, None, None)
    return MISTRAL.launch_argv(spec, request(tmp_path, model_path=folder))


_WINDOW = "max_position_embeddings"
_QWEN = "Qwen3_5ForConditionalGeneration"


def _flag(argv: list[str]) -> str | None:
    return argv[argv.index("--max-model-len") + 1] if "--max-model-len" in argv else None


@pytest.mark.parametrize(
    ("window", "nested", "flag"),
    [
        (8192, False, "4096"),
        (2048, False, "2048"),
        (4096, True, "4096"),
        (1024, True, "1024"),
    ],
)
def test_the_vision_context_flag_never_exceeds_the_window_of_the_model(
    window: int, nested: bool, flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    assert _flag(_vision_launch(tmp_path, [_QWEN], window, nested)) == flag


def test_a_capped_vision_architecture_with_no_declared_window_gets_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    folder = _vision_folder(tmp_path, {"architectures": [_QWEN]})
    spec = MISTRAL.role_spec(WorkerRole.VISION, None, None)
    assert _flag(MISTRAL.launch_argv(spec, request(tmp_path, model_path=folder))) is None


@pytest.mark.parametrize("window", ["many", 0, True])
def test_a_capped_vision_architecture_with_a_window_that_is_no_count_gets_no_flag(
    window: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    folder = _vision_folder(tmp_path, {"architectures": [_QWEN], _WINDOW: window})
    spec = MISTRAL.role_spec(WorkerRole.VISION, None, None)
    assert _flag(MISTRAL.launch_argv(spec, request(tmp_path, model_path=folder))) is None


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
        "Gemma4UnifiedForCausalLM",
        "MuseGlimmerForConditionalGeneration",
    ],
)
def test_a_vision_architecture_whose_loader_caps_the_window_gets_the_context_flag(
    architecture: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    argv = _vision_launch(tmp_path, [architecture])
    assert argv[argv.index("--max-model-len") + 1] == "4096"


@pytest.mark.parametrize(
    "architectures",
    [["Idefics3ForConditionalGeneration"], ["Qwen3_5ForConditionalGeneration", "Other"], []],
)
def test_a_vision_architecture_whose_loader_refuses_the_flag_gets_none(
    architectures: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    assert "--max-model-len" not in _vision_launch(tmp_path, architectures)


@pytest.mark.parametrize("config", ["not json", "[]", '{"architectures": "Gemma4ForCausalLM"}'])
def test_a_vision_folder_with_an_unreadable_architecture_gets_no_context_flag(
    config: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    folder = tmp_path / "vlm"
    folder.mkdir()
    (folder / "config.json").write_text(config, encoding="utf-8")
    spec = MISTRAL.role_spec(WorkerRole.VISION, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, model_path=folder))
    assert "--max-model-len" not in argv


def _no_read() -> list[NvidiaCard]:
    raise AssertionError("a CPU build read the driver")


def test_a_host_with_no_card_offers_the_cpu_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    assert MISTRAL.device_targets(Path("mistralrs")) == {RoleTarget.CPU}


def test_the_swap_extras_name_the_one_served_model() -> None:
    assert MISTRAL.swap_extras(None) == {"useModelName": "default"}  # type: ignore[arg-type]


def test_a_cpu_build_reads_no_card_and_lists_no_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    monkeypatch.setattr(nvidia_mod, "cards", _no_read)
    probe = MISTRAL.discover_devices(tmp_path / "never-run")
    assert (probe.devices, probe.spoke_protocol) == ([], False)


def test_a_metal_build_reads_no_card_and_lists_no_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Darwin", "arm64"))
    monkeypatch.setattr(nvidia_mod, "cards", _no_read)
    probe = MISTRAL.discover_devices(tmp_path / "never-run")
    assert (probe.devices, probe.spoke_protocol) == ([], False)


def test_the_probes_answer_unknown_without_running_the_binary(tmp_path: Path) -> None:
    assert not MISTRAL.check_readback(tmp_path)
    assert not MISTRAL.supports_fit_target(tmp_path)
    assert not MISTRAL.supports_shared_kv(tmp_path)
    assert MISTRAL.model_meta(tmp_path) is None
    assert MISTRAL.assert_loadable("repo", "file", None) is None
    assert MISTRAL.supports_architecture("anything")
    assert "mistral.rs" in MISTRAL.identity(tmp_path / "mistralrs")


def make_pack(path: Path) -> str:
    """A wheel with one executable under pkg/bin and an MIT licence; returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/mistralrs")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_missing_install_is_not_found_for_the_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    with pytest.raises(ProviderError) as exc:
        MISTRAL.binary()
    assert exc.value.kind is ProviderErrorKind.NOT_FOUND


def test_the_engine_counts_as_installed_only_with_a_record_and_finds_its_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    assert EngineName.MISTRAL not in installed_engines()
    with pytest.raises(ProviderError, match=r"engine install mistral\.rs"):
        MISTRAL.binary()
    sha = make_pack(tmp_path / "pack.whl")
    publish(monkeypatch, EngineName.MISTRAL, sha)
    pin = MISTRAL.pin_record()
    assert (pin["engine"], pin["backend"]) == ("mistral.rs", "cpu")
    lock = PinLock(pin["engine"], pin["version"], pin["backend"], {}, ("MIT",), sha, "mistralrs")
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, tmp_path / "pack.whl")
    assert EngineName.MISTRAL in installed_engines()
    assert MISTRAL.binary().parts[-3:] == ("pkg", "bin", "mistralrs")


def test_a_pin_with_no_published_pack_refuses_a_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    with pytest.raises(NoPublishedPackError, match=r"No pack of mistral\.rs is published yet\."):
        pin_lock(MISTRAL.pin_record())


@pytest.mark.parametrize(
    ("host", "reason"),
    [
        (lambda: caps("Windows", "AMD64"), "The release has no CUDA archive for Windows."),
        (lambda: caps("Linux", "x86_64", gpus=[L4], cuda=(12, 4)), OLD_DRIVER_REASON),
        (lambda: caps("Linux", "x86_64"), "so the CPU archive runs."),
    ],
)
def test_the_reason_for_a_cpu_archive_reaches_the_user_through_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: Any, reason: str
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "detect", host)
    sha = make_pack(tmp_path / "pack.whl")
    publish(monkeypatch, EngineName.MISTRAL, sha)
    built = engine_service.install("mistral.rs", tmp_path / "pack.whl")
    assert (built.backend, reason in built.note) == ("cpu", True)


def _cuda_install_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA host with the cpu pack published and a download that copies it."""
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "allow_http_engines", True)
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(runnable, "detect", lambda: caps("Linux", "x86_64", gpus=[L4]))
    wheel = tmp_path / "pack" / "mistral.whl"
    wheel.parent.mkdir()
    publish(monkeypatch, EngineName.MISTRAL, make_pack(wheel))

    def fake_fetch(lock: Any, cell: str, *, dest: Path, **_signals: Any) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        copy = dest / wheel.name
        copy.write_bytes(wheel.read_bytes())
        return copy

    monkeypatch.setattr(engine_service, "fetch_frozen", fake_fetch)


def test_a_cuda_host_installs_the_cuda_pack_and_the_service_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cuda_install_host(tmp_path, monkeypatch)
    built = engine_service.install("mistral.rs")
    assert (built.backend, built.note) == ("cuda-sm89", "")


def test_a_cuda_host_installs_the_cuda_pack_and_the_cli_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cuda_install_host(tmp_path, monkeypatch)
    result = CliRunner().invoke(cli_app, ["--json", "engine", "install", "mistral.rs"])
    body = json.loads(result.stdout.strip().splitlines()[-1])
    assert (result.exit_code, body["installed"], body["note"]) == (
        0,
        "mistral.rs 0.9.5 (cuda-sm89)",
        "",
    )


def test_a_cuda_host_installs_the_cuda_pack_and_rest_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cuda_install_host(tmp_path, monkeypatch)
    with create_test_client([routes.engine_install_route], debug=False) as client:
        res = client.post("/api/engines/install", json={"engine": "mistral.rs"})
    body = res.json()
    assert (res.status_code, body["backend"], body["note"]) == (200, "cuda-sm89", "")


def test_a_cuda_host_installs_the_cuda_pack_and_mcp_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cuda_install_host(tmp_path, monkeypatch)
    body = mcp_server.engine_install_tool("mistral.rs")
    assert (body["backend"], body["note"]) == ("cuda-sm89", "")


def test_a_cuda_host_lists_mistral_as_installable_and_then_as_the_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cuda_install_host(tmp_path, monkeypatch)
    assert "mistral.rs" in engine_service.engine_list().payload()["not_installed"]
    engine_service.install("mistral.rs")
    builds = engine_service.engine_list().payload()["builds"]
    assert [(b["engine"], b["backend"]) for b in builds] == [("mistral.rs", "cuda-sm89")]


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
    publish(monkeypatch, EngineName.MISTRAL, sha, key="linux-x86_64-cuda-sm89")
    installed_engines()
    assert reads == [False]
    built = engine_service.install("mistral.rs", tmp_path / "pack.whl")
    assert (built.backend, built.note, reads) == ("cuda-sm89", "", [False, True])
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


@pytest.mark.real_host_read
def test_a_cold_list_with_nothing_installed_starts_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    publish(monkeypatch, EngineName.MISTRAL, "a" * 64)
    runnable._read_host.cache_clear()
    started = _count_processes(monkeypatch)
    assert runnable._read_host.cache_info().currsize == 0
    assert EngineName.MISTRAL not in installed_engines()
    assert (started, runnable._read_host.cache_info().currsize) == ([], 0)
    # Control: the same instrument sees the host read when it happens.
    runnable.session_host()
    assert started != []
    runnable._read_host.cache_clear()


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
    publish(monkeypatch, EngineName.MISTRAL, sha)
    engine_service.install("mistral.rs", tmp_path / "pack.whl")
    runnable._read_host.cache_clear()
    reads.clear()
    assert EngineName.MISTRAL in installed_engines()
    installed_engines()
    assert reads == [True]
    runnable._read_host.cache_clear()


def test_rerank_is_refused_for_the_engine_before_its_install_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "reranker_model", "org/repo/file.gguf")
    with pytest.raises(engine_service.EngineRefusedError, match="does not serve rerank") as raised:
        engine_service.use_for_role(WorkerRole.RERANK, "org/repo/file.gguf", "mistral.rs")
    assert "chat, embed and vision" in str(raised.value)


def test_binary_on_a_host_with_no_cell_says_the_engine_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = {"url": "https://example.test/x.whl", "sha256": "b" * 64, "size": 1, "min_os": None}
    record = {**MISTRAL.pin_record(), "sha256": "a" * 64, "platforms": {"macos-arm64-cpu": row}}
    monkeypatch.setattr(type(MISTRAL), "pin_record", lambda _self: record)
    monkeypatch.setattr(cfg, "models_dir", tmp_path)
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64"))
    with pytest.raises(ProviderError, match="not installed"):
        MISTRAL.binary()


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION])
@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_device_flags_follow_the_subcommand(
    role: WorkerRole, target: RoleTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the subcommand the engine ignores --cpu (20536 MiB measured)."""
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Windows", "x86_64"))
    spec = MISTRAL.role_spec(role, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, target=target))
    subcommand = argv.index(
        {WorkerRole.EMBED: "embedding", WorkerRole.VISION: "multimodal"}.get(role, "text")
    )
    assert argv.index("--cpu") == subcommand + 1


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.VISION])
@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_a_cuda_host_runs_the_gpu_target_on_the_cuda_pack_and_the_cpu_target_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: WorkerRole, target: RoleTarget
) -> None:
    host = caps("Linux", "x86_64", gpus=[L4], cuda=(12, 8))
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pick = MISTRAL.pick_build(host)
    assert (pick.backend, pick.reason) == ("cuda-sm89", "")
    assert MISTRAL.device_targets(Path("mistralrs")) == {RoleTarget.CPU, RoleTarget.GPU}
    spec = MISTRAL.role_spec(role, None, None)
    argv = MISTRAL.launch_argv(spec, request(tmp_path, target=target))
    subcommand = argv.index("multimodal" if role is WorkerRole.VISION else "text")
    if target is RoleTarget.CPU:
        assert argv[subcommand + 1] == "--cpu"
        assert "--paged-attn" not in argv
    else:
        assert argv[subcommand + 1 : subcommand + 3] == ["--paged-attn", "off"]
        assert "--cpu" not in argv


# ---- a role across several cards ----


def _host_with_cards(
    monkeypatch: pytest.MonkeyPatch, system: str = "Linux", *, cards: int = 2
) -> None:
    gpus = [L4] * cards if system != "Darwin" else None
    machine = "arm64" if system == "Darwin" else "x86_64"
    monkeypatch.setattr(runnable, "session_host", lambda: caps(system, machine, gpus=gpus))


def _split_argv(
    tmp_path: Path, split: tuple[int, ...], role: WorkerRole = WorkerRole.CHAT, **over: Any
) -> list[str]:
    model = over.pop("model_path", tmp_path / "m.gguf")
    req = request(tmp_path, model_path=model, devices=tuple(range(len(split))), tensor_split=split)
    return MISTRAL.launch_argv(MISTRAL.role_spec(role, None, None), req)


def _one_card_argv(tmp_path: Path, role: WorkerRole = WorkerRole.CHAT, **over: Any) -> list[str]:
    return _split_argv(tmp_path, (1,), role, **over)


@pytest.mark.parametrize(
    ("system", "cards", "label"),
    [("Darwin", 2, "Metal"), ("Windows", 2, "CPU"), ("Linux", 0, "CPU"), ("Linux", 2, "CUDA")],
)
def test_a_build_that_cannot_split_refuses_a_split(
    system: str, cards: int, label: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host_with_cards(monkeypatch, system, cards=cards)
    with pytest.raises(
        ProviderError, match=rf"mistral\.rs cannot split m\.gguf.*{label}.*llama\.cpp engine"
    ):
        _split_argv(tmp_path, (1, 1))


def test_a_two_card_launch_on_a_cuda_host_is_refused_by_the_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1000)
    _host_with_cards(monkeypatch)
    monkeypatch.setattr(type(MISTRAL), "binary", lambda _self: tmp_path / "mistralrs")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr(planning, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    cards = {i: FleetDevice("CUDA", i, f"card{i}", 24 * GIB, 22 * GIB) for i in (0, 1)}
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
    with pytest.raises(ProviderError, match=r"mistral\.rs cannot split m\.gguf.*CUDA build"):
        planning._launch_for(plan, "ref", cards, engine=MISTRAL)


def test_the_metal_pin_record_declares_the_mit_licence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Darwin", "arm64"))
    assert MISTRAL.pin_record()["licenses"] == ["MIT"]


@pytest.mark.parametrize("cards", [0, 1, 2])
@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_every_launch_sets_no_environment_variable(target: RoleTarget, cards: int) -> None:
    assert MISTRAL.target_env(target, cards) == {}
    assert MISTRAL.runtime_env() == {}


def _estimate(model: Path, split: tuple[int, ...]) -> Any:
    request_ = MemoryRequest(model, 4096, 1, 99, True, cfg.kv_cache_type, tensor_split=split)
    return MISTRAL.estimate_memory(request_)


@pytest.mark.parametrize("split", [(), (1,), (3, 1)])
def test_a_split_estimate_is_the_whole_file_with_no_per_card_share(
    split: tuple[int, ...], tmp_path: Path
) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 3200)
    estimate = _estimate(model, split)
    assert estimate.per_device == ()
    assert (estimate.kind, estimate.device_bytes) == (MemoryKind.FIXED_RESERVE, 3200)


_stated = pytest.mark.usefixtures("stated_sizes")


def _staged_chat_peak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    card_gib: tuple[int, ...],
    *,
    system: str = "Linux",
    model_gib: int = 30,
    engine: Any = MISTRAL,
) -> Any:
    """Stage a chat model of *model_gib* on *engine* and return the planner's peak estimator."""
    model = tmp_path / "m.gguf"
    state_size(model, model_gib * GIB)
    _host_with_cards(monkeypatch, system, cards=len(card_gib))
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr(planning, "_placement_estimate_ctx", lambda _r, _p, _m, *, engine: 4096)
    return planning._peak_estimator({WorkerRole.CHAT: "ref"}, {WorkerRole.CHAT: engine})


def _placed_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    card_gib: tuple[int, ...],
    *,
    system: str = "Linux",
    model_gib: int = 30,
    engine: Any = MISTRAL,
) -> Any:
    """Place a chat model of *model_gib* on cards of *card_gib* with *engine*."""
    from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement

    peak = _staged_chat_peak(
        tmp_path, monkeypatch, card_gib, system=system, model_gib=model_gib, engine=engine
    )
    chat = ModelPlacementInput(role=WorkerRole.CHAT, est_vram_bytes=model_gib * GIB)
    devices = [(index, gib * GIB) for index, gib in enumerate(card_gib)]
    return plan_placement([chat], devices, estimate_peak=peak, refuse_oversize=True)


def test_the_ladder_of_split_ratios_keeps_the_proportional_and_the_shifted_ratio() -> None:
    from lilbee.providers.fleet.placement import _split_ratio_candidates

    remaining = {0: float(24 * GIB), 1: float(16 * GIB)}
    assert _split_ratio_candidates([0, 1], remaining) == ((24, 16), (102, 58))


@pytest.mark.parametrize(
    ("system", "cards", "label"), [("Darwin", 2, "Metal"), ("Linux", 2, "CUDA")]
)
@_stated
def test_placement_offers_no_split_on_a_build_that_cannot_split(
    system: str,
    cards: int,
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    placed = _placed_on(tmp_path, monkeypatch, (24,) * cards, system=system, engine=MISTRAL)
    assert placed.instances == ()
    assert placed.unplaceable_roles == (WorkerRole.CHAT,)
    assert f"cannot split m.gguf across 2 cards on the {label} build" in caplog.text


def _sweep_calls(refuse_every: int, message: str = "refused") -> tuple[int, int]:
    """Run the split sweep over 30 cards; return (estimates run, splits refused)."""
    from lilbee.providers.fleet.placement import ModelPlacementInput, _place_split

    seen = {"asked": 0, "estimated": 0, "refused": 0}

    def estimate(_role: WorkerRole, _ratio: tuple[int, ...]) -> tuple[int, ...]:
        seen["asked"] += 1
        if refuse_every and seen["asked"] % refuse_every == 0:
            seen["refused"] += 1
            raise SplitRefusedError(message)
        seen["estimated"] += 1
        return ()  # the wrong length: the candidate never fits

    remaining = {idx: float((40 + idx) * GIB) for idx in range(30)}
    chat = ModelPlacementInput(role=WorkerRole.CHAT, est_vram_bytes=30 * GIB)
    assert _place_split(chat, remaining, estimate) is None
    return seen["estimated"], seen["refused"]


def test_the_sweep_stops_after_the_estimate_budget_however_many_splits_are_refused() -> None:
    from lilbee.providers.fleet.placement import _MAX_SPLIT_ESTIMATES

    estimated_alone, _ = _sweep_calls(refuse_every=0)
    assert estimated_alone == _MAX_SPLIT_ESTIMATES
    estimated, refused = _sweep_calls(refuse_every=3)
    assert refused > 0
    assert estimated == _MAX_SPLIT_ESTIMATES


def test_a_refusal_is_logged_once_for_each_message(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    _sweep_calls(refuse_every=3, message="only evenly")
    refusals = [r.getMessage() for r in caplog.records if "-card split for" in r.getMessage()]
    assert len(refusals) == 1
    assert refusals[0].endswith("chat: only evenly")


def test_a_refused_ratio_leaves_the_other_ratios_of_that_card_count_to_try() -> None:
    from lilbee.providers.fleet.placement import ModelPlacementInput, _place_split

    asked: list[tuple[int, ...]] = []

    def estimate(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        asked.append(ratio)
        if len(asked) == 1:
            raise SplitRefusedError("refused")
        return (GIB, GIB)

    remaining = {0: float(24 * GIB), 1: float(16 * GIB)}
    chat = ModelPlacementInput(role=WorkerRole.CHAT, est_vram_bytes=30 * GIB)
    placed = _place_split(chat, remaining, estimate)
    assert placed is not None
    assert placed.plan.devices == (0, 1)
    assert placed.plan.tensor_split == asked[1]
    assert len(asked) == 2 and asked[0] != asked[1]


def _unequal_cards() -> list[FleetDevice]:
    return [
        FleetDevice("CUDA", i, f"gpu{i}", gib * GIB, gib * GIB) for i, gib in enumerate((24, 16))
    ]


def _saved_chat_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, model_gib: int = 30
) -> Any:
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    _staged_chat_peak(tmp_path, monkeypatch, (24, 16), model_gib=model_gib)
    chat = planning._estimate_role(WorkerRole.CHAT, "ref", engine=MISTRAL, device_count=2, slots=1)
    return planning._placement_or_auto(
        PlacementSpec.from_json(raw),
        [chat],
        {WorkerRole.CHAT: "ref"},
        _unequal_cards(),
        engines={WorkerRole.CHAT: MISTRAL},
        unified_budget=None,
        host_cards=planning.HostCards(_unequal_cards()),
    )


@_stated
def test_a_saved_placement_the_split_rule_refuses_falls_back_to_automatic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    raw = '{"chat": {"devices": [0, 1], "tensor_split": [3, 2]}}'
    placed, applied, why = _saved_chat_placement(tmp_path, monkeypatch, raw)
    assert applied is False
    assert "mistral.rs cannot split m.gguf" in why
    assert all(instance.tensor_split == () for instance in placed.instances)
    assert "mistral.rs cannot split m.gguf" in caplog.text
    assert "using automatic placement instead" in caplog.text


@_stated
def test_a_saved_placement_with_a_role_off_the_cards_refuses_that_role_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.providers.fleet.placement_spec import TargetRefusedError

    raw = '{"chat": {"devices": [0, 1], "tensor_split": [3, 2]}, "embed": {"target": "cpu"}}'
    with pytest.raises(TargetRefusedError, match=r"mistral\.rs cannot split m\.gguf") as caught:
        _saved_chat_placement(tmp_path, monkeypatch, raw)
    assert list(caught.value.reasons) == [WorkerRole.CHAT]


def test_a_manual_placement_does_not_hide_other_estimator_failures() -> None:
    from lilbee.providers.fleet.placement import placement_from_spec
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    def unsizable(_role: WorkerRole, _ratio: tuple[int, ...]) -> tuple[int, ...]:
        raise ProviderError("the model cannot be sized")

    with pytest.raises(ProviderError, match="cannot be sized"):
        placement_from_spec(
            PlacementSpec.from_json('{"chat": {"devices": [0]}}'),
            (WorkerRole.CHAT,),
            {0: 24 * GIB},
            estimate_peak=unsizable,
        )


# ---- a cached GGUF across several cards: the checks read its snapshot file ----

_CACHED_NAME = "Qwen3-4B-Q4_K_M.gguf"


def _cached_gguf(tmp_path: Path, size: int = 3200) -> tuple[Path, Path]:
    """A cached single-file GGUF as the registry hands it out: its blob and its snapshot file."""
    repo = tmp_path / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / "blobs" / "9375c1fd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * size)
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    entry = snapshot / _CACHED_NAME
    entry.symlink_to(blob)
    return blob, entry


def test_a_cpu_build_refusal_of_a_cached_gguf_names_the_snapshot_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: caps("Linux", "x86_64", gpus=[]))
    blob, _entry = _cached_gguf(tmp_path)
    with pytest.raises(
        ProviderError, match=rf"mistral\.rs cannot split {_CACHED_NAME} across 2 cards on the CPU"
    ):
        MISTRAL.check_split(blob, (1, 1))


@pytest.mark.parametrize("shard", [None, "9b"])
@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED])
def test_a_one_card_launch_of_a_cache_blob_with_no_snapshot_file_is_refused_with_the_repair(
    role: WorkerRole, shard: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host_with_cards(monkeypatch)
    blobs = tmp_path / "models--Qwen--Qwen3-4B-GGUF" / "blobs"
    blob = (blobs / shard if shard else blobs) / "9375c1fd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 3200)
    with pytest.raises(
        ProviderError,
        match=r"cannot launch this model: its model file is incomplete or damaged\. "
        r"Run 'lilbee model list', remove that model with 'lilbee model rm', "
        r"then pull it again\.",
    ) as refused:
        _one_card_argv(tmp_path, role, model_path=blob)
    assert blob.name not in str(refused.value)
