"""The OpenVINO Model Server adapter: gate, launch argv, routes and the rerank shape."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps, planning
from lilbee.providers.fleet.engines import choice, ovms, proof
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RerankShape,
    UserInstalled,
)
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.roles import RoleTarget, WorkerRole

BINARY = Path("/engines/bin/ovms")
MODEL = Path("/models/bge-base-en-v1.5-int8-ov")
IR_REPO = "OpenVINO/bge-base-en-v1.5-int8-ov"
INSTALL_DOCS = (
    "https://docs.openvino.ai/2026/model-server/ovms_docs_deploying_server_baremetal.html"
)
"""The upstream install page; its Ubuntu 24 steps unpack the 2026.4.1 archive and put
``ovms/bin`` on PATH."""
INTEL = 0x8086
NVIDIA = 0x10DE


def _host(
    system: str = "Linux", machine: str = "x86_64", display: tuple[int, ...] = ()
) -> HostCaps:
    return host_caps.collect(
        system=system, machine=machine, cpu_info={}, nvidia=None, display_vendors=display
    )


def _request(target: RoleTarget = RoleTarget.CPU) -> LaunchRequest:
    return LaunchRequest(
        binary=BINARY,
        model_path=MODEL,
        devices=(),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=2048,
        target=target,
    )


def _argv(role: WorkerRole, target: RoleTarget = RoleTarget.CPU) -> list[str]:
    return OVMS.launch_argv(OVMS.role_spec(role, None, None), _request(target))


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.OVMS] is OVMS


def test_the_engine_loads_openvino_ir_and_serves_four_roles() -> None:
    assert OVMS.formats == {ModelFormat.OPENVINO}
    assert OVMS.served_roles == {
        WorkerRole.CHAT,
        WorkerRole.EMBED,
        WorkerRole.RERANK,
        WorkerRole.VISION,
    }


def test_the_http_port_is_rest_port_and_the_grpc_port_is_not_set() -> None:
    argv = _argv(WorkerRole.CHAT)
    assert argv[argv.index("--rest_port") + 1] == PORT_SLOT
    assert "--port" not in argv


def test_each_role_names_its_task() -> None:
    tasks = {
        role: (lambda a: a[a.index("--task") + 1])(_argv(role))
        for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION)
    }
    assert tasks == {
        WorkerRole.CHAT: "text_generation",
        WorkerRole.EMBED: "embeddings",
        WorkerRole.RERANK: "rerank",
        WorkerRole.VISION: "text_generation",
    }


def test_the_vision_role_launches_as_a_text_generation_server() -> None:
    spec = OVMS.role_spec(WorkerRole.VISION, None, None)
    assert spec.extra_args == ("--task", "text_generation")


def test_the_launch_binds_loopback_and_serves_the_one_model_name() -> None:
    argv = _argv(WorkerRole.EMBED)
    assert argv[0] == str(BINARY)
    assert argv[argv.index("--model_path") + 1] == str(MODEL)
    assert argv[argv.index("--model_name") + 1] == "default_model"
    assert argv[argv.index("--rest_bind_address") + 1] == "127.0.0.1"


def test_the_target_sets_the_device() -> None:
    cpu = _argv(WorkerRole.CHAT, RoleTarget.CPU)
    gpu = _argv(WorkerRole.CHAT, RoleTarget.GPU)
    assert cpu[cpu.index("--target_device") + 1] == "CPU"
    assert gpu[gpu.index("--target_device") + 1] == "GPU"


def test_a_role_the_engine_does_not_serve_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OVMS, "served_roles", frozenset({WorkerRole.CHAT}))
    with pytest.raises(ProviderError):
        OVMS.role_spec(WorkerRole.VISION, None, None)


def test_the_gate_offers_linux_x86_64() -> None:
    gate = OVMS.host_gate(_host())
    assert (gate.ok, gate.known) == (True, True)


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "arm64"), ("Windows", "AMD64")])
def test_the_gate_refuses_another_platform_in_user_words(system: str, machine: str) -> None:
    gate = OVMS.host_gate(_host(system, machine))
    assert not gate.ok
    assert gate.reason.startswith("OpenVINO Model Server is not offered on ")


def test_an_unread_platform_is_unknown() -> None:
    gate = OVMS.host_gate(
        host_caps.collect(system="Plan9", machine="mips", cpu_info={}, nvidia=None)
    )
    assert gate.known is False


def test_the_targets_follow_the_graphics_cards_of_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host(display=(INTEL,)))
    assert OVMS.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}
    monkeypatch.setattr(runnable, "session_host", lambda: _host(display=(NVIDIA,)))
    assert OVMS.device_targets(BINARY) == {RoleTarget.CPU}


def test_the_swap_entry_checks_the_ready_route_and_names_the_model() -> None:
    extras = OVMS.swap_extras(None)  # type: ignore[arg-type]
    assert extras == {"checkEndpoint": "/v2/health/ready", "useModelName": "default_model"}


def test_the_rerank_profile_uses_the_cohere_shape_under_v3() -> None:
    profile = OVMS.request_profile(WorkerRole.RERANK)
    assert (profile.rerank_path, profile.rerank_shape) == ("/v3/rerank", RerankShape.DOCUMENTS)
    assert profile.rerank_model == "default_model"
    assert profile.batch_limit == 100


def test_the_other_roles_have_no_rerank_route() -> None:
    assert OVMS.request_profile(WorkerRole.EMBED).rerank_path is None
    assert OVMS.request_profile(WorkerRole.CHAT).rerank_path is None


def test_the_pin_record_has_no_pack_and_names_no_licence() -> None:
    """lilbee ships no OVMS pack, so the record carries no digest, no platform and no licence."""
    record = OVMS.pin_record()
    assert (record["sha256"], record["platforms"], record["licenses"]) == (None, {}, [])


def test_the_engine_is_one_the_user_installs() -> None:
    assert isinstance(OVMS, UserInstalled)
    assert OVMS.install_docs_url == INSTALL_DOCS
    assert INSTALL_DOCS in OVMS.install_hint


def test_an_ovms_on_path_is_the_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(BINARY))
    assert OVMS.binary() == BINARY
    assert OVMS.identity(BINARY) == f"ovms 2026.4.1 {BINARY}"


def test_the_lookup_asks_for_ovms(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(ovms.shutil, "which", lambda name: asked.append(name))
    assert OVMS.user_binary() is None
    assert asked == ["ovms"]


def test_a_missing_install_says_where_to_get_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ovms.shutil, "which", lambda name: None)
    with pytest.raises(ProviderError) as exc:
        OVMS.binary()
    text = str(exc.value)
    assert text.startswith("The OpenVINO Model Server engine is not installed. ")
    assert INSTALL_DOCS in text
    assert "PATH" in text
    assert "LD_LIBRARY_PATH" in text
    assert "lilbee engine install" not in text


def test_an_ovms_on_path_counts_as_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(BINARY))
    assert EngineName.OVMS in choice.installed_engines()
    monkeypatch.setattr(ovms.shutil, "which", lambda name: None)
    assert EngineName.OVMS not in choice.installed_engines()


def test_install_refuses_with_the_user_install_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: _host())
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("ovms")
    text = str(exc.value)
    assert "No pack of ovms is published yet." in text
    assert INSTALL_DOCS in text


def test_the_engine_row_says_install_it_yourself_with_its_docs_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host())
    monkeypatch.setattr(ovms.shutil, "which", lambda name: None)
    rows = {o.name: o for o in engine_service.engine_options(f"{IR_REPO}#openvino")}
    row = rows["ovms"]
    assert (row.installed, row.route, row.docs_url) == (
        False,
        engine_service.InstallRoute.BY_HAND,
        INSTALL_DOCS,
    )


def _archive(root: Path, *, lib: bool = True) -> Path:
    """An unpacked release archive: ``ovms/bin/ovms`` and, when asked, ``ovms/lib``."""
    (root / "ovms" / "bin").mkdir(parents=True)
    if lib:
        (root / "ovms" / "lib").mkdir()
    binary = root / "ovms" / "bin" / "ovms"
    binary.write_text("", encoding="utf-8")
    return binary


def test_the_launch_env_puts_the_archive_lib_folder_on_the_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _archive(tmp_path)
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(binary))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert OVMS.runtime_env() == {"LD_LIBRARY_PATH": str(tmp_path / "ovms" / "lib")}


def test_the_launch_env_keeps_a_library_path_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _archive(tmp_path)
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(binary))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/other")
    expected = os.pathsep.join([str(tmp_path / "ovms" / "lib"), "/opt/other"])
    assert OVMS.runtime_env() == {"LD_LIBRARY_PATH": expected}


def test_the_launch_env_follows_a_link_to_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _archive(tmp_path / "unpacked")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    link = link_dir / "ovms"
    link.symlink_to(binary)
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(link))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert OVMS.runtime_env() == {"LD_LIBRARY_PATH": str(tmp_path / "unpacked" / "ovms" / "lib")}


def test_the_launch_env_is_empty_when_the_binary_has_no_lib_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _archive(tmp_path, lib=False)
    monkeypatch.setattr(ovms.shutil, "which", lambda name: str(binary))
    assert OVMS.runtime_env() == {}


def test_the_launch_env_is_empty_when_ovms_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ovms.shutil, "which", lambda name: None)
    assert OVMS.runtime_env() == {}


def test_the_engine_has_no_optional_features_to_probe() -> None:
    assert OVMS.supports_architecture("anything") is True
    probe = OVMS.discover_devices(BINARY)
    assert probe.devices == []
    assert not any(
        (
            OVMS.check_readback(BINARY),
            OVMS.supports_fit_target(BINARY),
            OVMS.supports_shared_kv(BINARY),
        )
    )
    assert OVMS.model_meta(MODEL) is None
    assert OVMS.assert_loadable("repo", "file", None) is None  # type: ignore[arg-type]
    assert OVMS.target_env(RoleTarget.CPU, 1) == {}


def test_the_memory_estimate_charges_the_weights_of_the_ir_folder(tmp_path: Path) -> None:
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "openvino_model.bin").write_bytes(b"\0" * 4096)
    request = MemoryRequest(
        model_path=tmp_path,
        ctx=2048,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    estimate = OVMS.estimate_memory(request)
    assert estimate.device_bytes >= 4096


def test_the_proof_table_lists_every_device_as_not_run() -> None:
    devices = ("Intel CPU", "Intel integrated graphics", "Intel Arc graphics", "Intel NPU")
    assert proof.experimental_devices("ovms") == devices
    assert not proof.is_proven("ovms")


def test_rerank_bodies_and_scores_follow_the_shape() -> None:
    profile = OVMS.request_profile(WorkerRole.RERANK)
    assert profile.rerank_body("q", ["a", "b"]) == {
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
        "model": "default_model",
    }
    payload = {
        "results": [{"index": 1, "relevance_score": 0.7}, {"index": 0, "relevance_score": 0.2}]
    }
    assert profile.rerank_scores(payload) == {0: 0.2, 1: 0.7}


def test_an_installed_ir_folder_is_placeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "openvino_model.xml").write_bytes(b"x" * 100)
    (tmp_path / "openvino_model.bin").write_bytes(b"y" * 900)
    monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda _ref: tmp_path)
    assert planning._role_weights_bytes(WorkerRole.CHAT, "org/ir#openvino") == 1000
    assert planning.role_model_placeable(WorkerRole.CHAT, "org/ir#openvino", 0, engine=OVMS)


_INTERNVL2_FILES = (
    "openvino_language_model",
    "openvino_vision_embeddings_model",
    "openvino_text_embeddings_model",
    "openvino_tokenizer",
    "openvino_detokenizer",
)


def _vision_folder(root: Path) -> Path:
    """A folder with the file names of OpenVINO/InternVL2-1B-int4-ov: no ``openvino_model`` pair."""
    for stem in _INTERNVL2_FILES:
        (root / f"{stem}.xml").write_bytes(b"x" * 10)
        (root / f"{stem}.bin").write_bytes(b"y" * 100)
    return root


def test_an_installed_vision_ir_folder_is_placeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda _ref: tmp_path)
    _vision_folder(tmp_path)
    ref = "org/vision#openvino"
    assert planning._role_weights_bytes(WorkerRole.VISION, ref) == 3 * 110
    assert planning.role_model_placeable(WorkerRole.VISION, ref, 0, engine=OVMS)
