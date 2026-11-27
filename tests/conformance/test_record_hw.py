"""The GPU recording driver starts each engine on the GPU, checks the device and refuses a CPU run.

The engine is a script that listens on the port in its command line. Everything else is real:
the adapters build the command line, the driver starts the process, and the recorder's own
device check reads the reply and the log.
"""

from __future__ import annotations

import json
import socket
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.engines.llama import pack_backend
from lilbee.providers.roles import WorkerRole
from tests.conformance import record_hw
from tests.conformance.record_cpu_compare import Model

_ENGINE = """#!{python}
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
if "--help" in argv:
    print("usage: engine [options]")
    flags = ("--memory", "--alias", "--no-context-shift", "--flash-attn", "--mmproj", "--pooling")
    for flag in flags:
        print(f"  {{flag}} VALUE  an option line")
    sys.exit(0)
mode = os.environ["FAKE_HW_DEVICE"]
if mode == "die":
    sys.exit(3)
port = int(argv[argv.index("--port") + 1])
if mode in ("cuda", "log-only"):
    print("Starting FlashBert model on Cuda(CudaDevice(DeviceId(0)))", flush=True)
print("env", os.environ.get("HW_RUNTIME"), os.environ.get("HW_TARGET"), flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        rows = {{"cuda": "CUDA0", "cpu": "CPU", "vulkan": "Vulkan0"}}
        if self.path == "/health":
            body = b"ok"
        elif self.path == "/memory" and mode in rows:
            body = json.dumps({{"data": [{{"name": rows[mode], "model": 1}}]}}).encode()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""
_RESULTS_PAGE = """# Engine results

| Recording | Roles | Machine | Result | Version | Fixture |
|---|---|---|---|---|---|
| llama.cpp-cpu | chat, embed | CPU host | pass, 2 of 2 steps | 1.0 | `llama.cpp-cpu.json` |
| tei-hw | embed, rerank | NVIDIA GPU | not recorded |  |  |
| lemonade | chat | AMD NPU | not recorded, no machine |  |  |

**Experimental.** Text after the table.
"""
_CELL = "linux-x86_64-cuda-sm89"
_BACKEND = "cuda-sm89"
_L4 = {
    "driver": "570.0",
    "cuda_driver": 12080,
    "gpus": [{"name": "NVIDIA L4", "cc": [8, 9]}],
}


def _host(nvidia: dict[str, Any] | None) -> host_caps.HostCaps:
    return host_caps.collect(
        system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=nvidia
    )


def _free_base_port() -> int:
    """A port with the three above it free, for a run of up to four servers."""
    for base in range(18500, 19500, 7):
        socks: list[socket.socket] = []
        try:
            for port in range(base, base + 4):
                sock = socket.socket()
                socks.append(sock)
                sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            for sock in socks:
                sock.close()
        return base
    raise AssertionError("no free ports")


def _answers(url: str) -> bool:
    try:
        urllib.request.urlopen(url + "/health", timeout=1)  # noqa: S310  a local http URL
    except (urllib.error.URLError, OSError):
        return False
    return True


class _Run:
    """One call of the driver with the install, the model fetch and the recorder replaced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.commands: list[list[str]] = []
        self.recorder_code = 0
        self.installed_backend = _BACKEND
        self.results = tmp_path / "engine-results.md"
        self.results.write_text(_RESULTS_PAGE, encoding="utf-8")
        binary = tmp_path / "engine"
        binary.write_text(_ENGINE.format(python=sys.executable), encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        self.binary = binary
        monkeypatch.setattr(runnable, "session_host", lambda: _host(_L4))
        monkeypatch.setattr(engine_service, "install", self._install)
        for adapter in ADAPTERS.values():
            monkeypatch.setattr(adapter, "binary", lambda: binary, raising=False)
        monkeypatch.setattr(record_hw, "fetch_model", self._fetch)
        monkeypatch.setattr(record_hw.subprocess, "call", self._record)

    def _install(self, name: str) -> engine_service.InstalledBuild:
        return engine_service.InstalledBuild(name, "9.9", self.installed_backend)

    def _fetch(self, model: Model, root: Path) -> Path:
        path = self.tmp / "models" / model.folder / model.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def _record(self, command: list[str]) -> int:
        """Stand in for the recorder: it writes the recording and its log, as the real one does."""
        self.commands.append(command)
        out = Path(_flag(command, "--out"))
        out.parent.mkdir(parents=True, exist_ok=True)
        name = _flag(command, "--server")
        steps = ["embed", "rerank", "named"]
        doc = {
            "server": name,
            "version": _flag(command, "--version"),
            "roles": _flag(command, "--roles").split(","),
            "results": {step: {"ok": True, "detail": ""} for step in steps},
        }
        out.write_text(json.dumps(doc), encoding="utf-8")
        lines = [f"engine: {name}", "backend: cuda", "models: x"]
        lines += [f"STEP {name} {step} ok=True" for step in steps]
        out.with_suffix(".log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.recorder_code

    def main(self, engine: str, *extra: str, backend: str = "cuda") -> int:
        return record_hw.main(
            [
                "--engine",
                engine,
                "--cell",
                _CELL,
                "--backend",
                backend,
                "--name",
                f"{engine}-hw",
                "--models",
                str(self.tmp / "models"),
                "--logs",
                str(self.tmp / "logs"),
                "--out-dir",
                str(self.tmp / "out"),
                "--results-doc",
                str(self.results),
                "--base-port",
                str(_free_base_port()),
                *extra,
            ]
        )


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Run:
    monkeypatch.setenv("FAKE_HW_DEVICE", "cuda")
    return _Run(monkeypatch, tmp_path)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class TestLaunch:
    def _argv(self, run: _Run, engine: EngineName, role: WorkerRole) -> list[str]:
        model = record_hw._MODELS[engine][role]
        part = record_hw.Part(role, model, run.tmp / model.folder / model.file, "served")
        return record_hw.launch_argv(ADAPTERS[engine], part, run.binary, None, 8123)

    def test_mistral_runs_on_the_gpu(self, run: _Run) -> None:
        argv = self._argv(run, EngineName.MISTRAL, WorkerRole.CHAT)
        assert argv[argv.index("--port") + 1] == "8123"
        assert "--cpu" not in argv
        assert argv[argv.index("--paged-attn") + 1] == "off"

    def test_llama_offloads_every_layer_and_names_no_device(self, run: _Run) -> None:
        argv = self._argv(run, EngineName.LLAMA, WorkerRole.CHAT)
        assert int(_flag(argv, "--n-gpu-layers")) != 0
        assert "--device" not in argv
        assert _flag(argv, "--alias") == "served"
        assert _flag(argv, "--port") == "8123"

    def test_llama_asks_for_the_memory_route_when_the_binary_has_it(self, run: _Run) -> None:
        assert "--memory" in self._argv(run, EngineName.LLAMA, WorkerRole.CHAT)

    def test_llama_vision_carries_the_projector(self, run: _Run) -> None:
        model = record_hw._MODELS[EngineName.LLAMA][WorkerRole.VISION]
        part = record_hw.Part(WorkerRole.VISION, model, run.tmp / "v.gguf", "served")
        projector = run.tmp / "mmproj.gguf"
        argv = record_hw.launch_argv(ADAPTERS[EngineName.LLAMA], part, run.binary, projector, 8123)
        assert _flag(argv, "--mmproj") == str(projector)

    def test_tei_takes_the_port_at_the_end(self, run: _Run) -> None:
        argv = self._argv(run, EngineName.TEI, WorkerRole.EMBED)
        assert argv[-2:] == ["--port", "8123"]

    def test_tei_keeps_the_cards_visible(self) -> None:
        env = ADAPTERS[EngineName.TEI].target_env(record_hw.RoleTarget.GPU, 1)
        assert "CUDA_VISIBLE_DEVICES" not in env


class TestRun:
    def test_a_cuda_run_records_each_role_and_stops_the_servers(
        self, run: _Run, tmp_path: Path
    ) -> None:
        assert run.main("mistral.rs") == 0
        (command,) = run.commands
        assert command[:3] == [sys.executable, "-m", "tests.conformance.record_presets"]
        assert "--strict" in command
        assert _flag(command, "--engine") == "mistral.rs"
        assert _flag(command, "--backend") == "cuda"
        assert _flag(command, "--roles") == "chat,embed,vision"
        assert _flag(command, "--models") == "chat=default,embed=default,vision=default"
        assert _flag(command, "--version") == "mistral.rs 9.9 " + _CELL
        assert _flag(command, "--server") == "mistral.rs-hw"
        assert "--paged-attn off" in _flag(command, "--server-flags")
        assert _flag(command, "--out") == str(tmp_path / "out" / "mistral.rs-hw.json")
        assert Path(_flag(command, "--engine-log")).name.startswith("mistral.rs-chat-")
        assert "--embed-url" in command
        assert "--vision-url" in command
        assert "--rerank-url" not in command
        assert not _answers(_flag(command, "--url"))
        assert not _answers(_flag(command, "--embed-url"))

    def test_tei_puts_the_embedder_first_and_the_reranker_beside_it(self, run: _Run) -> None:
        assert run.main("tei") == 0
        (command,) = run.commands
        assert _flag(command, "--roles") == "embed,rerank"
        assert _flag(command, "--models") == "embed=st_embed,rerank=bge-reranker-v2-m3"
        assert "--embed-url" not in command
        assert _flag(command, "--url") != _flag(command, "--rerank-url")

    @pytest.mark.parametrize(
        ("engine", "roles", "models"),
        [
            (
                "llama.cpp",
                "chat,embed,rerank,vision",
                "chat=Qwen3-1.7B-Q8_0,embed=Qwen3-Embedding-0.6B-Q8_0,"
                "rerank=bge-reranker-v2-m3-Q8_0,vision=SmolVLM-256M-Instruct-Q8_0",
            ),
            (
                "ik_llama.cpp",
                "chat,embed",
                "chat=Qwen3-1.7B-Q8_0,embed=Qwen3-Embedding-0.6B-Q8_0",
            ),
        ],
    )
    def test_the_llama_family_names_each_server_by_the_stem_of_its_file(
        self, run: _Run, engine: str, roles: str, models: str
    ) -> None:
        assert run.main(engine) == 0
        (command,) = run.commands
        assert _flag(command, "--roles") == roles
        assert _flag(command, "--models") == models

    def test_the_log_alone_proves_the_device_of_an_engine_with_no_memory_route(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_HW_DEVICE", "log-only")
        assert run.main("tei") == 0
        assert len(run.commands) == 1

    def test_the_engine_starts_with_the_environment_of_its_adapter_on_the_gpu(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tei = ADAPTERS[EngineName.TEI]
        monkeypatch.setattr(tei, "runtime_env", lambda: {"HW_RUNTIME": "libs"})
        monkeypatch.setattr(tei, "target_env", lambda target, cards: {"HW_TARGET": target.value})
        assert run.main("tei") == 0
        (log,) = (run.tmp / "logs").glob("tei-embed-*.log")
        assert "env libs gpu" in log.read_text(encoding="utf-8")

    def test_the_return_code_is_the_recorders(self, run: _Run) -> None:
        run.recorder_code = 1
        assert run.main("tei") == 1

    def test_unknown_arguments_go_to_the_recorder(self, run: _Run) -> None:
        run.main("tei", "--window", "4096")
        assert run.commands[0][-2:] == ["--window", "4096"]

    @pytest.mark.parametrize(
        ("mode", "refusal"),
        [
            ("none", "--backend cuda: the engine lists no device"),
            ("cpu", "--backend cuda: the engine reports cpu"),
            ("vulkan", "--backend cuda: the engine reports vulkan"),
        ],
    )
    def test_a_run_without_a_cuda_device_records_nothing(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch, mode: str, refusal: str
    ) -> None:
        monkeypatch.setenv("FAKE_HW_DEVICE", mode)
        with pytest.raises(SystemExit) as refused:
            run.main("mistral.rs")
        assert refusal in str(refused.value)
        assert run.commands == []
        assert not (run.tmp / "out").exists()

    def test_an_engine_that_exits_is_named(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_HW_DEVICE", "die")
        with pytest.raises(SystemExit, match="exited with 3"):
            run.main("tei")
        assert run.commands == []

    def test_a_pack_of_another_cell_stops_the_run_before_any_server_starts(self, run: _Run) -> None:
        run.installed_backend = "cuda-sm80"
        with pytest.raises(SystemExit, match="linux-x86_64-cuda-sm80, not " + _CELL):
            run.main("tei")
        assert list((run.tmp / "logs").glob("*.log")) == []

    def test_vulkan_host_flag_is_for_llama_on_vulkan(self, run: _Run) -> None:
        with pytest.raises(SystemExit, match=r"llama\.cpp only"):
            run.main("tei", "--as-vulkan-host")
        with pytest.raises(SystemExit, match="vulkan backend"):
            run.main("llama.cpp", "--as-vulkan-host")


class TestRecorderArguments:
    @pytest.mark.parametrize("engine", ["mistral.rs", "tei", "llama.cpp", "ik_llama.cpp"])
    def test_the_embed_size_is_the_size_of_the_embed_model(self, run: _Run, engine: str) -> None:
        assert run.main(engine) == 0
        assert _flag(run.commands[0], "--embed-dim") == "1024"

    def test_every_embed_model_has_a_size(self) -> None:
        for models in record_hw._MODELS.values():
            assert models[WorkerRole.EMBED].repo in record_hw._EMBED_DIMS

    def test_tei_asks_for_the_tei_rerank_wire(self, run: _Run) -> None:
        assert run.main("tei") == 0
        assert _flag(run.commands[0], "--wire") == "tei"

    @pytest.mark.parametrize("engine", ["mistral.rs", "llama.cpp", "ik_llama.cpp"])
    def test_the_other_engines_keep_the_default_wire(self, run: _Run, engine: str) -> None:
        assert run.main(engine) == 0
        assert "--wire" not in run.commands[0]


class TestResultsRow:
    def test_a_good_run_writes_its_row_and_names_the_card_in_the_log(self, run: _Run) -> None:
        assert run.main("mistral.rs") == 0
        row = (
            "| mistral.rs-hw | chat, embed, vision | NVIDIA L4 | pass, 3 of 3 steps "
            f"| mistral.rs 9.9 {_CELL} | `mistral.rs-hw.json` |"
        )
        lines = run.results.read_text(encoding="utf-8").split("\n")
        assert lines.count(row) == 1
        assert lines[lines.index(row) - 1].startswith("| lemonade |")
        log = (run.tmp / "out" / "mistral.rs-hw.log").read_text(encoding="utf-8").split("\n")
        assert log.index("gpu: NVIDIA L4") < log.index("STEP mistral.rs-hw embed ok=True")

    def test_the_row_of_a_name_in_the_table_is_replaced(self, run: _Run) -> None:
        assert run.main("tei") == 0
        text = run.results.read_text(encoding="utf-8")
        assert text.count("| tei-hw |") == 1
        assert "| tei-hw | embed, rerank | NVIDIA L4 | pass, 3 of 3 steps |" in text
        assert "| `tei-hw.json` |\n| lemonade" in text

    def test_the_table_ends_where_it_did_and_the_text_after_it_stays(self, run: _Run) -> None:
        assert run.main("mistral.rs") == 0
        assert run.results.read_text(encoding="utf-8").endswith(
            f"| mistral.rs 9.9 {_CELL} | `mistral.rs-hw.json` |\n\n"
            "**Experimental.** Text after the table.\n"
        )

    def test_a_run_that_fails_a_step_writes_no_row(self, run: _Run) -> None:
        run.recorder_code = 1
        before = run.results.read_text(encoding="utf-8")
        assert run.main("mistral.rs") == 1
        assert run.results.read_text(encoding="utf-8") == before
        assert "gpu:" not in (run.tmp / "out" / "mistral.rs-hw.log").read_text(encoding="utf-8")

    def test_a_host_with_no_cuda_card_needs_the_card_named(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runnable, "session_host", lambda: _host(None))
        with pytest.raises(SystemExit, match="--gpu"):
            run.main("tei")
        assert not (run.tmp / "logs").exists()
        monkeypatch.setattr(runnable, "session_host", lambda: _host(_L4))
        assert run.main("tei", "--gpu", "AMD Instinct MI300X") == 0
        assert "| tei-hw | embed, rerank | AMD Instinct MI300X |" in run.results.read_text(
            encoding="utf-8"
        )

    def test_the_card_of_a_vulkan_run_is_the_card_of_the_live_host(
        self, run: _Run, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_HW_DEVICE", "vulkan")
        run.installed_backend = "vulkan"
        code = run.main(
            "llama.cpp", "--as-vulkan-host", "--cell", "linux-x86_64-vulkan", backend="vulkan"
        )
        assert code == 0
        assert "| llama.cpp-hw | chat, embed, rerank, vision | NVIDIA L4 |" in (
            run.results.read_text(encoding="utf-8")
        )

    def test_a_missing_results_page_stops_the_run_before_the_install(self, run: _Run) -> None:
        with pytest.raises(SystemExit, match="results page"):
            run.main("tei", "--results-doc", str(run.tmp / "none.md"))
        assert not (run.tmp / "logs").exists()
        assert run.commands == []


def test_the_documented_setup_installs_the_providers_extra() -> None:
    page = Path(record_hw.__file__).parents[2] / "docs" / "engine-integration.md"
    assert "uv sync --python 3.13 --extra providers" in page.read_text(encoding="utf-8")


class TestVulkanHost:
    def test_a_cuda_card_gets_the_cuda_pack_unless_presented_as_vulkan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = _host(_L4)
        monkeypatch.setattr(runnable, "session_host", lambda: live)
        assert pack_backend(live) == "cuda"
        assert pack_backend(record_hw.vulkan_host()) == "vulkan"

    def test_the_session_host_is_the_vulkan_host_inside_the_block_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = _host(_L4)
        monkeypatch.setattr(runnable, "detect", lambda: live)
        runnable.refresh_session_host()
        with record_hw.presented_as(record_hw.vulkan_host()):
            assert runnable.session_host().gpus == ()
        assert runnable.session_host().gpus == live.gpus
        with record_hw.presented_as(None):
            assert runnable.session_host().gpus == live.gpus
