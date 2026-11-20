"""The recorder writes the backend the engine reports, and refuses a run whose report differs."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import pytest

from tests.conformance import record_presets


@contextmanager
def _engine(payload: object | None, seen: list[str] | None = None) -> Iterator[str]:
    """A local engine whose /memory answers *payload*, or 404 when *payload* is None."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if seen is not None:
                seen.append(self.headers.get("Authorization", ""))
            if self.path != "/memory" or payload is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _rows(*names: str) -> dict:
    return {"data": [{"name": name, "model": 1} for name in names]}


class TestReportedBackend:
    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            (["CUDA0", "CUDA_Host"], "cuda"),
            (["CUDA0", "CPU"], "cuda"),
            (["MTL0"], "metal"),
            (["Vulkan1"], "vulkan"),
            (["CPU", "CPU_Mapped"], "cpu"),
            (["CUDA_Host"], "cpu"),
            (["Host"], "cpu"),
            (["CUDA0", "Vulkan0"], "cuda+vulkan"),
            (["Wibble0"], "wibble"),
            (["7"], "7"),
            ([], "none"),
        ],
    )
    def test_the_labels_name_one_backend(self, devices: list[str], expected: str) -> None:
        assert record_presets.reported_backend(devices) == expected

    def test_a_cpu_run_with_no_report_is_accepted(self) -> None:
        record_presets.check_backend("cpu", "none")

    def test_a_matching_report_is_accepted(self) -> None:
        record_presets.check_backend("cuda", "cuda")

    def test_a_gpu_label_with_no_report_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="--backend cuda: the engine lists no device"):
            record_presets.check_backend("cuda", "none")

    def test_a_different_report_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="--backend cuda: the engine reports cpu"):
            record_presets.check_backend("cuda", "cpu")


class TestReportedDevices:
    def test_a_row_with_no_bytes_is_a_listed_device_not_a_used_one(self) -> None:
        payload = {"data": [{"name": "CUDA0", "model": 0}, {"name": "CPU", "model": 5}]}
        with _engine(payload) as url:
            assert record_presets.reported_devices(url, None) == ["CPU"]

    def test_the_labels_come_from_the_engine_reply(self) -> None:
        with _engine(_rows("CUDA0", "CPU")) as url:
            assert record_presets.reported_devices(url, None) == ["CPU", "CUDA0"]

    def test_an_engine_without_the_route_lists_nothing(self) -> None:
        with _engine(None) as url:
            assert record_presets.reported_devices(url, None) == []

    def test_an_engine_that_does_not_answer_lists_nothing(self) -> None:
        with _engine(None) as url:
            pass
        assert record_presets.reported_devices(url, None) == []

    def test_a_reply_that_is_not_json_lists_nothing(self) -> None:
        with _engine("not rows") as url:
            assert record_presets.reported_devices(url, None) == []

    def test_the_token_is_sent_when_given(self) -> None:
        seen: list[str] = []
        with _engine(_rows("CUDA0"), seen) as url:
            record_presets.reported_devices(url, "tok")
        assert seen == ["Bearer tok"]


def _record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str, backend: str, *extra: str
) -> tuple[int, Path]:
    """Run the recorder against *url* with the steps stubbed out; return its rc and output."""
    out = tmp_path / "rec.json"
    argv = [
        "record_presets",
        "--server",
        "eng",
        "--version",
        "1",
        "--url",
        url,
        "--roles",
        "chat",
        "--models",
        "chat=m",
        "--backend",
        backend,
        "--out",
        str(out),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(record_presets, "run_steps", lambda setup: {})

    @contextmanager
    def no_config(*args: object) -> Iterator[None]:
        yield

    monkeypatch.setattr(record_presets, "configured", no_config)
    return record_presets.main(), out


class TestRecorderMain:
    def test_a_gpu_label_the_engine_does_not_confirm_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(None) as url, pytest.raises(SystemExit, match="lists no device"):
            _record(monkeypatch, tmp_path, url, "cuda")
        assert not (tmp_path / "rec.json").exists()
        assert not (tmp_path / "rec.log").exists()

    def test_a_cpu_engine_cannot_carry_a_cuda_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(_rows("CPU")) as url, pytest.raises(SystemExit, match="reports cpu"):
            _record(monkeypatch, tmp_path, url, "cuda")
        assert not (tmp_path / "rec.log").exists()

    def test_a_gpu_engine_cannot_carry_a_cpu_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(_rows("CUDA0")) as url, pytest.raises(SystemExit, match="reports cuda"):
            _record(monkeypatch, tmp_path, url, "cpu")

    def test_a_gpu_label_with_only_a_pinned_host_row_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(_rows("CUDA_Host")) as url, pytest.raises(SystemExit, match="reports cpu"):
            _record(monkeypatch, tmp_path, url, "cuda")
        assert not (tmp_path / "rec.log").exists()

    def test_a_gpu_label_with_a_device_that_holds_no_bytes_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        payload = {"data": [{"name": "CUDA0", "model": 0}]}
        with _engine(payload) as url, pytest.raises(SystemExit, match="lists no device"):
            _record(monkeypatch, tmp_path, url, "cuda")
        assert not (tmp_path / "rec.log").exists()

    def test_a_confirmed_label_is_written_with_the_reported_devices(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(_rows("CUDA0", "CPU")) as url:
            rc, out = _record(monkeypatch, tmp_path, url, "cuda")
        assert rc == 0
        lines = out.with_suffix(".log").read_text(encoding="utf-8").splitlines()
        assert "backend: cuda" in lines
        assert "reported-backend: cuda" in lines
        assert "devices: CPU,CUDA0" in lines

    def test_a_cpu_engine_with_no_report_is_written_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(None) as url:
            rc, out = _record(monkeypatch, tmp_path, url, "cpu")
        assert rc == 0
        lines = out.with_suffix(".log").read_text(encoding="utf-8").splitlines()
        assert "reported-backend: none" in lines
        assert "devices: none" in lines

    def test_the_device_url_names_the_server_that_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(None) as front, _engine(_rows("MTL0")) as engine:
            rc, out = _record(monkeypatch, tmp_path, front, "metal", "--device-url", engine)
        assert rc == 0
        assert "devices: MTL0" in out.with_suffix(".log").read_text(encoding="utf-8")

    def test_the_token_goes_only_to_the_server_it_was_given_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[str] = []
        monkeypatch.setenv("REC_TOKEN", "tok")
        with _engine(None, seen) as front, _engine(_rows("CUDA0"), seen) as engine:
            _record(
                monkeypatch,
                tmp_path,
                front,
                "cuda",
                "--token-env",
                "REC_TOKEN",
                "--device-url",
                engine,
            )
        assert seen == [""]


# Startup lines copied from the committed recording tests/conformance/fixtures/engines/tei-cuda.log
# (an RTX 3090 run of one text-embeddings-router process per model, 2026-10-31).
_TEI_CUDA_LOG = """\
  2026-10-31T13:23:13.743093Z  INFO text_embeddings_router: router/src/lib.rs:284: Starting model backend
  2026-10-31T13:23:14.261696Z  INFO text_embeddings_backend_candle: backends/candle/src/lib.rs:395: Starting FlashBert model on Cuda(CudaDevice(DeviceId(1)))
  2026-10-31T13:23:14.657541Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1878: Starting HTTP server: 0.0.0.0:5801
  2026-10-31T13:23:18.036296Z  INFO text_embeddings_backend_candle: backends/candle/src/lib.rs:424: Starting FlashBert model on Cuda(CudaDevice(DeviceId(1)))
"""  # noqa: E501
# The same recording without the line that names the device.
_TEI_NO_DEVICE_LOG = "".join(
    line for line in _TEI_CUDA_LOG.splitlines(keepends=True) if "model on" not in line
)
# Lines copied from tests/conformance/fixtures/engines/llama.cpp-cuda.log (RTX 3090 run).
_LLAMA_CUDA_LOG = """\
  [45325] 0.00.233.880 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 3090 (24126 MiB, 23861 MiB free)
  [45325] 0.00.776.313 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:a1:00.0) - 23861 MiB free
"""  # noqa: E501


_HELD: list[IO[str]] = []


@pytest.fixture(autouse=True)
def _close_held_logs() -> Iterator[None]:
    yield
    while _HELD:
        _HELD.pop().close()


def _log(tmp_path: Path, text: str, *, held: bool = True) -> str:
    """A log file; with *held* the test process keeps it open, as the engine keeps its output."""
    path = tmp_path / "engine.log"
    path.write_text(text, encoding="utf-8")
    if held:
        _HELD.append(path.open("a", encoding="utf-8"))
    return str(path)


class TestLoggedDevices:
    def test_the_startup_line_of_a_gpu_engine_names_the_device(self, tmp_path: Path) -> None:
        devices = record_presets.logged_devices(Path(_log(tmp_path, _TEI_CUDA_LOG)))
        assert devices == ["Cuda1"]
        assert record_presets.reported_backend(devices) == "cuda"

    def test_the_device_line_of_the_llama_engine_names_the_label(self, tmp_path: Path) -> None:
        devices = record_presets.logged_devices(Path(_log(tmp_path, _LLAMA_CUDA_LOG)))
        assert devices == ["CUDA0"]

    def test_a_log_without_a_device_line_names_no_device(self, tmp_path: Path) -> None:
        assert record_presets.logged_devices(Path(_log(tmp_path, _TEI_NO_DEVICE_LOG))) == []

    def test_a_log_that_cannot_be_read_stops_the_run(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match=r"--engine-log .*missing\.log"):
            record_presets.logged_devices(tmp_path / "missing.log")


class TestRecorderWithAnEngineLog:
    def test_a_gpu_label_is_accepted_on_the_log_of_an_engine_with_no_memory_route(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(None) as url:
            rc, out = _record(
                monkeypatch, tmp_path, url, "cuda", "--engine-log", _log(tmp_path, _TEI_CUDA_LOG)
            )
        assert rc == 0
        lines = out.with_suffix(".log").read_text(encoding="utf-8").splitlines()
        assert "devices: Cuda1" in lines
        assert "devices-from: engine-log" in lines
        assert "reported-backend: cuda" in lines

    def test_a_gpu_label_is_refused_when_the_log_names_no_device(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_NO_DEVICE_LOG)
        with _engine(None) as url, pytest.raises(SystemExit, match="lists no device"):
            _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", log)
        assert not (tmp_path / "rec.json").exists()
        assert not (tmp_path / "rec.log").exists()

    def test_a_cpu_label_is_refused_when_the_log_names_a_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_CUDA_LOG)
        with _engine(None) as url, pytest.raises(SystemExit, match="reports cuda"):
            _record(monkeypatch, tmp_path, url, "cpu", "--engine-log", log)
        assert not (tmp_path / "rec.log").exists()

    def test_a_log_that_disagrees_with_the_engine_reply_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_CUDA_LOG)
        with (
            _engine(_rows("CPU")) as url,
            pytest.raises(SystemExit, match="the log names cuda, the engine reply names cpu"),
        ):
            _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", log)
        assert not (tmp_path / "rec.json").exists()

    def test_the_engine_reply_gives_the_devices_when_the_log_agrees(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_CUDA_LOG)
        with _engine(_rows("CUDA0")) as url:
            rc, out = _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", log)
        assert rc == 0
        lines = out.with_suffix(".log").read_text(encoding="utf-8").splitlines()
        assert "devices: CUDA0" in lines
        assert "devices-from: memory" in lines

    def test_a_log_with_no_device_line_does_not_override_the_engine_reply(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_NO_DEVICE_LOG)
        with _engine(_rows("CUDA0")) as url:
            rc, out = _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", log)
        assert rc == 0
        assert "devices-from: memory" in out.with_suffix(".log").read_text(encoding="utf-8")

    def test_a_log_the_engine_does_not_hold_open_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A file the operator wrote proves nothing: the listening process must hold it open."""
        forged = _log(tmp_path, _TEI_CUDA_LOG, held=False)
        with _engine(None) as url, pytest.raises(SystemExit, match="does not hold this file open"):
            _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", forged)
        assert not (tmp_path / "rec.json").exists()
        assert not (tmp_path / "rec.log").exists()

    def test_a_log_held_by_another_process_than_the_listener_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_CUDA_LOG)
        monkeypatch.setattr(record_presets, "_holds_open", lambda pid, path: False)
        with _engine(None) as url, pytest.raises(SystemExit, match="does not hold this file open"):
            _record(monkeypatch, tmp_path, url, "cuda", "--engine-log", log)

    def test_a_log_for_an_engine_that_is_not_local_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, _TEI_CUDA_LOG)
        with pytest.raises(SystemExit, match="is not a local port"):
            record_presets.require_engine_output("http://gpu-box.example:5801", Path(log))

    def test_an_unreadable_process_table_refuses_the_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def denied(**kwargs: object) -> list[object]:
            raise record_presets.psutil.AccessDenied()

        monkeypatch.setattr(record_presets.psutil, "net_connections", denied)
        with pytest.raises(SystemExit, match="process table is not readable"):
            record_presets.require_engine_output("http://127.0.0.1:5801", tmp_path / "e.log")

    def test_a_closed_system_socket_list_falls_back_to_each_process(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """macOS lists every socket only to root; each process still lists its own to its user."""

        def denied(**kwargs: object) -> list[object]:
            raise record_presets.psutil.AccessDenied()

        log = _log(tmp_path, _TEI_CUDA_LOG)
        monkeypatch.setattr(record_presets.psutil, "net_connections", denied)
        with _engine(None) as url:
            record_presets.require_engine_output(url, Path(log))

    def test_the_process_scan_skips_a_process_that_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        psutil = record_presets.psutil

        class Proc:
            def __init__(self, pid: int, conns: list[object] | None) -> None:
                self.pid, self._conns = pid, conns

            def net_connections(self, kind: str) -> list[object]:
                if self._conns is None:
                    raise psutil.AccessDenied(self.pid)
                return self._conns

        def sock(port: int, status: str = psutil.CONN_LISTEN) -> object:
            return SimpleNamespace(status=status, laddr=SimpleNamespace(ip="127.0.0.1", port=port))

        procs = [
            Proc(11, None),
            Proc(12, [sock(5801, psutil.CONN_ESTABLISHED), sock(6000)]),
            Proc(13, [sock(5801)]),
        ]
        monkeypatch.setattr(psutil, "process_iter", lambda: iter(procs))
        assert record_presets._own_listeners(5801) == {13}

    def test_a_process_that_vanished_holds_nothing_open(self, tmp_path: Path) -> None:
        assert not record_presets._holds_open(2**22 + 7, tmp_path / "e.log")

    def test_a_run_with_no_evidence_is_recorded_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _engine(None) as url:
            rc, out = _record(monkeypatch, tmp_path, url, "cpu")
        assert rc == 0
        assert "devices-from: none" in out.with_suffix(".log").read_text(encoding="utf-8")
