"""The recorder writes the backend the engine reports, and refuses a run whose report differs."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
