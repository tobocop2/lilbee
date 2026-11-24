"""A stalled model download tells the caller while it runs, not only at the end.

huggingface_hub resumes a stalled file in place and says so only in its log.
The download child turns each of those log records into a retry notice and
sends it to the parent, which hands it to the caller's ``on_retry``.
"""

from __future__ import annotations

import io
import logging
import socket
import threading
from pathlib import Path
from typing import Any

import httpx2
import pytest

from lilbee.catalog import download_process as dp
from lilbee.catalog import hf_client
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.providers.fleet.engines import LLAMA
from lilbee.runtime.cancellation import TaskCancelledError

_SIZE = 64 * 1024
_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_ETAG = "d811d811d811d811d811d811d811d811d811d811"
_REPO = "acme/stall-GGUF"
_FILE = "stall-Q4_K_M.gguf"


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo=_REPO,
        gguf_filename=_FILE,
        size_gb=0.001,
        min_ram_gb=0.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


class _StallServer:
    """A Hub stand-in whose file downloads go quiet after the headers.

    Every HEAD gets the Hub's metadata headers. A closed-range GET (the GGUF
    header probe) gets 404, which reads as a model. Every other GET gets its
    status line and headers; with *half_first* the first one also gets half the
    body. Then the socket stays open and silent until the test ends.
    """

    def __init__(self, *, half_first: bool) -> None:
        self.half_first = half_first
        self.gets = 0
        self._held: list[socket.socket] = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        for conn in [*self._held, self._sock]:
            conn.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return
            data += chunk
        lines = data.split(b"\r\n")
        method = lines[0].split(b" ")[0]
        rng = next((ln.split(b"=", 1)[1] for ln in lines if ln.lower().startswith(b"range:")), b"")
        if method == b"HEAD":
            conn.sendall(
                (
                    "HTTP/1.1 200 OK\r\n"
                    f'X-Repo-Commit: {_COMMIT}\r\nX-Linked-Etag: "{_ETAG}"\r\n'
                    f'ETag: "{_ETAG}"\r\nX-Linked-Size: {_SIZE}\r\n'
                    f"Content-Length: {_SIZE}\r\nConnection: close\r\n\r\n"
                ).encode()
            )
            conn.close()
            return
        start, _, end = rng.partition(b"-")
        if end:
            conn.sendall(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            conn.close()
            return
        self.gets += 1
        self._held.append(conn)
        first = int(start) if start else 0
        if first:
            conn.sendall(
                (
                    f"HTTP/1.1 206 Partial Content\r\nContent-Length: {_SIZE - first}\r\n"
                    f"Content-Range: bytes {first}-{_SIZE - 1}/{_SIZE}\r\n\r\n"
                ).encode()
            )
        else:
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {_SIZE}\r\n\r\n".encode())
        if self.half_first and self.gets == 1:
            conn.sendall(b"x" * (_SIZE // 2))


@pytest.fixture
def stall_server() -> Any:
    server = _StallServer(half_first=True)
    yield server
    server.close()


class TestTheHubResumeRecord:
    """The hub's own log text is not a contract, so a real hub transfer pins it."""

    def test_every_resume_of_the_real_hub_reaches_the_sink_naming_the_timeout(
        self, stall_server: _StallServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from huggingface_hub import constants
        from huggingface_hub import file_download as hub

        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
        monkeypatch.setattr(constants, "HF_HUB_DOWNLOAD_TIMEOUT", 0.3)
        monkeypatch.setattr(hub.time, "sleep", lambda _s: None)
        causes: list[str] = []

        with (
            hf_client.hub_resumes_to(causes.append),
            pytest.raises(httpx2.TimeoutException) as raised,
        ):
            hub.http_get(f"{stall_server.url}/f", io.BytesIO(), expected_size=_SIZE)

        # Half the body arrived, then the hub resumed in place until it gave up.
        assert causes == ["timed out"] * (dp._HUB_ATTEMPTS - 1), raised.value
        assert stall_server.gets == dp._HUB_ATTEMPTS

    def test_the_record_is_still_dropped_from_the_log(self) -> None:
        hub_logger = logging.getLogger("huggingface_hub.file_download")
        record = logging.LogRecord(
            hub_logger.name,
            logging.WARNING,
            __file__,
            0,
            "Error while downloading from %s: %s\nTrying to resume download...",
            ("http://127.0.0.1:1/f", "timed out"),
            None,
        )
        causes: list[str] = []

        with hf_client.hub_resumes_to(causes.append):
            kept = hub_logger.filter(record)

        assert (kept, causes) == (False, ["timed out"])

    def test_with_no_sink_open_the_record_is_dropped_and_goes_nowhere(self) -> None:
        hub_logger = logging.getLogger("huggingface_hub.file_download")
        record = logging.LogRecord(
            hub_logger.name,
            logging.WARNING,
            __file__,
            0,
            "x: y\nTrying to resume download",
            (),
            None,
        )

        assert not hub_logger.filter(record)

    def test_an_unrelated_record_does_not_reach_the_sink(self) -> None:
        hub_logger = logging.getLogger("huggingface_hub.file_download")
        record = logging.LogRecord(
            hub_logger.name, logging.WARNING, __file__, 0, "Downloaded f in 2s", (), None
        )
        causes: list[str] = []

        with hf_client.hub_resumes_to(causes.append):
            assert hub_logger.filter(record)

        assert causes == []

    def test_a_record_with_no_cause_hands_over_its_first_line(self) -> None:
        assert hf_client._resume_cause("Trying to resume download") == ("Trying to resume download")


class _Conn:
    """The send side of the child's pipe, recording what it sends."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    def send(self, message: Any) -> None:
        self.sent.append(message)


class TestTheChildCountsResumes:
    def test_resumes_count_up_and_restart_once_bytes_arrive(self) -> None:
        conn = _Conn()
        progress = dp._PipeProgress(conn)  # type: ignore[arg-type]
        heard: list[tuple[int, int, str]] = []
        resumes = dp._HubResumes(_REPO, progress, lambda *notice: heard.append(notice))

        resumes("timed out")
        resumes("timed out")
        progress(10, 100)
        resumes("timed out")

        cause = f"Network error downloading {_REPO}: timed out"
        assert heard == [(1, 6, cause), (2, 6, cause), (1, 6, cause)]

    def test_the_child_sends_each_resume_and_each_outer_retry_to_the_parent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import lilbee.catalog.download as dl

        conn = _Conn()
        monkeypatch.setattr(dp, "_silence_output", lambda: None)

        def _fetch(entry: Any, models_dir: Any, token: Any, engine: Any, **signals: Any) -> Path:
            logging.getLogger("huggingface_hub.file_download").warning(
                "Error while downloading from %s: %s\nTrying to resume download...",
                "http://127.0.0.1:1/f",
                "timed out",
            )
            signals["on_retry"](1, 3, "Network error downloading acme/stall-GGUF: reset")
            return tmp_path / "x.gguf"

        monkeypatch.setattr(dl, "fetch_model_files", _fetch)

        dp._run_download_child(conn, _entry(), str(tmp_path), None, LLAMA)  # type: ignore[arg-type]

        assert [m.kind for m in conn.sent] == ["retry", "retry", "done"]
        assert (conn.sent[0].attempt, conn.sent[0].attempts) == (1, 6)
        assert "timed out" in conn.sent[0].cause
        assert (conn.sent[1].attempt, conn.sent[1].attempts) == (1, 3)


class _Flag:
    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value


@pytest.mark.download_child
@pytest.mark.timeout(120)
def test_a_real_child_reports_the_hub_resume_to_the_parent_naming_the_timeout(
    stall_server: _StallServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real spawned child, the real hub and a server that stalls mid-body."""
    monkeypatch.setenv("HF_ENDPOINT", stall_server.url)
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1")  # the hub reads it as whole seconds
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    for name in ("HF_HUB_OFFLINE", "HF_TOKEN", "LILBEE_HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    cancel = _Flag()
    heard: list[tuple[int, int, str]] = []

    def on_retry(attempt: int, attempts: int, cause: str) -> None:
        heard.append((attempt, attempts, cause))
        cancel.value = True  # one notice is the proof; stop the download here

    with pytest.raises(TaskCancelledError):
        dp.download_in_subprocess(
            _entry(), tmp_path, None, LLAMA, on_progress=None, on_retry=on_retry, cancel=cancel
        )

    assert len(heard) == 1
    attempt, attempts, cause = heard[0]
    assert (attempt, attempts) == (1, dp._HUB_ATTEMPTS)
    assert cause == f"Network error downloading {_REPO}: timed out"
