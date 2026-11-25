"""A cancel of an add over real HTTP removes what the add did not finish.

Every test drives the real app over real HTTP (uvicorn on a loopback port), with a real LanceDB
store and the real sync. Only the embedder is a stand-in: it parks on any text that holds the
slow marker until the test releases it, so the add is stopped in the middle of a file.

The contract:

* ``POST /api/add`` takes an optional ``add_id`` in its body, ``POST /api/add/upload`` takes it as
  a query parameter. ``POST /api/add/cancel`` with ``{"add_id": ...}`` answers 200 with
  ``{"add_id": ..., "cancelling": true}`` while that add runs.
* After the cancel the add stream ends with one ``done`` event whose data carries
  ``"cancelled": true`` and ``not_added``, the names the add did not add.
* A source root with no file indexed since the add's sync started is un-registered; its files on
  disk are not touched. A root with a finished file stays, and the next sync adds the rest.
* An upload's copy in the documents folder that the upload created and the sync did not index is
  deleted.
* A stream that is only dropped, with no cancel call, keeps its sources.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import numpy as np
import pytest

import lilbee.app.services as svc_mod
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.store import Store
from tests.conftest import make_mock_services
from tests.server.conftest import parse_sse_events
from tests.server.test_handlers import _auth_headers, _ServedApp

SLOW = "SLOWMARKER"
_WAIT = 60.0


class _Gate:
    """An embedder that parks on the slow marker until released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.streams: list[_Stream] = []

    def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        if any(SLOW in text for text in texts):
            self.entered.set()
            self.release.wait(_WAIT * 2)
        return [np.full(768, 0.1, dtype=np.float32) for _ in texts]

    def embedder(self) -> mock.MagicMock:
        embedder = mock.MagicMock()
        embedder.embed_batch.side_effect = lambda texts, **_kw: self._vectors(list(texts))
        embedder.embed.side_effect = lambda text, **_kw: self._vectors([text])[0]
        embedder.embed_query.return_value = np.full(768, 0.1, dtype=np.float32)
        embedder.validate_model.return_value = True
        return embedder


@pytest.fixture
def gate() -> Iterator[_Gate]:
    g = _Gate()
    yield g
    g.release.set()


@pytest.fixture(autouse=True)
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: _Gate) -> Iterator[Store]:
    """A data root with a real store, the parking embedder, and config.toml read by each sync."""
    snapshot = cfg.model_copy()
    root = tmp_path / "library"
    cfg.data_root = root
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.linked_roots = {}
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    store = Store(cfg)
    svc_mod.set_services(make_mock_services(store=store, embedder=gate.embedder()))
    try:
        yield store
    finally:
        gate.release.set()
        svc_mod.set_services(None)
        for name in type(cfg).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


@contextlib.contextmanager
def _served(gate: _Gate) -> Iterator[httpx.Client]:
    """The served app and a client; the gate opens and each add stream ends before the server stops.

    A parked add that outlived its server would hold the sync of the next test.
    """
    with _ServedApp() as url, httpx.Client(base_url=url, timeout=_WAIT) as http:
        try:
            yield http
        finally:
            gate.release.set()
            for stream in gate.streams:
                stream.ended.wait(_WAIT)


class _Stream:
    """Reads one SSE response on its own thread and records each event as it lands."""

    def __init__(self, http: httpx.Client, method: str, url: str, **kwargs: Any) -> None:
        self.events: list[tuple[str, dict]] = []
        self.ended = threading.Event()
        self.status: int | None = None
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._read, args=(http, method, url), kwargs=kwargs, daemon=True
        )
        self._thread.start()

    def _read(self, http: httpx.Client, method: str, url: str, **kwargs: Any) -> None:
        try:
            with http.stream(method, url, **kwargs) as resp:
                self.status = resp.status_code
                block = ""
                for line in resp.iter_lines():
                    block += line + "\n"
                    if line == "":
                        with self._cond:
                            self.events.extend(parse_sse_events(block.encode()))
                            self._cond.notify_all()
                        block = ""
        finally:
            with self._cond:
                self.ended.set()
                self._cond.notify_all()

    def wait_for(self, predicate: Callable[[list[tuple[str, dict]]], bool]) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: predicate(self.events), _WAIT)

    def done(self) -> dict | None:
        assert self.ended.wait(_WAIT), "the add stream never ended"
        frames = [data for event, data in self.events if event == "done"]
        assert len(frames) <= 1, f"more than one done event: {frames}"
        return frames[0] if frames else None


def _registry() -> dict[str, str]:
    return dict(settings.load(cfg.data_root).get("linked_roots") or {})


def _keys(store: Store) -> list[str]:
    return sorted(source["filename"] for source in store.get_sources())


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _cancel(http: httpx.Client, add_id: str) -> None:
    resp = http.post("/api/add/cancel", json={"add_id": add_id}, headers=_auth_headers())
    assert resp.status_code == 200, (
        f"POST /api/add/cancel answered {resp.status_code}: REST has no explicit cancel for an add"
    )
    assert resp.json() == {"add_id": add_id, "cancelling": True}


def _sync(http: httpx.Client) -> None:
    resp = http.post("/api/sync", json={}, headers=_auth_headers(), timeout=_WAIT * 2)
    assert resp.status_code in (200, 201), resp.text


def test_a_cancel_removes_a_file_the_add_did_not_finish(tmp_path, library, gate):
    src = _write(tmp_path / "src" / "slow.md", f"# Slow\n\n{SLOW} a page that never embeds.\n")
    before = src.read_bytes()
    body = {"paths": [str(src)], "add_id": "t539o-one"}

    with _served(gate) as http:
        stream = _Stream(http, "POST", "/api/add", json=body, headers=_auth_headers())
        gate.streams.append(stream)
        assert gate.entered.wait(_WAIT), "the add never reached the embedder"
        assert _registry().get("slow.md") == str(src.resolve())
        _cancel(http, "t539o-one")
        done = stream.done()
        assert done is not None, "a cancelled add sent no done event"
        assert done.get("cancelled") is True
        assert done.get("not_added") == ["slow.md"]
        assert "slow.md" not in _registry()
        assert "slow.md" not in _keys(library)
        gate.release.set()
        _sync(http)
        assert "slow.md" not in _keys(library), "the next sync indexed the cancelled file"

    assert src.read_bytes() == before, "the cancel changed the user's file"


def test_a_cancel_keeps_a_folder_with_a_finished_file(tmp_path, library, gate):
    notes = tmp_path / "src" / "notes"
    _write(notes / "fast.md", "# Fast\n\nA page about the harbour schedule.\n")
    _write(notes / "slow.md", f"# Slow\n\n{SLOW} a page that parks.\n")
    body = {"paths": [str(notes)], "add_id": "t539o-folder"}

    def fast_done(events: list[tuple[str, dict]]) -> bool:
        return any(
            e == "file_done" and str(d.get("file", "")).endswith("fast.md") for e, d in events
        )

    with _served(gate) as http:
        stream = _Stream(http, "POST", "/api/add", json=body, headers=_auth_headers())
        gate.streams.append(stream)
        assert gate.entered.wait(_WAIT), "the add never reached the embedder"
        assert stream.wait_for(fast_done), "fast.md never finished"
        _cancel(http, "t539o-folder")
        done = stream.done()
        assert done is not None and done.get("cancelled") is True
        assert done.get("not_added") == []
        assert _registry().get("notes") == str(notes.resolve())
        assert "notes/fast.md" in _keys(library)
        assert "notes/slow.md" not in _keys(library)
        gate.release.set()
        _sync(http)
        assert "notes/slow.md" in _keys(library), "the next sync did not add the rest"


def test_a_cancel_deletes_the_upload_copy_it_did_not_index(library, gate):
    content = f"# Upload\n\n{SLOW} an uploaded page.\n".encode()
    files = {"data": ("slow-upload.md", content)}

    with _served(gate) as http:
        stream = _Stream(
            http,
            "POST",
            "/api/add/upload",
            files=files,
            params={"add_id": "t539o-upload"},
            headers=_auth_headers(),
        )
        gate.streams.append(stream)
        assert gate.entered.wait(_WAIT), "the upload never reached the embedder"
        assert (cfg.documents_dir / "slow-upload.md").is_file()
        _cancel(http, "t539o-upload")
        done = stream.done()
        assert done is not None and done.get("cancelled") is True
        assert done.get("not_added") == ["slow-upload.md"]
        assert not (cfg.documents_dir / "slow-upload.md").exists()
        assert "slow-upload.md" not in _keys(library)
        gate.release.set()
        _sync(http)
        assert "slow-upload.md" not in _keys(library)


def test_a_dropped_stream_keeps_the_source(tmp_path, library, gate):
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    text = f"# Kept\n\n{SLOW} a page that stays registered after the stream drops.\n"
    src = _write(tmp_path / "src" / "kept.md", text)
    released = threading.Event()
    release = IngestLockRegistry.release

    def _recording(registry, acquired):
        held = [name for name, _lock in acquired]  # the release empties the list
        release(registry, acquired)
        if "kept.md" in held:
            released.set()

    body = {"paths": [str(src)], "add_id": "t539o-dropped"}
    with (
        mock.patch.object(IngestLockRegistry, "release", _recording),
        _served(gate) as http,
    ):
        with http.stream("POST", "/api/add", json=body, headers=_auth_headers()) as resp:
            # A collected line iterator closes its response, so the test keeps this one.
            lines = resp.iter_lines()
            first = ""
            with contextlib.suppress(StopIteration):
                first = next(line for line in lines if line.startswith("event:"))
            assert gate.entered.wait(_WAIT), f"the add never reached the embedder; first {first!r}"
        gate.release.set()
        assert released.wait(_WAIT), "the add never unwound"

    assert _registry().get("kept.md") == str(src.resolve())
