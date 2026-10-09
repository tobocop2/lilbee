"""Tests for the /api/add endpoint and SSE progress streaming."""

import asyncio
import contextlib
import hashlib
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import uvicorn
from litestar.testing import AsyncTestClient
from xberg import Metadata

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.server import auth as _auth_mod
from tests._async_wait import poll_until
from tests.server.conftest import parse_sse_events as _parse_sse_events


def _auth_headers() -> dict[str, str]:
    """Return Authorization header using the current session token."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.data_root = tmp_path
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.linked_roots = {}
    cfg.concept_graph = False
    # Configured-but-not-installed, the state this module always ran under
    # before the defaults became unconfigured.
    cfg.chat_model = "owner/chat-GGUF/chat.Q4_K_M.gguf"
    cfg.embedding_model = "owner/embed-GGUF/embed.Q8_0.gguf"
    yield docs
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so handlers never touch real backends."""
    from tests.conftest import make_mock_services

    embedder = mock.MagicMock()
    embedder.embed.return_value = np.full(768, 0.1, dtype=np.float32)
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    embedder.validate_model.return_value = True
    services = make_mock_services(embedder=embedder)
    # /api/chat now goes through chat_dispatch, which validates the requested
    # model against the KnownModelCache. Pre-load both the registry and the
    # cache so resolve() finds cfg.chat_model.
    chat_manifest = mock.MagicMock()
    chat_manifest.ref = cfg.chat_model
    chat_manifest.task = "chat"
    services.registry.list_installed = mock.MagicMock(return_value=[chat_manifest])
    services.known_models.refs = mock.MagicMock(return_value={cfg.chat_model})
    services.known_models.resolve = mock.MagicMock(
        side_effect=lambda model: model if model == cfg.chat_model else None
    )
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture(autouse=True)
def reset_ingest_locks():
    """Clear per-source ingest locks so each test starts with a clean registry.

    Locks are bound to the event loop; leaking them across tests produces
    cryptic 'attached to a different loop' errors.
    """
    from lilbee.app.services import get_services

    get_services().ingest_lock_registry.reset()
    yield
    get_services().ingest_lock_registry.reset()


def _make_xberg_result(text: str = "Some extracted text. " * 20, num_chunks: int = 1):
    chunks = []
    for i in range(num_chunks):
        chunk_text = text[i * len(text) // num_chunks : (i + 1) * len(text) // num_chunks]
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = mock.MagicMock(chunk_index=i, first_page=None, last_page=None)
        chunks.append(chunk)
    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    result.pages = []
    result.metadata = Metadata()
    return result


@mock.patch(
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
)
class TestAddEndpoint:
    async def test_add_single_file(self, mock_extract_file, isolated_env, tmp_path):
        """POST /api/add with a valid file streams SSE events and adds it."""
        from lilbee.server.app import create_app

        src = tmp_path / "input.txt"
        src.write_text("Hello world content for testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        event_types = [e[0] for e in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types

    async def test_add_upload_writes_content_and_indexes(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """POST /api/add/upload writes the uploaded bytes into documents_dir and ingests them."""
        from lilbee.server.app import create_app

        content = b"Uploaded content that the server could not have read by path."
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("notes.txt", content, "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        event_types = [e[0] for e in _parse_sse_events(resp.content)]
        assert "file_start" in event_types
        assert "done" in event_types
        # The client's bytes landed in the corpus even though no server path existed.
        assert (isolated_env / "notes.txt").read_bytes() == content

    @pytest.mark.parametrize(
        ("indexed", "moved"),
        [
            pytest.param(["index.md"], True, id="one-match-moves"),
            pytest.param(["index.md", "old/index.md"], False, id="two-matches-write-plainly"),
        ],
    )
    async def test_add_upload_moves_the_one_file_with_the_same_content(
        self, mock_extract_file, isolated_env, mock_svc, indexed, moved
    ):
        """The same bytes under a new name move the indexed file, so sync repoints its key."""
        from lilbee.server.app import create_app

        content = b"The same note, uploaded once by basename and once by path."
        digest = hashlib.sha256(content).hexdigest()
        for name in indexed:
            (isolated_env / name).parent.mkdir(parents=True, exist_ok=True)
            (isolated_env / name).write_bytes(content)
        mock_svc.store.get_sources.return_value = [
            {"filename": name, "file_hash": digest, "chunk_count": 1} for name in indexed
        ]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("notes/alpha/index.md", content, "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert (isolated_env / "notes/alpha/index.md").read_bytes() == content
        assert (isolated_env / "index.md").exists() is not moved

    async def test_add_upload_writes_plainly_when_the_matching_file_is_gone(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """A source row whose file left the disk cannot be moved; the upload is written as is."""
        from lilbee.server.app import create_app

        content = b"Indexed once, then its file was deleted by hand."
        mock_svc.store.get_sources.return_value = [
            {"filename": "gone.md", "file_hash": hashlib.sha256(content).hexdigest()}
        ]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("notes/gone.md", content, "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert (isolated_env / "notes/gone.md").read_bytes() == content

    async def test_add_upload_leaves_a_file_with_other_content_alone(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        from lilbee.server.app import create_app

        (isolated_env / "index.md").write_bytes(b"different bytes")
        mock_svc.store.get_sources.return_value = [
            {"filename": "index.md", "file_hash": hashlib.sha256(b"different bytes").hexdigest()}
        ]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("notes/index.md", b"new bytes", "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert (isolated_env / "index.md").read_bytes() == b"different bytes"
        assert (isolated_env / "notes/index.md").read_bytes() == b"new bytes"

    async def test_add_upload_rejects_path_traversal_filename(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """A crafted ../ filename is rejected outright; nothing is written."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("../../escape.txt", b"safe", "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 400
        assert not (isolated_env / "escape.txt").exists()
        assert not (isolated_env.parent.parent / "escape.txt").exists()

    async def test_add_upload_preserves_relative_paths(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """An uploaded tree keeps its layout: same-basename files don't collide."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[
                    ("data", ("pkg_a/__init__.py", b"a", "text/plain")),
                    ("data", ("pkg_b/__init__.py", b"b", "text/plain")),
                ],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert (isolated_env / "pkg_a" / "__init__.py").read_bytes() == b"a"
        assert (isolated_env / "pkg_b" / "__init__.py").read_bytes() == b"b"

    async def test_add_upload_many_small_files_accepted(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """Hundreds of small uploads are fine: the guard is body size, not count."""
        from lilbee.server.app import create_app

        files = [("data", (f"f{i}.txt", b"x", "text/plain")) for i in range(285)]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add/upload", files=files, headers=_auth_headers())
        assert resp.status_code == 201

    def test_validate_uploads_rejects_bad_input(self, mock_extract_file, isolated_env):
        """validate_upload_names guards empty/malformed names and keeps relative paths."""
        from lilbee.server.handlers.ingest import validate_upload_names

        with pytest.raises(ValueError, match="no files"):
            validate_upload_names([])
        with pytest.raises(ValueError, match="invalid upload filename"):
            validate_upload_names([""])
        with pytest.raises(ValueError, match="must be relative"):
            validate_upload_names(["/etc/passwd"])
        with pytest.raises(ValueError, match="must be relative"):
            validate_upload_names(["C:\\dir\\file.txt"])
        with pytest.raises(ValueError, match="may not contain"):
            validate_upload_names(["../../a/b.txt"])
        with pytest.raises(ValueError, match="vector graphic, not a document"):
            validate_upload_names(["logo.svg"])
        # Relative paths survive so an uploaded tree keeps its layout.
        assert validate_upload_names(["src/pkg/__init__.py"]) == ["src/pkg/__init__.py"]
        # Backslash separators and ./ prefixes normalize to POSIX relative form.
        assert validate_upload_names(["./src\\a.py"]) == ["src/a.py"]
        # A multipart part may carry no filename at all.
        with pytest.raises(ValueError, match="invalid upload filename"):
            validate_upload_names([None])

    async def test_add_nonexistent_file_in_errors(self, mock_extract_file, isolated_env, tmp_path):
        """Nonexistent paths appear in the summary errors list."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": ["/no/such/file.txt"]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "/no/such/file.txt" in summary["errors"]

    async def test_add_refused_format_in_errors(self, mock_extract_file, isolated_env, tmp_path):
        """A file whose format lilbee does not index is refused at add time, with the reason."""
        from lilbee.server.app import create_app

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(drawing)]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert summary["errors"] == ["logo.svg: vector graphic, not a document"]
        assert summary["copied"] == []

    async def test_add_where_nothing_reaches_the_corpus_skips_the_sync(
        self, mock_extract_file, isolated_env
    ):
        """A batch with nothing reaching the corpus must not run the
        whole-vault sync, which holds the ingest lock for nothing."""
        from lilbee.server.app import create_app

        with mock.patch("lilbee.data.ingest.sync", new_callable=Mock) as sync_mock:
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/add", json={"paths": ["/no/such/file.txt"]}, headers=_auth_headers()
                )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert summary["copied"] == []
        assert summary["name_taken"] == [] and summary["overlapping"] == []
        sync_mock.assert_not_called()

    async def test_add_with_force_flag(self, mock_extract_file, isolated_env, tmp_path):
        """The force flag re-points a root whose label is already taken."""
        from lilbee.core import settings
        from lilbee.server.app import create_app

        one = tmp_path / "a" / "dup"
        one.mkdir(parents=True)
        settings.set_value(cfg.data_root, "linked_roots", {"dup": str(one)})
        two = tmp_path / "b" / "dup"
        two.mkdir(parents=True)

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(two)], "force": True}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "dup" in summary["copied"]
        assert cfg.linked_roots["dup"] == str(two.resolve())

    async def test_done_event_carries_the_add_summary(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """The done event carries the add summary and the sync lists under it."""
        from lilbee.server.app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for done event testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        done_data = next(d for t, d in events if t == "done")
        assert done_data["copied"] == ["doc.txt"]
        assert done_data["errors"] == []
        assert done_data["sync"]["added"] == ["doc.txt"]
        assert done_data["sync"]["failed"] == []

    async def test_file_start_has_total_and_current(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """file_start event includes total_files and current_file."""
        from lilbee.server.app import create_app

        src = tmp_path / "progress.txt"
        src.write_text("Progress tracking test.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        file_start = next(d for t, d in events if t == "file_start")
        assert file_start["total_files"] >= 1
        assert file_start["current_file"] >= 1

    async def test_add_with_ocr_applies_to_that_request_only(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """The ocr field overrides the setting during this add's extraction only."""
        from lilbee.data.extract.document import _effective_ocr_mode
        from lilbee.server.app import create_app

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 content")
        cfg.ocr = OcrMode.AUTO
        observed: list[OcrMode] = []

        async def _capture(*args, **kwargs):
            observed.append(_effective_ocr_mode())
            return _make_xberg_result()

        mock_extract_file.side_effect = _capture
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(src)], "ocr": "all"},
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert observed and set(observed) == {OcrMode.ALL}
        assert cfg.ocr is OcrMode.AUTO

    @pytest.mark.parametrize(
        ("route", "body", "detail"),
        [
            ("/api/add", {"paths": ["x"], "ocr": "some"}, "'auto', 'all' or 'off'"),
            ("/api/add", {"paths": ["x"], "enable_ocr": False}, "enable_ocr is replaced by ocr"),
            ("/api/sync", {"ocr": "some"}, "'auto', 'all' or 'off'"),
            ("/api/sync", {"enable_ocr": False}, "enable_ocr is replaced by ocr"),
        ],
    )
    async def test_a_bad_or_retired_ocr_field_is_a_400(
        self, mock_extract_file, isolated_env, route, body, detail
    ):
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(route, json=body, headers=_auth_headers())

        assert resp.status_code == 400
        assert detail in resp.text
        mock_extract_file.assert_not_called()

    async def test_add_emits_heartbeat_during_slow_sync(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """Root-cause fix for obsidian-lilbee-v8y.

        A long-running vision OCR pass can go >120s without emitting a
        progress event. The plugin's STREAM_IDLE_TIMEOUT_MS (120s) then
        aborts the stream. The server must emit 'heartbeat' SSE events at
        cfg.sse_heartbeat_interval whenever the producer queue is idle so
        the plugin's withIdleTimeout keeps resetting. This test drives a
        sync that sleeps longer than the heartbeat interval and asserts
        at least one heartbeat event was delivered to the HTTP client.
        """
        from lilbee.server.app import create_app

        src = tmp_path / "slow.txt"
        src.write_text("Slow sync content for heartbeat test.")
        cfg.sse_heartbeat_interval = 0.2

        async def slow_sync(
            force_rebuild=False, quiet=False, *, on_progress=None, cancel=None, **_kw
        ):
            from lilbee.data.ingest import SyncResult

            await asyncio.sleep(0.6)
            return SyncResult(added=["slow.txt"])

        with mock.patch("lilbee.data.ingest.sync", side_effect=slow_sync):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
                )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        heartbeats = [d for t, d in events if t == "heartbeat"]
        assert heartbeats, f"expected heartbeat events during slow sync, got {events}"
        assert "ts" in heartbeats[0]


@mock.patch(
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
)
class TestIngestStreamTerminalEvent:
    """Each ingest stream closes with exactly one ``done`` frame.

    A client dispatches on the event name. Two frames under one name are
    distinguishable only by payload shape, which means by arrival order, and
    nothing in the protocol fixes that order.
    """

    async def test_sync_stream_closes_with_one_done(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """POST /api/sync ends on a single done carrying the sync result."""
        from lilbee.server.app import create_app

        (isolated_env / "indexed.txt").write_text(
            "Content the sync pass indexes.", encoding="utf-8"
        )

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/sync", headers=_auth_headers())

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        names = [name for name, _payload in events]
        assert names.count("done") == 1, names
        assert names[-1] == "done", names
        assert events[-1][1]["added"] == ["indexed.txt"]

    async def test_add_stream_closes_with_one_done(self, mock_extract_file, isolated_env, tmp_path):
        """POST /api/add ends on a single done carrying the add summary."""
        from lilbee.server.app import create_app

        src = tmp_path / "added.txt"
        src.write_text("Content the add pass copies and indexes.", encoding="utf-8")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        names = [name for name, _payload in events]
        assert names.count("done") == 1, names
        assert names[-1] == "done", names
        assert events[-1][1]["copied"] == ["added.txt"]

    async def test_upload_stream_closes_with_one_done(self, mock_extract_file, isolated_env):
        """POST /api/add/upload ends on a single done carrying the upload summary."""
        from lilbee.server.app import create_app

        content = b"Content the upload pass writes and indexes."

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files=[("data", ("uploaded.txt", content, "text/plain"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        names = [name for name, _payload in events]
        assert names.count("done") == 1, names
        assert names[-1] == "done", names
        assert events[-1][1]["copied"] == ["uploaded.txt"]

    async def test_upload_ocr_options_reach_the_extraction_config(
        self, mock_extract_file, isolated_env
    ):
        """POST /api/add/upload?ocr=...&ocr_timeout=... overrides OCR for this upload."""
        from lilbee.data.extract.document import _effective_ocr_mode, _effective_ocr_timeout
        from lilbee.server.app import create_app

        observed: dict[str, object] = {}

        async def _capture(*args, **kwargs):
            observed["ocr"] = _effective_ocr_mode()
            observed["ocr_timeout"] = _effective_ocr_timeout()
            return _make_xberg_result()

        mock_extract_file.side_effect = _capture

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                params={"ocr": "off", "ocr_timeout": "17"},
                files=[("data", ("scan.pdf", b"content", "application/pdf"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert observed == {"ocr": OcrMode.OFF, "ocr_timeout": 17.0}

    @pytest.mark.parametrize(
        ("params", "detail"),
        [
            ({"ocr": "some"}, "ocr must be one of auto, all, off; got 'some'"),
            (
                {"enable_ocr": "false"},
                "enable_ocr is replaced by ocr; set ocr to one of auto, all, off",
            ),
        ],
    )
    async def test_upload_refuses_a_bad_or_retired_ocr_query(
        self, mock_extract_file, isolated_env, params, detail
    ):
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                params=params,
                files=[("data", ("scan.pdf", b"content", "application/pdf"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 400
        assert detail in resp.text
        mock_extract_file.assert_not_called()

    async def test_upload_rejects_negative_ocr_timeout(self, mock_extract_file, isolated_env):
        """A negative ocr_timeout is a 400; extraction never runs."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                params={"ocr_timeout": "-5"},
                files=[("data", ("scan.pdf", b"content", "application/pdf"))],
                headers=_auth_headers(),
            )

        assert resp.status_code == 400
        mock_extract_file.assert_not_called()


class TestAddValidation:
    async def test_empty_paths_returns_400(self, isolated_env):
        """POST /api/add with empty paths list returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": []}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_missing_paths_returns_400(self, isolated_env):
        """POST /api/add without paths key returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"force": True}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_negative_ocr_timeout_returns_400(self, isolated_env, tmp_path):
        """POST /api/add with a negative ocr_timeout returns 400 before any file copy."""
        from lilbee.server.app import create_app

        src = tmp_path / "added.txt"
        src.write_text("content", encoding="utf-8")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(src)], "ocr_timeout": -5},
                headers=_auth_headers(),
            )
        assert resp.status_code == 400
        assert not (isolated_env / "added.txt").exists()

    async def test_hundreds_of_paths_accepted(self, isolated_env, tmp_path):
        """POST /api/add has no file-count cap (paths can be nonexistent)."""
        from lilbee.server.app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(285)]
        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post("/api/add", json={"paths": paths}, headers=_auth_headers())
        # Request is valid even though every path is nonexistent.
        assert resp.status_code == 201


class TestSseStreamCallback:
    async def test_callback_enqueues_formatted_sse(self):
        """The SSE callback formats events correctly."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        sse.callback("file_start", {"file": "test.txt", "total_files": 1, "current_file": 1})
        item = sse.queue.get_nowait()
        assert item is not None
        assert item.startswith("event: file_start\n")
        assert '"file": "test.txt"' in item

    async def test_callback_from_thread_uses_threadsafe(self):
        """When called from a worker thread, uses call_soon_threadsafe."""
        import concurrent.futures

        from lilbee.server.handlers import SseStream

        sse = SseStream()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, sse.callback, "embed", {"chunk": 1})
            await future

        # Give the event loop a tick to process the call_soon_threadsafe
        await asyncio.sleep(0)
        item = sse.queue.get_nowait()
        assert item is not None
        assert "embed" in item


class TestSseStreamDrain:
    async def test_drain_delivers_every_event_across_poll_boundaries(self):
        """Events put near the poll timeout boundary are never dropped."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        total = 12

        async def produce() -> None:
            for i in range(total):
                # Sleeps straddle drain's 0.1s poll period to hit the boundary.
                await asyncio.sleep(0.03 + (i % 4) * 0.025)
                sse.queue.put_nowait(f"event: tick\ndata: {i}\n\n")
            sse.queue.put_nowait(None)

        task = asyncio.create_task(produce())
        received = [item async for item in sse.drain(task, "drain test")]
        assert len([r for r in received if "tick" in r]) == total

    async def test_drain_flushes_events_behind_the_sentinel(self):
        """A producer that finishes before drain starts still gets its progress out."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        # Sentinel already queued; the progress callback is still in the loop's
        # ready callbacks, as happens when the producer outruns the consumer.
        sse.queue.put_nowait(None)
        asyncio.get_running_loop().call_soon(sse.queue.put_nowait, "event: embed\ndata: {}\n\n")
        task = asyncio.create_task(asyncio.sleep(0))
        received = [item async for item in sse.drain(task, "drain flush test")]
        assert any("embed" in r for r in received)


class TestOptionsPassthrough:
    """Verify generation options are extracted from request body and passed through."""

    async def test_ask_passes_options(self, isolated_env):
        from lilbee.server.app import create_app
        from lilbee.server.handlers.sse import _resolve_generation_options

        with mock.patch(
            "lilbee.server.handlers.rag._resolve_generation_options",
            wraps=_resolve_generation_options,
        ) as spy:
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/ask",
                    json={"question": "test", "options": {"temperature": 0.3}},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 201
        assert "answer" in resp.json()
        # The body's options must actually reach the generation-options resolver,
        # not just produce a successful response.
        spy.assert_any_call({"temperature": 0.3})

    async def test_chat_passes_options(self, isolated_env):
        from lilbee.server.app import create_app
        from lilbee.server.handlers.sse import _resolve_generation_options

        with mock.patch(
            "lilbee.server.handlers.rag._resolve_generation_options",
            wraps=_resolve_generation_options,
        ) as spy:
            async with AsyncTestClient(create_app()) as client:
                # top_k 0 takes the deliberate pure-LLM path; with retrieval
                # on, an empty isolated library now answers EMPTY_LIBRARY
                # before options are ever resolved.
                resp = await client.post(
                    "/api/chat",
                    json={
                        "question": "test",
                        "history": [],
                        "top_k": 0,
                        "options": {"seed": 42},
                    },
                    headers=_auth_headers(),
                )
        assert resp.status_code == 201
        assert "answer" in resp.json()
        spy.assert_any_call({"seed": 42})

    async def test_ask_without_options(self, isolated_env):
        """Request without options field still works."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/ask", json={"question": "test"}, headers=_auth_headers())
        assert resp.status_code == 201


class TestCreateApp:
    def test_app_has_add_route(self):
        """The Litestar app registers the /api/add route."""
        from lilbee.server.app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/api/add" in paths


class TestAddIngestMutex:
    """Tests for the per-source ingest mutex gating ``/api/add``."""

    async def _collect(self, gen):
        """Drain an async generator of SSE strings into (event, data) pairs."""
        text = ""
        async for frame in gen:
            text += frame
        return _parse_sse_events(text.encode())

    async def test_try_acquire_rejects_while_held(self, isolated_env):
        """A second ``_try_acquire_source`` for a held name returns ``None``."""
        from lilbee.app.services import get_services

        first = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert first is not None
        try:
            second = await get_services().ingest_lock_registry.try_acquire("doc.txt")
            assert second is None
        finally:
            first.release()

    async def test_try_acquire_succeeds_after_release(self, isolated_env):
        """Once released, the same name can be acquired again."""
        from lilbee.app.services import get_services

        first = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert first is not None
        first.release()
        second = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert second is not None
        second.release()

    async def test_try_acquire_distinct_names_run_parallel(self, isolated_env):
        """Locks for different source names are independent."""
        from lilbee.app.services import get_services

        a = await get_services().ingest_lock_registry.try_acquire("a.txt")
        b = await get_services().ingest_lock_registry.try_acquire("b.txt")
        assert a is not None
        assert b is not None
        a.release()
        b.release()

    async def test_canonical_name_matches_basename(self, isolated_env):
        """Canonical source names match the label a registered root keys under."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        assert IngestLockRegistry.canonical_source_name("/some/path/doc.txt") == "doc.txt"
        assert IngestLockRegistry.canonical_source_name("doc.txt") == "doc.txt"

    async def test_release_evicts_entry_so_registry_does_not_grow(self, isolated_env):
        """A long-lived daemon must not keep one lock per filename forever."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        registry = IngestLockRegistry()
        for i in range(50):
            acquired, busy = await registry.acquire([f"doc{i}.txt"])
            assert not busy
            registry.release(acquired)
        # Every name was released, so no entry should linger.
        assert registry._locks == {}

    async def test_release_keeps_entry_for_still_held_name(self, isolated_env):
        """Releasing one batch must not evict a name another batch still holds."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        registry = IngestLockRegistry()
        held, _ = await registry.acquire(["doc.txt"])
        held_lock = held[0][1]
        # A second acquire for the same name is rejected (busy), nothing to evict.
        also, busy = await registry.acquire(["doc.txt"])
        assert also == [] and busy == ["doc.txt"]
        # Releasing a DIFFERENT batch (other.txt) must not evict doc.txt, and must
        # not disturb doc.txt's lock identity.
        other, _ = await registry.acquire(["other.txt"])
        registry.release(other)
        assert "other.txt" not in registry._locks  # the released name is evicted
        assert registry._locks.get("doc.txt") is held_lock  # held entry untouched
        registry.release(held)
        assert registry._locks == {}

    async def test_acquire_dedups_repeated_names(self, isolated_env):
        """The registry locks each distinct name once."""
        from lilbee.app.services import get_services

        registry = get_services().ingest_lock_registry
        acquired, busy = await registry.acquire(["doc.txt", "doc.txt"])
        try:
            assert busy == []
            assert [name for name, _ in acquired] == ["doc.txt"]
        finally:
            registry.release(acquired)

    def test_add_locks_server_paths_by_basename(self):
        """/api/add flattens into documents_dir, so two paths sharing a
        basename are one source and must share one lock."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        assert IngestLockRegistry.canonical_source_name("/x/doc.txt") == "doc.txt"
        assert IngestLockRegistry.canonical_source_name("/y/doc.txt") == "doc.txt"

    async def test_second_concurrent_add_emits_already_ingesting(self, isolated_env, tmp_path):
        """Second /api/add for a held source yields already_ingesting, no done."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "holdme.txt"
        src.write_text("payload")

        lock = await get_services().ingest_lock_registry.try_acquire("holdme.txt")
        assert lock is not None
        try:
            events = await self._collect(add_files_stream([str(src)]))
        finally:
            lock.release()

        event_types = [e[0] for e in events]
        assert event_types == ["already_ingesting"]
        assert events[0][1] == {"source": "holdme.txt"}

    async def test_partial_contention_partitions_paths(self, isolated_env, tmp_path):
        """Held paths emit already_ingesting; free paths still run to done."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        held = tmp_path / "held.txt"
        held.write_text("held payload")
        free = tmp_path / "free.txt"
        free.write_text("free payload")

        lock = await get_services().ingest_lock_registry.try_acquire("held.txt")
        assert lock is not None
        try:
            with mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                return_value=_make_xberg_result(),
            ):
                events = await self._collect(add_files_stream([str(held), str(free)]))
        finally:
            lock.release()

        event_types = [e[0] for e in events]
        already = [d for t, d in events if t == "already_ingesting"]
        assert {"source": "held.txt"} in already
        assert "done" in event_types
        # The contended path must not be ingested under the holder's lock: only
        # the free path is copied (regression guard for the lock-bypass bug).
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "free.txt" in summary["copied"]
        assert "held.txt" not in summary["copied"]
        # The done event is the only frame a client is guaranteed to still have,
        # so it has to say the batch was partial. Without this a caller reads
        # done as "all ingested" and never retries the contended file.
        assert summary["already_ingesting"] == ["held.txt"]
        assert "held.txt" not in summary["name_taken"]
        assert "held.txt" not in summary["overlapping"]

    async def test_distinct_relative_paths_get_distinct_locks(self, isolated_env):
        """Two uploads that land at different paths must not share one lock.

        The registry reduced every key to its basename. That is right for
        /api/add, where register_sources keys a root by its basename, but
        uploads keep their relative layout, so src/util.py and tests/util.py
        are different files that were being serialized against each other.
        """
        from lilbee.app.services import get_services

        registry = get_services().ingest_lock_registry
        acquired, busy = await registry.acquire(["src/util.py", "tests/util.py"])
        try:
            assert busy == []
            assert sorted(name for name, _lock in acquired) == ["src/util.py", "tests/util.py"]
        finally:
            registry.release(acquired)

    async def test_concurrent_different_sources_run_in_parallel(self, isolated_env, tmp_path):
        """Disjoint sources do not contend: both requests complete with done."""
        from lilbee.server.handlers import add_files_stream

        a = tmp_path / "a.txt"
        a.write_text("alpha")
        b = tmp_path / "b.txt"
        b.write_text("beta")

        async def _run(path: Path):
            text = ""
            with mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                return_value=_make_xberg_result(),
            ):
                async for frame in add_files_stream([str(path)]):
                    text += frame
            return _parse_sse_events(text.encode())

        events_a, events_b = await asyncio.gather(_run(a), _run(b))
        assert "done" in [t for t, _ in events_a]
        assert "done" in [t for t, _ in events_b]
        assert "already_ingesting" not in [t for t, _ in events_a]
        assert "already_ingesting" not in [t for t, _ in events_b]

    async def test_mutex_released_on_exception(self, isolated_env, tmp_path):
        """If ingest raises, the source mutex is released so retries succeed."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "boom.txt"
        src.write_text("contents")

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("ingest exploded")

        with mock.patch("lilbee.data.ingest.sync", new=_boom):
            events = await self._collect(add_files_stream([str(src)]))

        assert any(t == "error" for t, _ in events)

        retry = await get_services().ingest_lock_registry.try_acquire("boom.txt")
        assert retry is not None
        retry.release()

    async def test_a_cancelled_add_keeps_the_source_for_the_next_sync(self, isolated_env, tmp_path):
        """A REST client has no cancel of its own, so a stopped add keeps what it registered."""
        from lilbee.core.config import cfg
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        registered_during_sync: list[dict[str, str]] = []

        async def _cancelled(*_args, **_kwargs):
            registered_during_sync.append(dict(cfg.linked_roots))
            raise asyncio.CancelledError

        with (
            mock.patch("lilbee.data.ingest.sync", new=_cancelled),
            contextlib.suppress(asyncio.CancelledError),
        ):
            await self._collect(add_files_stream([str(src)]))

        assert registered_during_sync == [{"scan.txt": str(src.resolve())}]
        assert cfg.linked_roots == {"scan.txt": str(src.resolve())}

    async def test_mutex_released_on_task_cancellation(self, isolated_env, tmp_path):
        """Task cancellation during ingest still releases the source mutex."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "slow.txt"
        src.write_text("contents")

        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(30)

        gen = add_files_stream([str(src)])
        with mock.patch("lilbee.data.ingest.sync", new=_hang):
            outer = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.05)
            outer.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await outer
            await gen.aclose()

        retry = await get_services().ingest_lock_registry.try_acquire("slow.txt")
        assert retry is not None
        retry.release()


class _ServedApp:
    """The real app served by uvicorn on a loopback port, in a background thread."""

    def __init__(self) -> None:
        from lilbee.server.app import create_app

        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._socket.getsockname()[1]}"
        self._server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._socket]}, daemon=True
        )

    def __enter__(self) -> str:
        self._thread.start()
        assert poll_until(lambda: self._server.started), "the server did not start"
        return self.url

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(10)
        self._socket.close()


def _first_event(lines: Iterator[str]) -> str:
    """The name of the first SSE event in *lines*."""
    return next(line.split(":", 1)[1].strip() for line in lines if line.startswith("event:"))


class TestAddDisconnect:
    """A client that closes the /api/add stream over real HTTP, as the Obsidian plugin does."""

    @pytest.fixture()
    def parked_sync(self):
        """A sync that reports its first file, then waits to be cancelled; counts its starts."""
        from lilbee.runtime.progress import EventType, FileStartEvent

        started: list[None] = []

        async def _parked(*_args, on_progress, **_kwargs):
            started.append(None)
            on_progress(
                EventType.FILE_START, FileStartEvent(file="x", current_file=1, total_files=1)
            )
            await asyncio.sleep(30)

        with mock.patch("lilbee.data.ingest.sync", new=_parked):
            yield started

    @pytest.fixture()
    def released(self):
        """The source names whose ingest lock was released, which happens after the run unwinds."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        names: list[str] = []
        release = IngestLockRegistry.release

        def _recording(registry, acquired):
            held = [name for name, _lock in acquired]
            release(registry, acquired)
            names.extend(held)

        with mock.patch.object(IngestLockRegistry, "release", _recording):
            yield names

    def test_a_dropped_stream_keeps_the_source(self, tmp_path, parked_sync, released):
        """The plugin's idle abort, or a dropped socket, is not a user cancel."""
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        body = {"paths": [str(src)]}

        with _ServedApp() as url, httpx.Client(base_url=url, timeout=30) as http:
            with http.stream("POST", "/api/add", json=body, headers=_auth_headers()) as resp:
                assert _first_event(resp.iter_lines()) == "file_start"
            assert poll_until(lambda: "scan.txt" in released), "the add never unwound"

        assert cfg.linked_roots == {"scan.txt": str(src.resolve())}

    def test_a_return_on_already_ingesting_keeps_the_sources_it_started(
        self, tmp_path, parked_sync, released
    ):
        """The plugin stops reading at already_ingesting; the sources that did start stay."""
        busy = tmp_path / "busy.txt"
        busy.write_text("busy", encoding="utf-8")
        new = tmp_path / "new.txt"
        new.write_text("new", encoding="utf-8")
        both = {"paths": [str(busy), str(new)]}

        with _ServedApp() as url, httpx.Client(base_url=url, timeout=30) as http:
            holder_body = {"paths": [str(busy)]}
            with http.stream("POST", "/api/add", json=holder_body, headers=_auth_headers()) as held:
                # A collected line iterator closes its response, which would free busy.txt.
                held_lines = held.iter_lines()
                assert _first_event(held_lines) == "file_start"
                with http.stream("POST", "/api/add", json=both, headers=_auth_headers()) as resp:
                    assert _first_event(resp.iter_lines()) == "already_ingesting"
                    assert poll_until(lambda: len(parked_sync) == 2), "new.txt never started"
                assert poll_until(lambda: "new.txt" in released), "the add never unwound"

        assert cfg.linked_roots["new.txt"] == str(new.resolve())


class TestAddRegistersOffTheLoop:
    """The registration of /api/add runs in a thread and no cancel leaves it half observed."""

    @pytest.fixture()
    def held_registration(self):
        """A registration that waits in its thread until released; records what happens."""
        from lilbee.app.ingest import RegisterResult

        entered, release = threading.Event(), threading.Event()
        events: list[str] = []

        def _held(paths, *, force):
            events.append(f"registering on {threading.current_thread().name}")
            entered.set()
            assert release.wait(20), "the loop never ran while the registration was held"
            events.append("registered")
            return RegisterResult(registered=[path.name for path in paths])

        with mock.patch("lilbee.server.handlers.ingest.register_sources", _held):
            yield entered, release, events

    @staticmethod
    async def _until(event: threading.Event) -> None:
        async with asyncio.timeout(20):
            while not event.is_set():
                await asyncio.sleep(0.01)

    async def test_the_loop_runs_while_the_registration_is_held(self, tmp_path, held_registration):
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        entered, release, events = held_registration
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        synced = mock.AsyncMock(return_value=mock.Mock(model_dump=lambda: {}))

        with mock.patch("lilbee.data.ingest.sync", synced):
            task = asyncio.create_task(_run_add([str(src)], False, None, None, SseStream()))
            await self._until(entered)
            assert not task.done()
            release.set()
            summary = await task

        assert events[0] != f"registering on {threading.main_thread().name}"
        assert summary.copied == ["scan.txt"] and synced.await_count == 1

    async def test_a_cancel_during_the_registration_lands_when_the_sync_starts(
        self, tmp_path, held_registration
    ):
        """What a disconnect does: the stream's flag and a task cancel, both mid-registration."""
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        entered, release, events = held_registration
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        sse = SseStream()

        async def _sync(*_args, **_kwargs):
            events.append("sync started")
            await asyncio.sleep(0)
            events.append("sync ran on")

        with mock.patch("lilbee.data.ingest.sync", _sync):
            task = asyncio.create_task(_run_add([str(src)], False, None, None, sse))
            await self._until(entered)
            sse.cancel.set()
            task.cancel()
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "the cancel did not wait for the registration"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert events[1:] == ["registered", "sync started"]

    async def test_a_closed_stream_keeps_the_source_locked_until_it_is_registered(
        self, tmp_path, held_registration
    ):
        from lilbee.runtime.ingest_lock import IngestLockRegistry
        from lilbee.server.handlers import add_files_stream

        entered, release, events = held_registration
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        real_release = IngestLockRegistry.release

        def _recording(registry, acquired):
            events.append("unlocked")
            real_release(registry, acquired)

        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(30)

        gen = add_files_stream([str(src)])
        with (
            mock.patch.object(IngestLockRegistry, "release", _recording),
            mock.patch("lilbee.data.ingest.sync", new=_hang),
        ):
            reader = asyncio.create_task(gen.__anext__())
            await self._until(entered)
            reader.cancel()
            await asyncio.sleep(0.05)
            assert "unlocked" not in events and not reader.done()
            release.set()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await reader
            await gen.aclose()

        assert events[1:] == ["registered", "unlocked"]

    async def test_a_registration_that_raises_ends_the_add_with_its_error(self, tmp_path):
        from lilbee.runtime.lock import SyncRunningError
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        with (
            mock.patch(
                "lilbee.server.handlers.ingest.register_sources",
                side_effect=SyncRunningError(
                    "A sync or a wiki build is running. Add notes again when it ends."
                ),
            ),
            pytest.raises(SyncRunningError, match="Add notes again when it ends"),
        ):
            await _run_add([str(tmp_path)], False, None, None, SseStream())


class TestAddIngestHardening:
    """Option-A hardening: ``sync()`` always passes ``needs_cleanup=True``."""

    @staticmethod
    def _cleanup_sources(store) -> list[str]:
        """Sources whose batched write carried a cleanup delete."""
        sources: list[str] = []
        for call in store.write_chunks_batch.call_args_list:
            for item in call.args[0]:
                if item.needs_cleanup:
                    sources.append(item.source)
        return sources

    async def test_new_file_triggers_cleanup(self, isolated_env, tmp_path, mock_svc):
        """New files still carry a cleanup delete in their batched write: closes
        the orphaned-chunks race when a prior ingest died before upsert_source."""
        from lilbee.data.ingest import sync

        src = isolated_env / "fresh.txt"
        src.write_text("Fresh content.")

        store = mock_svc.store
        # No prior sources: existing_sources lookup returns nothing.
        store.get_sources.return_value = []

        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            await sync(quiet=True)

        # Option-A hardening: cleanup is requested even for 'new' files.
        assert "fresh.txt" in self._cleanup_sources(store)

    async def test_retry_after_orphaned_chunks_cleans_up(self, isolated_env, tmp_path, mock_svc):
        """If a prior run left chunks without an ``upsert_source`` record,
        the retry path removes them in the same transaction as the re-add.
        """
        from lilbee.data.ingest import sync

        src = isolated_env / "orphan.txt"
        src.write_text("Recovered content.")

        store = mock_svc.store
        # No source row, yet chunks exist on disk (the crashed-previous-run
        # scenario). The batched write's cleanup delete is idempotent.
        store.get_sources.return_value = []

        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            await sync(quiet=True)

        # The stale chunks are removed in the same batched write that re-adds them.
        assert "orphan.txt" in self._cleanup_sources(store)
        store.write_chunks_batch.assert_called()


class TestValidateAddPathsRejectsNamelessPaths:
    def test_a_path_with_no_final_component_is_rejected(self):
        """Path(x).name never contains separators, so the traversal check alone
        could not fail. What it must catch is an empty name, which would make
        documents_dir itself the copy destination."""
        from lilbee.server.handlers.ingest import validate_add_paths

        with pytest.raises(ValueError, match="does not name a file"):
            validate_add_paths({"paths": ["/"]})

    def test_a_normal_path_still_passes(self):
        from lilbee.server.handlers.ingest import validate_add_paths

        paths, _force, _ocr, _timeout = validate_add_paths({"paths": ["/tmp/report.pdf"]})
        assert paths == ["/tmp/report.pdf"]
