"""Tests for wiki REST API route stubs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from litestar.testing import AsyncTestClient

from lilbee.core.config import cfg
from lilbee.core.security import PathTraversalError
from lilbee.runtime.progress import EventType, WikiPageEvent, WikiPhase, WikiPhaseEvent
from lilbee.server import auth as _auth_mod
from lilbee.wiki.shared import PENDING_MARKER_KEYWORD_PARSE


def _h() -> dict[str, str]:
    """Auth headers."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _sse_events(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into (event name, decoded data) pairs."""
    events: list[tuple[str, Any]] = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        if len(lines) != 2:
            continue
        events.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return events


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Redirect config paths to temp dir and disable wiki by default."""
    snapshot = cfg.model_copy()
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.wiki = False
    cfg.wiki_dir = "wiki"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _create_app():
    from lilbee.server.app import create_app

    return create_app()


def _make_wiki_page(wiki_root: Path, subdir: str, name: str, content: str = "") -> Path:
    """Create a wiki markdown file in the given subdirectory."""
    page_dir = wiki_root / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{name}.md"
    if not content:
        content = (
            "---\n"
            "title: Test Page\n"
            "generated_at: 2026-04-04T12:00:00Z\n"
            "sources: [documents/test.txt]\n"
            "faithfulness_score: 0.9\n"
            "---\n"
            "# Test Page\n\nSome content.\n"
        )
    page_path.write_text(content, encoding="utf-8")
    return page_path


class TestWikiDisabled:
    """All wiki endpoints return 404 when wiki is off."""

    async def test_list_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 404

    async def test_read_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/test", headers=_h())
        assert resp.status_code == 404

    async def test_drafts_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts", headers=_h())
        assert resp.status_code == 404

    async def test_drafts_diff_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts/diff/anything", headers=_h())
        assert resp.status_code == 404

    async def test_drafts_accept_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/anything", headers=_h())
        assert resp.status_code == 404

    async def test_drafts_reject_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.delete("/api/wiki/drafts/anything", headers=_h())
        assert resp.status_code == 404

    async def test_citations_reverse_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/citations", params={"source": "x"}, headers=_h())
        assert resp.status_code == 404

    async def test_lint_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/lint", headers=_h())
        assert resp.status_code == 404

    async def test_lint_status_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/lint/abc123", headers=_h())
        assert resp.status_code == 404

    async def test_prune_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 404

    async def test_build_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build", headers=_h())
        assert resp.status_code == 404

    async def test_update_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.patch("/api/wiki/update", headers=_h())
        assert resp.status_code == 404

    async def test_synthesize_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/synthesize", headers=_h())
        assert resp.status_code == 404

    async def test_status_returns_disabled_marker(self):
        # Status is intentionally readable when wiki is off so the plugin
        # can render a disabled-state hint without polling /api/config.
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        assert resp.json()["wiki_enabled"] is False


class TestWikiEnabled:
    """Wiki endpoints with wiki=True."""

    @pytest.fixture(autouse=True)
    def enable_wiki(self):
        cfg.wiki = True

    async def test_list_empty(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_with_pages(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "test-doc")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        pages = resp.json()
        assert len(pages) == 1
        assert pages[0]["slug"] == "summaries/test-doc"
        assert pages[0]["title"] == "Test Page"
        assert pages[0]["page_type"] == "summary"
        assert pages[0]["source_count"] == 1
        assert pages[0]["created_at"] == "2026-04-04T12:00:00+00:00"

    async def test_status_counts_all_content_subdirs(self, isolated_env: Path):
        """pages counts every content subdir (summary + 2 concepts here), not just
        summaries+drafts; the pending draft is not counted."""
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "s1")
        _make_wiki_page(wiki_root, "concepts", "c1")
        _make_wiki_page(wiki_root, "concepts", "c2")
        _make_wiki_page(wiki_root, "drafts", "d1")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["pages"] == 3
        assert body["drafts"] == 1

    async def test_list_multiple_subdirs(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "doc-a")
        _make_wiki_page(wiki_root, "synthesis", "typing")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        pages = resp.json()
        assert len(pages) == 2
        slugs = {p["slug"] for p in pages}
        assert "summaries/doc-a" in slugs
        assert "synthesis/typing" in slugs

    async def test_list_string_sources(self, isolated_env: Path):
        """Pages with sources as a comma-separated string still count correctly."""
        wiki_root = isolated_env / "wiki"
        content = (
            '---\ntitle: StringSrc\nsources: "a.md, b.md"\n'
            "faithfulness_score: 0.9\n---\n# StringSrc\n"
        )
        _make_wiki_page(wiki_root, "summaries", "string-src", content=content)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        pages = resp.json()
        assert len(pages) == 1
        assert pages[0]["source_count"] == 2

    async def test_read_existing_page(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "my-doc")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/my-doc", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "summaries/my-doc"
        assert body["title"] == "Test Page"
        assert "# Test Page" in body["content"]

    async def test_read_missing_page_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/nope", headers=_h())
        assert resp.status_code == 404

    async def test_page_citations(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services

        mock_svc = make_mock_services()
        mock_svc.store.get_citations_for_wiki.return_value = []
        monkeypatch.setattr("lilbee.app.services.get_services", lambda: mock_svc)
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "cited")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/cited/citations", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "summaries/cited"
        assert body["citations"] == []

    async def test_page_citations_missing_page(self, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/nope/citations", headers=_h())
        assert resp.status_code == 404

    async def test_citations_reverse_empty(self, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services

        mock_svc = make_mock_services()
        mock_svc.store.get_citations_for_source.return_value = []
        monkeypatch.setattr("lilbee.app.services.get_services", lambda: mock_svc)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get(
                "/api/wiki/citations", params={"source": "test.txt"}, headers=_h()
            )
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_citations_reverse_no_source(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/citations", headers=_h())
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_drafts_empty(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts", headers=_h())
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_drafts_with_pages(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "drafts", "failed-page")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts", headers=_h())
        assert resp.status_code == 200
        drafts = resp.json()
        assert len(drafts) == 1
        # The DraftInfo slug is relative to the drafts subdir, not the
        # wiki root: matches the CLI / service-layer contract.
        assert drafts[0]["slug"] == "failed-page"
        assert drafts[0]["pending_kind"] is None
        assert drafts[0]["faithfulness_score"] == 0.9
        assert drafts[0]["published_exists"] is False

    async def test_lint_returns_report(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services
        from lilbee.wiki import lint as lint_mod

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        monkeypatch.setattr(
            lint_mod,
            "lint_all",
            lambda store, config=None: lint_mod.LintReport(),
        )
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/lint", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert body["errors"] == 0
        assert body["warnings"] == 0
        assert body["issues"] == []

    async def test_lint_runs_off_the_event_loop(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """lint_all scans every page and embeds, so it runs in a worker thread, not
        on the event loop that serves every other REST/MCP request (bb-ziks.44)."""
        import threading

        from conftest import make_mock_services
        from lilbee.wiki import lint as lint_mod

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        loop_tid = threading.get_ident()
        seen: list[int] = []

        def fake_lint(store, config=None):
            seen.append(threading.get_ident())
            return lint_mod.LintReport()

        monkeypatch.setattr(lint_mod, "lint_all", fake_lint)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/lint", headers=_h())
        assert resp.status_code == 201
        assert seen and seen[0] != loop_tid

    async def test_lint_status_route_removed(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/lint/task-abc", headers=_h())
        assert resp.status_code == 404

    async def test_prune_returns_report(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert "records" in body
        assert body["archived"] == 0
        assert body["flagged"] == 0
        assert body["reconciled"] == 0

    async def test_prune_reports_reconciled_rows(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.wiki.prune import PruneAction, PruneRecord, PruneReport

        report = PruneReport(
            records=[
                PruneRecord(
                    wiki_source="wiki/concepts/gone.md",
                    action=PruneAction.RECONCILED,
                    reason="indexed rows without a page on disk",
                )
            ]
        )
        monkeypatch.setattr("lilbee.server.wiki.prune_mod.prune_wiki", lambda store: report)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert body["reconciled"] == 1
        assert body["archived"] == 0
        assert body["records"][0]["action"] == "reconciled"

    async def test_prune_runs_off_the_event_loop(self, monkeypatch: pytest.MonkeyPatch):
        """prune_wiki walks the whole tree + store; it runs in a worker thread, not
        on the event loop (bb-ziks.44)."""
        import threading

        from conftest import make_mock_services
        from lilbee.wiki import prune as prune_mod

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        loop_tid = threading.get_ident()
        seen: list[int] = []

        def fake_prune(store):
            seen.append(threading.get_ident())
            return prune_mod.PruneReport()

        monkeypatch.setattr(prune_mod, "prune_wiki", fake_prune)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 201
        assert seen and seen[0] != loop_tid

    async def test_status_runs_lint_off_the_event_loop(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The status route's lint pass embeds; it runs off the event loop (bb-ziks.44)."""
        import threading

        from conftest import make_mock_services
        from lilbee.wiki import lint as lint_mod

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        _make_wiki_page(isolated_env / "wiki", "summaries", "s1")  # so status reaches lint
        loop_tid = threading.get_ident()
        seen: list[int] = []

        def fake_lint(store, config=None, record_log=True):
            seen.append(threading.get_ident())
            return lint_mod.LintReport()

        monkeypatch.setattr(lint_mod, "lint_all", fake_lint)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        assert seen and seen[0] != loop_tid

    async def test_build_streams_progress_then_a_done_summary(self, monkeypatch):
        def fake_build(config, on_progress):
            on_progress(EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.EXTRACT))
            on_progress(
                EventType.WIKI_PAGE,
                WikiPageEvent(label="doc.txt", pages=1, current=1, total=1),
            )
            return {"paths": ["wiki/concepts/x.md"], "entities": 3, "count": 1}

        monkeypatch.setattr("lilbee.server.handlers.wiki.run_full_build", fake_build)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build", headers=_h())
        assert resp.status_code == 201
        events = _sse_events(resp.text)
        assert [name for name, _payload in events] == ["wiki_phase", "wiki_page", "done"]
        assert events[0][1]["phase"] == "extract"
        assert events[1][1]["label"] == "doc.txt"
        assert events[2][1] == {"paths": ["wiki/concepts/x.md"], "entities": 3, "count": 1}

    async def test_done_frame_carries_the_gate_stats(self, monkeypatch):
        """The terminal SSE frame is the summary dict, so stats reach the client."""
        from lilbee.wiki.stats import BuildStats

        stats = BuildStats()
        stats.record_published("wiki/concepts/x.md", 2)
        stats.record_drafted()
        stats.record_citations(rendered=2, dropped=1)
        monkeypatch.setattr(
            "lilbee.server.handlers.wiki.run_full_build",
            lambda config, on_progress: {
                "paths": ["wiki/concepts/x.md"],
                "entities": 3,
                "count": 1,
                "stats": stats.as_dict(),
            },
        )
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build", headers=_h())
        assert resp.status_code == 201
        name, payload = _sse_events(resp.text)[-1]
        assert name == "done"
        assert payload["stats"]["pages_published"] == 1
        assert payload["stats"]["pages_drafted"] == 1
        assert payload["stats"]["citation_verify_rate"] == 2 / 3

    async def test_build_dry_run_returns_candidates_as_json(self, monkeypatch):
        """dry_run answers with the entity candidates, not an SSE stream."""

        def build_boom(*a, **kw):
            raise AssertionError("dry_run must not build")

        monkeypatch.setattr("lilbee.server.handlers.wiki.run_full_build", build_boom)
        monkeypatch.setattr(
            "lilbee.wiki.generation.preview_build_entities",
            lambda config: [
                {
                    "slug": "ford",
                    "label": "Ford",
                    "kind": "entity",
                    "type_hint": "ORG",
                    "mentions": 2,
                    "sources": ["a.md"],
                }
            ],
        )
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build?dry_run=true", headers=_h())
        assert resp.status_code == 201
        payload = resp.json()
        assert payload["dry_run"] is True
        assert payload["count"] == 1
        assert payload["entities"][0]["slug"] == "ford"
        assert payload["note"]

    async def test_build_runs_off_the_event_loop(self, monkeypatch):
        """The build is CPU- and IO-bound, so it runs in a worker thread."""
        import threading

        loop_tid = threading.get_ident()
        seen: list[int] = []

        def fake_build(config, on_progress):
            seen.append(threading.get_ident())
            return {"paths": [], "entities": 0, "count": 0}

        monkeypatch.setattr("lilbee.server.handlers.wiki.run_full_build", fake_build)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build", headers=_h())
        assert resp.status_code == 201
        assert seen and seen[0] != loop_tid

    async def test_build_stream_reports_a_failed_run_as_an_error_event(self, monkeypatch):
        def fake_build(config, on_progress):
            raise RuntimeError("provider unreachable")

        monkeypatch.setattr("lilbee.server.handlers.wiki.run_full_build", fake_build)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/build", headers=_h())
        assert resp.status_code == 201
        events = _sse_events(resp.text)
        assert events == [("error", {"message": "provider unreachable"})]

    async def test_update_streams_the_same_build(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.server.handlers.wiki.run_full_build",
            lambda config, on_progress: {"paths": [], "entities": 0, "count": 0},
        )
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.patch("/api/wiki/update", headers=_h())
        assert resp.status_code == 200
        assert _sse_events(resp.text) == [("done", {"paths": [], "entities": 0, "count": 0})]

    async def test_synthesize_streams_progress_then_a_done_summary(self, monkeypatch):
        def fake_synthesize(config, on_progress):
            on_progress(
                EventType.WIKI_PAGE,
                WikiPageEvent(label="typing", pages=1, current=1, total=1),
            )
            return {"paths": ["wiki/synthesis/typing.md"], "count": 1}

        monkeypatch.setattr("lilbee.server.handlers.wiki.run_full_synthesize", fake_synthesize)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/synthesize", headers=_h())
        assert resp.status_code == 201
        events = _sse_events(resp.text)
        assert [name for name, _payload in events] == ["wiki_page", "done"]
        assert events[1][1] == {"paths": ["wiki/synthesis/typing.md"], "count": 1}

    async def test_synthesize_no_clusters_still_ends_with_done(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.server.handlers.wiki.run_full_synthesize",
            lambda config, on_progress: {"paths": [], "count": 0},
        )
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/synthesize", headers=_h())
        assert resp.status_code == 201
        assert _sse_events(resp.text) == [("done", {"paths": [], "count": 0})]

    async def test_status_empty_wiki(self, monkeypatch, tmp_path):
        from lilbee.wiki import lint as lint_mod

        def make_mock_services():
            services = type("S", (), {})()
            services.store = None
            return services

        monkeypatch.setattr("lilbee.app.services.get_services", make_mock_services)
        monkeypatch.setattr("lilbee.server.wiki.svc_mod.get_services", make_mock_services)
        monkeypatch.setattr(lint_mod, "lint_all", lambda *a, **kw: lint_mod.LintReport())
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["wiki_enabled"] is True
        assert body["pages"] == 0

    async def test_status_reports_drafts_but_excludes_them_from_pages(self, monkeypatch, tmp_path):
        from lilbee.wiki import lint as lint_mod

        wiki_root = cfg.data_root / cfg.wiki_dir
        _make_wiki_page(wiki_root, "summaries", "alpha")
        _make_wiki_page(wiki_root, "summaries", "beta")
        _make_wiki_page(wiki_root, "drafts", "gamma")

        def make_mock_services():
            services = type("S", (), {})()
            services.store = None
            return services

        monkeypatch.setattr("lilbee.server.wiki.svc_mod.get_services", make_mock_services)
        monkeypatch.setattr(lint_mod, "lint_all", lambda *a, **kw: lint_mod.LintReport())
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["summaries"] == 2
        assert body["drafts"] == 1
        # pages counts published content (the 2 summaries); pending drafts are not pages.
        assert body["pages"] == 2


def _make_draft(
    wiki_root: Path,
    slug: str,
    *,
    drift_pct: int | None = 42,
    pending_marker: str = "",
    faithfulness: float = 0.7,
) -> Path:
    """Write a draft markdown file with optional drift / pending markers."""
    draft_dir = wiki_root / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    path = draft_dir / f"{slug}.md"
    drift_line = ""
    if drift_pct is not None:
        drift_line = f"<!-- DRIFT: {drift_pct}% content changed - flagged for human review -->\n\n"
    body = (
        f"{drift_line}{pending_marker}"
        f"---\nfaithfulness_score: {faithfulness}\n---\n\n"
        "Draft body text.\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


class TestWikiDraftsEndpoints:
    """Draft review endpoints: list (upgraded), diff, accept, reject."""

    @pytest.fixture(autouse=True)
    def enable_wiki(self):
        cfg.wiki = True

    async def test_list_carries_pending_kind(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_draft(
            wiki_root,
            "parse-fail",
            drift_pct=None,
            pending_marker=(f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for section foo -->\n\n"),
        )
        _make_draft(wiki_root, "drifted", drift_pct=55, faithfulness=0.8)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts", headers=_h())
        assert resp.status_code == 200
        drafts = {d["slug"]: d for d in resp.json()}
        assert drafts["parse-fail"]["pending_kind"] == "parse"
        assert drafts["parse-fail"]["drift_ratio"] is None
        assert drafts["drifted"]["pending_kind"] is None
        assert drafts["drifted"]["drift_ratio"] == pytest.approx(0.55)
        assert drafts["drifted"]["faithfulness_score"] == pytest.approx(0.8)

    async def test_diff_happy(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_draft(wiki_root, "cv-manual", drift_pct=30, faithfulness=0.9)
        # Published counterpart to diff against.
        _make_wiki_page(wiki_root, "summaries", "cv-manual")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts/diff/cv-manual", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "cv-manual"
        assert "Draft body text" in body["diff"]

    async def test_diff_missing_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts/diff/nope", headers=_h())
        assert resp.status_code == 404
        assert "draft not found" in resp.json()["detail"]

    async def test_diff_traversal_slug_maps_to_generic_400(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod

        # The traversal guard raises ValueError carrying the absolute candidate
        # path; the route must map it to a generic 400 that does not leak it.
        def _raise(slug: str, root: Path) -> str:
            raise PathTraversalError(f"Path escapes allowed directory: {root}/secret.md")

        monkeypatch.setattr(server_wiki_mod, "diff_draft", _raise)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/drafts/diff/anything", headers=_h())
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid draft slug"

    async def test_accept_traversal_slug_maps_to_generic_400(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod

        def _raise(slug: str, root: Path, store: object) -> object:
            raise PathTraversalError(f"Path escapes allowed directory: {root}/secret.md")

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _raise)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/anything", headers=_h())
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid draft slug"

    async def test_accept_indexing_valueerror_is_not_masked(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod

        # A non-traversal ValueError from re-indexing (e.g. embedding-dim drift)
        # must NOT be mislabeled as a 400 "invalid draft slug"; it surfaces as 500.
        def _raise(slug: str, root: Path, store: object) -> object:
            raise ValueError("Vector dimension mismatch: expected 768, got 1024")

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _raise)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/real-slug", headers=_h())
        assert resp.status_code == 500
        assert resp.json().get("detail") != "invalid draft slug"

    async def test_reject_traversal_slug_maps_to_generic_400(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod

        def _raise(slug: str, root: Path) -> None:
            raise PathTraversalError(f"Path escapes allowed directory: {root}/secret.md")

        monkeypatch.setattr(server_wiki_mod, "reject_draft", _raise)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.delete("/api/wiki/drafts/anything", headers=_h())
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid draft slug"

    async def test_accept_happy(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.server import wiki as server_wiki_mod
        from lilbee.wiki.drafts import AcceptResult

        wiki_root = isolated_env / "wiki"
        _make_draft(wiki_root, "cv-manual", drift_pct=20)

        captured: dict[str, object] = {}

        def _fake_accept(slug: str, root: Path, store: object) -> AcceptResult:
            captured["slug"] = slug
            captured["root"] = root
            return AcceptResult(
                slug=slug,
                requested_slug=slug,
                moved_to=root / "summaries" / f"{slug}.md",
                reindexed_chunks=3,
            )

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _fake_accept)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/cv-manual", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert body["slug"] == "cv-manual"
        assert body["requested_slug"] == "cv-manual"
        assert body["reindexed_chunks"] == 3
        assert body["moved_to"].endswith("summaries/cv-manual.md")
        assert captured["slug"] == "cv-manual"

    async def test_accept_runs_off_the_event_loop(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """accept_draft re-chunks + embeds; it runs in a worker thread, not on the
        event loop that serves every other request (bb-ziks.44)."""
        import threading

        from lilbee.server import wiki as server_wiki_mod
        from lilbee.wiki.drafts import AcceptResult

        wiki_root = isolated_env / "wiki"
        _make_draft(wiki_root, "cv-manual", drift_pct=20)
        loop_tid = threading.get_ident()
        seen: list[int] = []

        def _fake_accept(slug: str, root: Path, store: object) -> AcceptResult:
            seen.append(threading.get_ident())
            return AcceptResult(
                slug=slug,
                requested_slug=slug,
                moved_to=root / "summaries" / f"{slug}.md",
                reindexed_chunks=1,
            )

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _fake_accept)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/cv-manual", headers=_h())
        assert resp.status_code == 201
        assert seen and seen[0] != loop_tid

    async def test_accept_stale_draft_returns_409(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod
        from lilbee.wiki.drafts import StaleDraftError

        def _fake_accept(slug: str, root: Path, store: object) -> None:
            raise StaleDraftError(f"published page for {slug} is newer than the draft")

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _fake_accept)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/cv-manual", headers=_h())
        assert resp.status_code == 409
        assert "newer than the draft" in resp.json()["detail"]

    async def test_accept_missing_404(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.server import wiki as server_wiki_mod

        def _fake_accept(slug: str, root: Path, store: object) -> None:
            raise FileNotFoundError(f"draft not found: {slug}")

        monkeypatch.setattr(server_wiki_mod, "accept_draft", _fake_accept)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/drafts/accept/nope", headers=_h())
        assert resp.status_code == 404
        assert "draft not found" in resp.json()["detail"]

    async def test_reject_happy(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        draft_path = _make_draft(wiki_root, "doomed", drift_pct=25)
        assert draft_path.is_file()
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.delete("/api/wiki/drafts/doomed", headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "doomed"
        assert not draft_path.exists()

    async def test_reject_missing_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.delete("/api/wiki/drafts/nope", headers=_h())
        assert resp.status_code == 404
        assert "draft not found" in resp.json()["detail"]


class TestFrontmatterParsing:
    def test_valid_frontmatter(self):
        from lilbee.wiki.shared import parse_frontmatter

        text = "---\ntitle: Hello\ngenerated_at: '2026-01-01'\n---\nBody"
        result = parse_frontmatter(text)
        assert result["title"] == "Hello"
        assert result["generated_at"] == "2026-01-01"

    def test_no_frontmatter(self):
        from lilbee.wiki.shared import parse_frontmatter

        assert parse_frontmatter("Just text") == {}

    def test_unclosed_frontmatter(self):
        from lilbee.wiki.shared import parse_frontmatter

        assert parse_frontmatter("---\ntitle: Hello\nNo close") == {}

    def test_multiple_sources(self):
        from lilbee.wiki.shared import parse_frontmatter

        text = "---\nsources: [a.txt, b.txt, c.txt]\n---\n"
        result = parse_frontmatter(text)
        assert result["sources"] == ["a.txt", "b.txt", "c.txt"]


class TestHelpers:
    def test_page_type_from_summaries(self, tmp_path: Path):
        from lilbee.wiki.browse import _page_type_from_path

        assert _page_type_from_path(tmp_path / "summaries" / "x.md", tmp_path) == "summary"

    def test_page_type_from_synthesis(self, tmp_path: Path):
        from lilbee.wiki.browse import _page_type_from_path

        assert _page_type_from_path(tmp_path / "synthesis" / "x.md", tmp_path) == "synthesis"

    def test_page_type_unknown(self, tmp_path: Path):
        from lilbee.wiki.browse import _page_type_from_path

        assert _page_type_from_path(tmp_path / "x.md", tmp_path) == "unknown"

    def test_page_type_unrelated_path(self, tmp_path: Path):
        from lilbee.wiki.browse import _page_type_from_path

        other = Path("/completely/different")
        assert _page_type_from_path(other / "x.md", tmp_path) == "unknown"

    def test_slug_from_path(self, tmp_path: Path):
        from lilbee.wiki.browse import _slug_from_path

        assert _slug_from_path(tmp_path / "summaries" / "doc.md", tmp_path) == "summaries/doc"

    def test_list_md_files_empty(self, tmp_path: Path):
        from lilbee.wiki.browse import list_md_files

        assert list_md_files(tmp_path / "nonexistent") == []

    def test_list_md_files_filters(self, tmp_path: Path):
        from lilbee.wiki.browse import list_md_files

        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        (tmp_path / "c.md").write_text("z")
        result = list_md_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)

    def test_build_page_info_no_frontmatter(self, tmp_path: Path):
        from lilbee.wiki.browse import build_page_info

        subdir = tmp_path / "summaries"
        subdir.mkdir()
        page = subdir / "my-doc.md"
        page.write_text("# Just a heading\n")
        info = build_page_info(page, tmp_path)
        # build_page_info prefers the first H1 when frontmatter has no title;
        # the filename stem is the last fallback.
        assert info.title == "Just a heading"
        assert info.source_count == 0
        assert info.created_at == ""

    def test_wiki_root(self, isolated_env: Path):
        from lilbee.server.wiki import _wiki_root

        assert _wiki_root() == isolated_env / "wiki"

    def test_find_page_exists(self, isolated_env: Path):
        from lilbee.server.wiki import _find_page

        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "found")
        result = _find_page("summaries/found")
        assert result is not None
        assert result.name == "found.md"

    def test_find_page_missing(self):
        from lilbee.server.wiki import _find_page

        assert _find_page("summaries/nope") is None

    def test_find_page_rejects_path_traversal(self):
        from lilbee.server.wiki import _find_page

        assert _find_page("../../etc/passwd") is None
        assert _find_page("summaries/../../../etc/passwd") is None


class TestPydanticModels:
    def test_wiki_citation_record_defaults(self):
        from lilbee.server.models import WikiCitationRecord

        c = WikiCitationRecord(
            wiki_source="wiki/summaries/doc.md",
            citation_key="src1",
            source_filename="doc.txt",
        )
        assert c.page_start == 0
        assert c.excerpt == ""
        assert c.wiki_chunk_index == 0
        assert c.created_at == ""


class TestWikiReadPathsDoNotWrite:
    """A route reachable without a token must not mutate the wiki tree."""

    @pytest.fixture(autouse=True)
    def enable_wiki(self):
        cfg.wiki = True

    async def test_listing_pages_does_not_rewrite_the_index(self, isolated_env: Path):
        """A read must not write. GET /api/wiki regenerated index.md, so a
        listing raced the build path over the same file; at the time the route
        was also unauthenticated, which let any caller drive those writes."""
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "test-doc")
        index_path = wiki_root / "index.md"
        index_path.write_text("stale index\n", encoding="utf-8")
        before = index_path.stat().st_mtime_ns

        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())

        assert resp.status_code == 200
        assert [p["slug"] for p in resp.json()] == ["summaries/test-doc"]
        assert index_path.read_text(encoding="utf-8") == "stale index\n"
        assert index_path.stat().st_mtime_ns == before

    async def test_status_requires_a_token(self):
        """The full-corpus lint behind status must not be an anonymous amplifier."""
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/status")
        assert resp.status_code == 401

    async def test_status_does_not_append_to_the_wiki_log(self, isolated_env: Path):
        """lint_all grew record_log=False for exactly this read-only caller."""
        from unittest import mock

        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "s1")
        with mock.patch("lilbee.server.wiki.lint_mod.lint_all") as lint_all:
            lint_all.return_value = mock.MagicMock(error_count=0, warning_count=0)
            async with AsyncTestClient(_create_app()) as client:
                resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        assert lint_all.call_args.kwargs["record_log"] is False


class TestWikiDisabledStatusIsCheap:
    async def test_status_skips_the_lint_when_the_wiki_is_disabled(self, isolated_env: Path):
        """A leftover wiki dir from a previous build used to make the disabled
        status poll run the whole lint anyway."""
        from unittest import mock

        cfg.wiki = False
        _make_wiki_page(isolated_env / "wiki", "summaries", "leftover")
        with mock.patch("lilbee.server.wiki.lint_mod.lint_all") as lint_all:
            async with AsyncTestClient(_create_app()) as client:
                resp = await client.get("/api/wiki/status", headers=_h())
        assert resp.status_code == 200
        assert resp.json()["wiki_enabled"] is False
        lint_all.assert_not_called()
