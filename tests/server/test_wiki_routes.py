"""Tests for wiki REST API route stubs."""

from __future__ import annotations

from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from lilbee.config import cfg
from lilbee.server import auth as _auth_mod


def _h() -> dict[str, str]:
    """Auth headers."""
    return {"Authorization": f"Bearer {_auth_mod._session_token}"}


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

    async def test_generate_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/generate/test.txt", headers=_h())
        assert resp.status_code == 404

    async def test_prune_returns_404(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 404


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

    async def test_list_multiple_subdirs(self, isolated_env: Path):
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "doc-a")
        _make_wiki_page(wiki_root, "concepts", "typing")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        pages = resp.json()
        assert len(pages) == 2
        slugs = {p["slug"] for p in pages}
        assert "summaries/doc-a" in slugs
        assert "concepts/typing" in slugs

    async def test_list_regenerates_index(self, isolated_env: Path):
        """When wiki/index.md exists, listing regenerates it."""
        wiki_root = isolated_env / "wiki"
        _make_wiki_page(wiki_root, "summaries", "test-doc")
        # Create an index.md so the regeneration path triggers
        (wiki_root / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki", headers=_h())
        assert resp.status_code == 200
        # index.md should be refreshed with actual entries
        index_text = (wiki_root / "index.md").read_text(encoding="utf-8")
        assert "test-doc" in index_text

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
        from lilbee import services as svc_mod

        mock_svc = make_mock_services()
        mock_svc.store.get_citations_for_wiki.return_value = []
        monkeypatch.setattr(svc_mod, "get_services", lambda: mock_svc)
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
        from lilbee import services as svc_mod

        monkeypatch.setattr(svc_mod, "get_services", make_mock_services)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/summaries/nope/citations", headers=_h())
        assert resp.status_code == 404

    async def test_citations_reverse_empty(self, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services
        from lilbee import services as svc_mod

        mock_svc = make_mock_services()
        mock_svc.store.get_citations_for_source.return_value = []
        monkeypatch.setattr(svc_mod, "get_services", lambda: mock_svc)
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
        assert drafts[0]["slug"] == "drafts/failed-page"

    async def test_lint_returns_report(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services
        from lilbee import services as svc_mod
        from lilbee.wiki import lint as lint_mod

        monkeypatch.setattr(svc_mod, "get_services", make_mock_services)
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

    async def test_lint_status_returns_501(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.get("/api/wiki/lint/task-abc", headers=_h())
        assert resp.status_code == 501

    async def test_generate_returns_result(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from conftest import make_mock_services
        from lilbee import services as svc_mod
        from lilbee.wiki import gen as gen_mod

        monkeypatch.setattr(svc_mod, "get_services", make_mock_services)
        monkeypatch.setattr(gen_mod, "generate_summary_page", lambda *a, **kw: None)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/generate/test.txt", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert body["source"] == "test.txt"
        assert body["status"] == "failed"

    async def test_generate_success(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch):
        from conftest import make_mock_services
        from lilbee import services as svc_mod
        from lilbee.wiki import gen as gen_mod

        monkeypatch.setattr(svc_mod, "get_services", make_mock_services)
        page_path = isolated_env / "wiki" / "summaries" / "test.md"
        monkeypatch.setattr(gen_mod, "generate_summary_page", lambda *a, **kw: page_path)
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/generate/test.txt", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "generated"
        assert body["source"] == "test.txt"
        assert "path" in body

    async def test_generate_path_traversal_blocked(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/generate/../../etc/passwd", headers=_h())
        assert resp.status_code == 404

    async def test_prune_returns_report(self):
        async with AsyncTestClient(_create_app()) as client:
            resp = await client.post("/api/wiki/prune", headers=_h())
        assert resp.status_code == 201
        body = resp.json()
        assert "records" in body
        assert body["archived"] == 0
        assert body["flagged"] == 0


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

    def test_page_type_from_concepts(self, tmp_path: Path):
        from lilbee.wiki.browse import _page_type_from_path

        assert _page_type_from_path(tmp_path / "concepts" / "x.md", tmp_path) == "concept"

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
        from lilbee.wiki.browse import _list_md_files

        assert _list_md_files(tmp_path / "nonexistent") == []

    def test_list_md_files_filters(self, tmp_path: Path):
        from lilbee.wiki.browse import _list_md_files

        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        (tmp_path / "c.md").write_text("z")
        result = _list_md_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)

    def test_build_summary_no_frontmatter(self, tmp_path: Path):
        from lilbee.server.wiki import _build_summary

        subdir = tmp_path / "summaries"
        subdir.mkdir()
        page = subdir / "my-doc.md"
        page.write_text("# Just a heading\n")
        summary = _build_summary(page, tmp_path)
        assert summary["title"] == "My Doc"
        assert summary["source_count"] == 0
        assert summary["created_at"] == ""

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
    def test_wiki_page_summary_defaults(self):
        from lilbee.server.models import WikiPageSummary

        s = WikiPageSummary(slug="test", title="Test", page_type="summary")
        assert s.source_count == 0
        assert s.created_at == ""

    def test_wiki_citation_defaults(self):
        from lilbee.server.models import WikiCitation

        c = WikiCitation(citation_key="src1", claim_type="fact", source_filename="doc.txt")
        assert c.page_start == 0
        assert c.excerpt == ""

    def test_lint_issue(self):
        from lilbee.server.models import LintIssue

        issue = LintIssue(
            wiki_source="page.md",
            citation_key="src1",
            status="stale_hash",
            message="Source changed",
        )
        assert issue.status == "stale_hash"
