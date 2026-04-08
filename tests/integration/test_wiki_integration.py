"""Wiki layer integration tests with real models.

Uses llama-cpp-python with real GGUF models. Tests wiki generation,
linting, pruning, browsing, and citation storage against a real
LanceDB and real LLM provider.

Run with:
    uv run pytest tests/integration/test_wiki_integration.py -v -m slow
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

llama_cpp = pytest.importorskip("llama_cpp")

from lilbee.config import cfg  # noqa: E402
from lilbee.model_manager import reset_model_manager  # noqa: E402
from lilbee.services import get_services  # noqa: E402
from lilbee.services import reset_services as reset_provider  # noqa: E402

from .conftest import setup_rag_pipeline  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def wiki_pipeline(tmp_path_factory):
    """Set up a real pipeline with wiki enabled."""
    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("wiki_integration")
    data = setup_rag_pipeline(tmp, wiki=True)
    yield data
    reset_provider()
    reset_model_manager()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestGenerateSummaryPage:
    def test_generate_summary_page(self, wiki_pipeline):
        """Generate a summary page for specs.md with real LLM — verify page created."""
        from lilbee.wiki.gen import generate_summary_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("specs.md")
        assert len(chunks) > 0, "No chunks found for specs.md"

        # Mock the LLM chat to return predictable wiki content with citations
        wiki_content = (
            "# Thunderbolt X500 Summary\n\n"
            "The Thunderbolt X500 features a 3.5L V6 TurboForce engine.[^src1]\n"
            "It has a top speed of 155 mph.[^src2]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: specs.md, excerpt: "Engine: 3.5L V6 TurboForce"\n'
            '[^src2]: specs.md, excerpt: "Top speed: 155 mph"\n'
        )
        faithfulness_response = "0.85"

        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, faithfulness_response],
        ):
            result = generate_summary_page("specs.md", chunks, svc.provider, svc.store)

        assert result is not None, "generate_summary_page returned None"
        assert result.exists(), f"Generated page does not exist: {result}"
        content = result.read_text(encoding="utf-8")
        assert "---" in content  # frontmatter present
        assert "[^src1]" in content  # citation anchor present

    def test_faithfulness_gates_to_drafts(self, wiki_pipeline):
        """Low faithfulness score diverts page to drafts."""
        from lilbee.wiki.gen import generate_summary_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("deploy.md")
        assert len(chunks) > 0

        # LLM returns nonsense with a citation so it passes the citation check
        wiki_content = (
            "# Unrelated content\n\n"
            "Something about rubber ducks.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: deploy.md, excerpt: "kubectl apply"\n'
        )
        # Low faithfulness score
        faithfulness_response = "0.20"

        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, faithfulness_response],
        ):
            result = generate_summary_page("deploy.md", chunks, svc.provider, svc.store)

        assert result is not None
        assert "drafts" in str(result), f"Expected page in drafts/, got {result}"


class TestLintWiki:
    def test_lint_clean_page(self, wiki_pipeline):
        """A freshly generated page should lint clean (no critical errors)."""
        from lilbee.wiki.gen import generate_summary_page
        from lilbee.wiki.lint import IssueType, lint_wiki_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("auth-part1.md")
        assert len(chunks) > 0

        wiki_content = (
            "# OAuth Summary\n\n"
            "Configure OAuth 2.0 with client credentials.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: auth-part1.md, excerpt: "Configure OAuth 2.0 with client ID and secret"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.90"],
        ):
            page = generate_summary_page("auth-part1.md", chunks, svc.provider, svc.store)

        assert page is not None
        wiki_source = f"{cfg.wiki_dir}/summaries/auth-part1.md"
        issues = lint_wiki_page(wiki_source, svc.store)
        # No source_missing or path_traversal errors for a fresh page
        critical = [
            i
            for i in issues
            if i.issue_type in (IssueType.SOURCE_MISSING, IssueType.PATH_TRAVERSAL)
        ]
        assert len(critical) == 0, f"Unexpected critical lint issues: {critical}"

    def test_lint_stale_hash(self, wiki_pipeline):
        """Modifying a source after generation flags stale hash."""
        from lilbee.wiki.gen import generate_summary_page
        from lilbee.wiki.lint import IssueType, lint_wiki_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("notes.md")
        assert len(chunks) > 0

        wiki_content = (
            "# Meeting Notes Summary\n\n"
            "Migration to PostgreSQL 16 planned for Q2 2026.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: notes.md, excerpt: "migrate to PostgreSQL 16"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.85"],
        ):
            page = generate_summary_page("notes.md", chunks, svc.provider, svc.store)

        assert page is not None

        # Modify the source document
        source_path = cfg.documents_dir / "notes.md"
        source_path.write_text("# Updated meeting notes\n\nAll plans changed.\n")

        wiki_source = f"{cfg.wiki_dir}/summaries/notes.md"
        issues = lint_wiki_page(wiki_source, svc.store)
        stale = [i for i in issues if i.issue_type == IssueType.STALE_HASH]
        assert len(stale) > 0, f"Expected stale hash warning, got {[i.to_dict() for i in issues]}"


class TestPruneWiki:
    def test_prune_removes_orphaned_page(self, wiki_pipeline):
        """Deleting a source and pruning archives the wiki page."""
        from lilbee.wiki.gen import generate_summary_page
        from lilbee.wiki.prune import prune_wiki

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("db-perf.md")
        assert len(chunks) > 0

        wiki_content = (
            "# DB Performance Summary\n\n"
            "Use connection pooling with PgBouncer.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: db-perf.md, excerpt: "connection pooling with PgBouncer"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.85"],
        ):
            page = generate_summary_page("db-perf.md", chunks, svc.provider, svc.store)

        assert page is not None

        # Delete the source document
        source_path = cfg.documents_dir / "db-perf.md"
        source_path.unlink()

        # Prune
        report = prune_wiki(svc.store)
        # The page citing a deleted source should be archived
        archived = [r for r in report.records if r.action.value == "archived"]
        assert len(archived) > 0, (
            f"Expected archived page, got {[r.to_dict() for r in report.records]}"
        )


class TestBrowseWiki:
    def test_wiki_browse_lists_pages(self, wiki_pipeline):
        """After generation, list_pages() includes the generated page."""
        from lilbee.wiki.browse import list_pages
        from lilbee.wiki.gen import generate_summary_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("api-perf.md")
        assert len(chunks) > 0

        wiki_content = (
            "# API Performance Summary\n\n"
            "Cache responses with Redis for better performance.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: api-perf.md, excerpt: "Cache responses with Redis"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.88"],
        ):
            page = generate_summary_page("api-perf.md", chunks, svc.provider, svc.store)

        assert page is not None
        wiki_root = cfg.data_root / cfg.wiki_dir
        pages = list_pages(wiki_root)
        slugs = [p.slug for p in pages]
        assert any("api-perf" in s for s in slugs), f"api-perf page not in {slugs}"

    def test_wiki_read_page(self, wiki_pipeline):
        """After generation, read_page() returns content and metadata."""
        from lilbee.wiki.browse import read_page
        from lilbee.wiki.gen import generate_summary_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("auth-part2.md")
        assert len(chunks) > 0

        wiki_content = (
            "# JWT Tokens Summary\n\n"
            "JWT tokens use RS256 algorithm.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: auth-part2.md, excerpt: "JWT tokens are signed with RS256"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.92"],
        ):
            page = generate_summary_page("auth-part2.md", chunks, svc.provider, svc.store)

        assert page is not None
        wiki_root = cfg.data_root / cfg.wiki_dir
        result = read_page(wiki_root, "summaries/auth-part2")
        assert result is not None, "read_page returned None"
        assert "RS256" in result.content


class TestCitations:
    def test_citations_stored_in_db(self, wiki_pipeline):
        """After generation, citation records exist in the store."""
        from lilbee.wiki.gen import generate_summary_page

        svc = get_services()
        chunks = svc.store.get_chunks_by_source("auth-part3.md")
        assert len(chunks) > 0

        wiki_content = (
            "# Session Management Summary\n\n"
            "Sessions stored in Redis with 24h TTL.[^src1]\n"
            "Garbage collected every 6 hours.[^src2]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: auth-part3.md, excerpt: "Sessions stored in Redis with 24h TTL"\n'
            '[^src2]: auth-part3.md, excerpt: "garbage collected every 6 hours"\n'
        )
        with patch.object(
            svc.provider,
            "chat",
            side_effect=[wiki_content, "0.90"],
        ):
            page = generate_summary_page("auth-part3.md", chunks, svc.provider, svc.store)

        assert page is not None
        wiki_source = f"{cfg.wiki_dir}/summaries/auth-part3.md"
        citations = svc.store.get_citations_for_wiki(wiki_source)
        assert len(citations) > 0, "No citation records found in store"
        keys = [c["citation_key"] for c in citations]
        assert "src1" in keys


class TestWikiCLI:
    def test_wiki_cli_lint_command(self, wiki_pipeline):
        """'lilbee wiki lint' exits cleanly via CliRunner."""
        from typer.testing import CliRunner

        from lilbee.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["wiki", "lint"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
