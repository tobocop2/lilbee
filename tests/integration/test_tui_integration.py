"""TUI integration tests with real backends.

These tests exercise the Textual TUI against a real RAG pipeline with
downloaded models and indexed documents. No mocks: real embeddings,
real LLM streaming, real LanceDB queries.

Run with:
    uv run pytest tests/integration/test_tui_integration.py -v -m slow
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer

from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg
from lilbee.core.services import get_services

pytestmark = pytest.mark.slow


class _IntegrationChatApp(App[None]):
    """Minimal app that pushes ChatScreen with real services."""

    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


async def _submit_slash(pilot, app, command: str) -> None:
    """Type a slash command into the chat input and press enter."""
    inp = app.screen.query_one("#chat-input", ChatInput)
    inp.value = command
    await pilot.press("enter")
    await pilot.pause()


def _walk_tree_labels(node):
    """Yield every node label in a Tree (for assertion lookups)."""
    yield str(node.label)
    for child in node.children:
        yield from _walk_tree_labels(child)


def _first_leaf_with_slug(node):
    """Return the first Tree leaf whose data holds a page slug, or None."""
    if node.data is not None and not node.allow_expand:
        return node
    for child in node.children:
        found = _first_leaf_with_slug(child)
        if found is not None:
            return found
    return None


class TestChatFlow:
    """Real chat with streaming LLM response."""

    @pytest.mark.timeout(360)
    async def test_chat_returns_real_answer(self, rag_pipeline) -> None:
        """Type a question about indexed docs, get a real streamed answer.

        The pause loop pairs ``pilot.pause`` with a 100 ms ``asyncio.sleep``
        so wallclock time actually advances. The chat worker subprocess
        cold-start (model load, mmap, llama-cpp init) is several seconds on
        CPU CI; without the sleep the 600 iterations spin-yield in
        milliseconds and the assertion fires before streaming ever begins.
        Bumped to 360s for the same reason as ``test_ask_includes_citations``:
        small-model RAG streaming on CPU runners is genuinely slow.
        """
        import asyncio

        from lilbee.core.services import reset_services

        app = _IntegrationChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from tests.integration.conftest import _CI_CHAT_REPO, _resolve_installed_ref

            cfg.chat_model = _resolve_installed_ref(_CI_CHAT_REPO)
            reset_services()
            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "What engine does the Thunderbolt X500 have?"
            await pilot.press("enter")

            # ~60s budget: 600 * 100ms covers chat worker cold-start (3-5s)
            # plus the small-model token stream on the slowest CPU runners.
            for _ in range(600):
                await pilot.pause()
                await asyncio.sleep(0.1)
                if not app.screen.streaming:
                    break

            assert not app.screen.streaming, "Streaming did not complete in time"

            # Wait for all workers to finish before app teardown
            # (llama-cpp segfaults if model is freed while worker thread reads from it)
            for worker in list(app.screen.workers):
                await worker.wait()

            assert len(app.screen._history) >= 2
            assistant_reply = app.screen._history[-1]["content"]
            assert len(assistant_reply) > 0, "Assistant reply was empty"

            # Verify the streamed response references retrieved context rather
            # than asserting specific engine facts: small chat models (Qwen3-0.6B
            # on CPU CI) frequently echo a different retrieved chunk verbatim
            # instead of synthesizing the answer. The integration concern here
            # is that the TUI streams a real RAG response end to end; factual
            # accuracy of the LLM is covered by ``test_ask_answer_references_facts``
            # in test_rag_integration.py, which is gated on
            # ``skip_if_small_chat_model``.
            reply_lower = assistant_reply.lower()
            assert any(
                term in reply_lower
                for term in (
                    "v6",
                    "3.5",
                    "turboforce",
                    "365",
                    "thunderbolt",
                    "engine",
                    "authentication",
                    "jwt",
                )
            ), f"Expected reply to reference indexed docs, got: {assistant_reply[:200]}"


class TestAddAndSync:
    """Add a file via the TUI and verify it becomes searchable."""

    async def test_add_file_becomes_searchable(self, rag_pipeline, tmp_path) -> None:
        """Add a file with unique content, verify search finds it."""
        test_file = tmp_path / "quantum_test.md"
        test_file.write_text(
            "# Quantum Teleportation Protocol\n\n"
            "Quantum entanglement enables instantaneous state transfer between "
            "qubits separated by arbitrary distances using Bell state measurements."
        )

        app = _IntegrationChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit_slash(pilot, app, f"/add {test_file}")

            for _ in range(600):
                await pilot.pause()
                if not app.screen._sync_active:
                    break

            assert not app.screen._sync_active, "Sync did not complete in time"

            results = get_services().searcher.search("quantum entanglement teleportation")
            sources = [r.source for r in results]
            assert "quantum_test.md" in sources, (
                f"Expected quantum_test.md in search results, got: {sources}"
            )


class TestDelete:
    """Delete a document and verify removal from search."""

    async def test_delete_removes_from_search(self, rag_pipeline) -> None:
        """Delete a document via /delete, verify it no longer appears in search."""
        results = get_services().searcher.search("Thunderbolt X500 engine")
        sources_before = [r.source for r in results]
        assert "specs.md" in sources_before, "specs.md should be findable before delete"

        app = _IntegrationChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit_slash(pilot, app, "/delete specs.md")

        results_after = get_services().searcher.search("Thunderbolt X500 engine")
        sources_after = [r.source for r in results_after]
        assert "specs.md" not in sources_after, "specs.md should be gone after delete"

        from lilbee.data.ingest import sync

        await sync(quiet=True)
        get_services().store.ensure_fts_index()


class TestStatusScreen:
    """Status screen with real knowledge base data."""

    async def test_status_shows_real_stats(self, rag_pipeline) -> None:
        """Status screen displays real document names and chunk counts."""
        from lilbee.cli.tui.screens.status import StatusScreen

        class _StatusApp(App[None]):
            CSS = ""

            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(StatusScreen())

        app = _StatusApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            table = app.screen.query_one("#docs-table", DataTable)
            row_count = table.row_count
            assert row_count >= 9, f"Expected >= 9 document rows, got {row_count}"

            rows = [table.get_row_at(i) for i in range(row_count)]
            doc_names = [str(row[0]) for row in rows]
            assert "specs.md" in doc_names, f"Expected specs.md in {doc_names}"


class TestSetCommand:
    """Config updates via /set."""

    async def test_set_updates_config(self, rag_pipeline) -> None:
        """'/set top_k 10' updates the config value."""
        original_top_k = cfg.top_k
        try:
            app = _IntegrationChatApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _submit_slash(pilot, app, "/set top_k 10")

            assert cfg.top_k == 10, f"Expected top_k=10, got {cfg.top_k}"
        finally:
            cfg.top_k = original_top_k


class TestModelSwitch:
    """Model switch via /model."""

    async def test_model_switch_updates_config(self, rag_pipeline) -> None:
        """'/model ollama/qwen3:0.6b' updates cfg.chat_model."""
        original_model = cfg.chat_model
        try:
            app = _IntegrationChatApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _submit_slash(pilot, app, "/model ollama/qwen3:0.6b")

            assert cfg.chat_model == "ollama/qwen3:0.6b", (
                f"Expected chat_model='ollama/qwen3:0.6b', got '{cfg.chat_model}'"
            )
        finally:
            cfg.chat_model = original_model


class TestCrawlAndSync:
    """Crawl a URL and verify content becomes searchable."""

    async def test_crawl_becomes_searchable(self, rag_pipeline, monkeypatch) -> None:
        """Crawl a local HTTP page, verify its content is indexed."""
        pytest.importorskip("crawl4ai")
        pytest.importorskip("pytest_httpserver")

        import ipaddress

        from pytest_httpserver import HTTPServer

        from lilbee.crawler import url_filter as crawler_url_filter

        html = (
            "<html><head><title>Test</title></head><body>"
            "<h1>Bioluminescent Jellyfish Research</h1>"
            "<p>Deep-sea jellyfish produce light through luciferin-luciferase "
            "reactions in specialized photocytes.</p>"
            "</body></html>"
        )

        loopback_v4 = ipaddress.ip_network("127.0.0.0/8")
        loopback_v6 = ipaddress.ip_network("::1/128")
        original_fn = crawler_url_filter.get_blocked_networks
        filtered = tuple(net for net in original_fn() if net not in (loopback_v4, loopback_v6))
        monkeypatch.setattr(crawler_url_filter, "get_blocked_networks", lambda: filtered)

        server = HTTPServer()
        server.expect_request("/jellyfish").respond_with_data(html, content_type="text/html")
        server.start()
        try:
            url = str(server.url_for("/jellyfish"))
            app = _IntegrationChatApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _submit_slash(pilot, app, f"/add {url}")

                for _ in range(600):
                    await pilot.pause()
                    if not app.screen._sync_active:
                        break

            results = get_services().searcher.search("bioluminescent jellyfish luciferin")
            assert len(results) > 0, "Expected crawled content to be searchable"
        finally:
            server.clear()
            server.stop()


class _WikiApp(App[None]):
    """Minimal app that pushes WikiScreen for integration tests."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.wiki import WikiScreen

        self.push_screen(WikiScreen())


class TestWikiScreen:
    """WikiScreen integration tests: verify page listing and content display."""

    @pytest.fixture(autouse=True)
    def _setup_wiki(self, tmp_path):
        """Create wiki pages on disk for the screen to discover."""
        wiki_root = tmp_path / "wiki"
        summaries = wiki_root / "summaries"
        summaries.mkdir(parents=True)

        (summaries / "test-doc.md").write_text(
            "---\n"
            "generated_by: test-model\n"
            "generated_at: 2026-01-01T00:00:00\n"
            'sources: ["doc.md"]\n'
            "faithfulness_score: 0.85\n"
            "---\n\n"
            "# Test Document Summary\n\n"
            "This is a test wiki page with real content.\n",
            encoding="utf-8",
        )
        (summaries / "second-doc.md").write_text(
            "---\n"
            "generated_by: test-model\n"
            "generated_at: 2026-01-02T00:00:00\n"
            'sources: ["other.md"]\n'
            "faithfulness_score: 0.92\n"
            "---\n\n"
            "# Second Document\n\n"
            "Another wiki page for testing.\n",
            encoding="utf-8",
        )

        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        cfg.data_dir = tmp_path
        cfg.data_root = tmp_path
        yield

    async def test_wiki_screen_shows_generated_pages(self):
        """WikiScreen sidebar lists generated pages."""
        from textual.widgets import Tree

        app = _WikiApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tree = app.screen.query_one("#wiki-page-list", Tree)
            labels = list(_walk_tree_labels(tree.root))
            assert any("Test Doc" in label for label in labels), (
                f"Test doc not in sidebar: {labels}"
            )
            assert any("Second Doc" in label for label in labels), (
                f"Second doc not in sidebar: {labels}"
            )

    async def test_wiki_screen_displays_content(self):
        """Selecting a page renders its markdown content."""
        from textual.widgets import Markdown, Tree

        app = _WikiApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tree = app.screen.query_one("#wiki-page-list", Tree)
            leaf = _first_leaf_with_slug(tree.root)
            assert leaf is not None, "No selectable wiki leaf found"
            tree.post_message(Tree.NodeSelected(leaf))
            await pilot.pause()
            content = app.screen.query_one("#wiki-content", Markdown)
            rendered = str(content._markdown)
            assert len(rendered) > 0, "Wiki content area is empty after selecting a page"
