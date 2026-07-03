"""Tests for the Lilbee programmatic API."""

from pathlib import Path
from unittest import mock

import pytest

from lilbee.core.config import cfg


def _fake_embed(text):
    return [0.1] * 768


def _fake_embed_batch(texts, **kwargs):
    return [[0.1] * 768 for _ in texts]


@pytest.fixture(autouse=True)
def _mock_embedder():
    """Mock embedding calls so tests run without a live model."""
    with mock.patch(
        "lilbee.providers.factory.create_provider",
        return_value=mock.MagicMock(
            embed=mock.MagicMock(side_effect=lambda texts: [_fake_embed(t) for t in texts]),
            pull_model=mock.MagicMock(),
            shutdown=mock.MagicMock(),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _isolate_cfg():
    """Restore global cfg after every test."""
    snapshot = cfg.model_copy()
    cfg.concept_graph = False
    cfg.query_expansion_count = 0
    cfg.hyde = False
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write_doc(docs_dir: Path, name: str, content: str) -> Path:
    """Write a markdown file into a documents directory."""
    path = docs_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestCreate:
    def test_create_with_documents_dir(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "myproject")
        assert bee.config.documents_dir.exists()
        assert bee.config.data_dir.exists()
        assert "myproject" in str(bee.config.data_root)

    def test_create_with_config(self, tmp_path):
        from lilbee import Lilbee

        custom = cfg.model_copy(
            update={
                "data_root": tmp_path,
                "documents_dir": tmp_path / "docs",
                "data_dir": tmp_path / "data",
                "lancedb_dir": tmp_path / "data" / "lancedb",
            },
        )
        bee = Lilbee(config=custom)
        assert bee.config.documents_dir == tmp_path / "docs"

    def test_create_with_both_raises(self, tmp_path):
        from lilbee import Lilbee

        custom = cfg.model_copy(
            update={
                "data_root": tmp_path,
                "documents_dir": tmp_path / "docs",
                "data_dir": tmp_path / "data",
                "lancedb_dir": tmp_path / "data" / "lancedb",
            },
        )
        with pytest.raises(ValueError, match="not both"):
            Lilbee(tmp_path / "dir", config=custom)

    def test_create_with_neither_uses_env(self, tmp_path, monkeypatch):
        from lilbee import Lilbee

        monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "envroot"))
        bee = Lilbee()
        assert "envroot" in str(bee.config.data_root)

    def test_user_supplied_provider_is_wired_through(self, tmp_path):
        """An explicit ``provider=`` argument replaces the auto-created provider.

        Constructing the embedder with the supplied mock and triggering an
        embed-bearing path (``search`` → ``Searcher`` → ``Embedder``) must call
        the mock's ``embed``, not the factory-built provider's.
        """
        from lilbee import Lilbee

        custom_provider = mock.MagicMock(
            embed=mock.MagicMock(side_effect=lambda texts: [[0.5] * 768 for _ in texts]),
            pull_model=mock.MagicMock(),
            shutdown=mock.MagicMock(),
        )
        with mock.patch("lilbee.providers.factory.create_provider") as factory:
            bee = Lilbee(tmp_path / "userprov", provider=custom_provider)
            # The factory must NOT have been called: the user's provider wins.
            factory.assert_not_called()

        # The composed Embedder must hold the user's provider.
        assert bee.embedder._provider is custom_provider

        # Triggering a search (empty index is fine) must route the embed call
        # through the user's provider, not via the factory-built one.
        bee.search("anything")
        assert custom_provider.embed.called


class TestSync:
    def test_sync_indexes_documents(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "notes.md", "# Notes\nThe answer is 42.")
        result = bee.sync()
        assert "notes.md" in result.added

    def test_sync_returns_result(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "a.md", "Hello world content here.")
        result = bee.sync()
        assert isinstance(result.added, list)
        assert isinstance(result.unchanged, int)


class TestSearch:
    def test_search_returns_results(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "info.md", "# Auth\nAuthentication uses OAuth2.")
        bee.sync()
        results = bee.search("authentication")
        assert len(results) > 0
        assert any("OAuth2" in r.chunk for r in results)

    def test_search_empty_index(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        results = bee.search("anything")
        assert results == []

    def test_search_respects_top_k(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        for i in range(5):
            _write_doc(
                bee.config.documents_dir,
                f"doc{i}.md",
                f"# Doc {i}\nContent about topic number {i} with enough words to chunk.",
            )
        bee.sync()
        results = bee.search("topic", top_k=2)
        assert len(results) <= 2


class TestAdd:
    def test_add_copies_and_syncs(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        external = tmp_path / "external.md"
        external.write_text("# External\nThis file lives outside the project.")
        result = bee.add([external])
        assert "external.md" in result.added
        found = bee.search("external")
        assert len(found) > 0


class TestRemove:
    def test_remove_deletes_from_index(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "gone.md", "# Gone\nThis will be removed shortly.")
        bee.sync()
        bee.remove("gone.md")
        status = bee.status()
        assert "gone.md" not in status["sources"]


class TestStatus:
    def test_status_returns_info(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "s.md", "# Status\nSome content for status check.")
        bee.sync()
        info = bee.status()
        assert info["document_count"] == 1
        assert "s.md" in info["sources"]
        assert "documents_dir" in info


class TestRebuild:
    def test_rebuild_recreates_index(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "rb.md", "# Rebuild\nRebuild test document.")
        bee.sync()
        result = bee.rebuild()
        assert "rb.md" in result.added


class TestPropertyAccessors:
    def test_store_property(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        assert bee.store is not None

    def test_embedder_property(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        assert bee.embedder is not None

    def test_searcher_property(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        assert bee.searcher is not None


class TestRemovePathTraversal:
    def test_remove_path_traversal_skips(self, tmp_path):
        """remove() with an unknown/traversal name is a no-op and deletes nothing."""
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "proj")
        _write_doc(bee.config.documents_dir, "legit.md", "# Legit\nSome content here.")
        bee.sync()
        bee.remove("../../etc/passwd")
        status = bee.status()
        assert "legit.md" in status["sources"]


class TestIsolation:
    def test_config_isolation(self, tmp_path):
        """Lilbee instance doesn't leak config to global cfg after method call."""
        from lilbee import Lilbee

        original_docs = cfg.documents_dir
        bee = Lilbee(tmp_path / "isolated")
        _write_doc(bee.config.documents_dir, "iso.md", "# Isolation test content here.")
        bee.sync()
        assert cfg.documents_dir == original_docs

    def test_services_stable_and_global_singleton_untouched(self, tmp_path):
        """Every call runs against the instance's own Services (no per-call rebuild),
        and the process-global services singleton is never created or swapped."""
        from lilbee import Lilbee
        from lilbee.app.services import peek_services

        before = peek_services()
        bee = Lilbee(tmp_path / "stable")
        _write_doc(bee.config.documents_dir, "s.md", "# Stable\nContent for a stability check.")
        services = bee._services
        bee.sync()
        # A second search does not tear down and rebuild the fleet/services: the
        # same container (and its store) backs sync, search, and status.
        bee.search("stability")
        bee.status()
        assert bee._services is services
        assert bee.store is services.store
        assert peek_services() is before

    def test_multiple_instances_sequential(self, tmp_path):
        """Two Lilbee instances with different dirs work sequentially."""
        from lilbee import Lilbee

        bee_a = Lilbee(tmp_path / "a")
        bee_b = Lilbee(tmp_path / "b")

        _write_doc(bee_a.config.documents_dir, "a.md", "# Alpha\nContent for project A.")
        _write_doc(bee_b.config.documents_dir, "b.md", "# Beta\nContent for project B.")

        bee_a.sync()
        bee_b.sync()

        status_a = bee_a.status()
        status_b = bee_b.status()
        assert "a.md" in status_a["sources"]
        assert "b.md" in status_b["sources"]
        assert "b.md" not in status_a["sources"]
        assert "a.md" not in status_b["sources"]


class TestClose:
    def test_close_shuts_down_and_is_idempotent(self, tmp_path):
        """close() shuts the provider and store down once; a second call is a no-op."""
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "closeme")
        provider = bee._services.provider
        with mock.patch.object(bee._services.store, "close") as store_close:
            bee.close()
            bee.close()
        provider.shutdown.assert_called_once()
        store_close.assert_called_once()


class TestPackageGetattr:
    def test_unknown_attribute_raises(self):
        """Package-level ``__getattr__`` raises AttributeError for unknown names."""
        import lilbee

        with pytest.raises(AttributeError, match="has no attribute 'definitely_not_a_thing'"):
            getattr(lilbee, "definitely_not_a_thing")  # noqa: B009

    def test_dunder_probe_skips_the_import_fallback(self, monkeypatch):
        """Introspection dunders raise immediately without attempting an import."""
        import importlib

        import lilbee

        def _boom(name, package=None):
            raise AssertionError("dunder probe must not import")

        monkeypatch.setattr(importlib, "import_module", _boom)
        with pytest.raises(AttributeError, match="__wrapped__"):
            getattr(lilbee, "__wrapped__")  # noqa: B009

    def test_submodule_attribute_self_heals(self, monkeypatch):
        """``lilbee.<submodule>`` re-imports when the package binding is missing.

        Dotted-path resolvers (monkeypatch.setattr, mock.patch) getattr their
        way down from the package; a missing binding must re-import, not raise.
        """
        import lilbee

        monkeypatch.delattr(lilbee, "providers")
        assert getattr(lilbee, "providers").__name__ == "lilbee.providers"  # noqa: B009

    def test_submodule_dependency_error_propagates(self, monkeypatch):
        """A submodule whose own dependency is missing surfaces that ImportError.

        Only a genuinely missing ``lilbee.<name>`` converts to AttributeError; a
        ModuleNotFoundError raised from inside an existing submodule must not be
        masked as a missing attribute.
        """
        import importlib

        import lilbee

        def _boom(name, package=None):
            raise ModuleNotFoundError("No module named 'notinstalled'", name="notinstalled")

        monkeypatch.setattr(importlib, "import_module", _boom)
        with pytest.raises(ModuleNotFoundError, match="notinstalled"):
            getattr(lilbee, "definitely_not_bound")  # noqa: B009


class TestMemory:
    """Real round-trip tests against the ``_memories`` LanceDB table."""

    def test_remember_and_list(self, tmp_path):
        from lilbee import Lilbee
        from lilbee.data.store import MemoryKind

        bee = Lilbee(tmp_path / "mem")
        fact_id = bee.remember("the project uses rust")
        pref_id = bee.remember("answer tersely", kind=MemoryKind.PREFERENCE, shared=True)
        assert fact_id != pref_id

        stored = {m.text: m for m in bee.memories()}
        assert set(stored) == {"the project uses rust", "answer tersely"}
        assert stored["answer tersely"].kind is MemoryKind.PREFERENCE
        assert stored["answer tersely"].shared is True

    def test_recall_returns_facts(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "mem")
        bee.remember("the Crown Vic brake bleed order is RR, LR, RF, LF")
        results = bee.recall("how do I bleed the brakes", top_k=5)
        assert any("brake bleed" in m.text for m in results)

    def test_forget_removes_memory(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "mem")
        memory_id = bee.remember("disposable note")
        assert bee.forget(memory_id) is True
        assert bee.memories() == []

    def test_forget_unknown_id_returns_false(self, tmp_path):
        from lilbee import Lilbee

        bee = Lilbee(tmp_path / "mem")
        assert bee.forget("does-not-exist") is False
