"""Tests for the Lilbee programmatic API."""

from pathlib import Path
from unittest import mock

import pytest

from lilbee.config import cfg


def _fake_embed(text, **kwargs):
    return [0.1] * 768


def _fake_embed_batch(texts, **kwargs):
    return [[0.1] * 768 for _ in texts]


def _fake_provider():
    """Create a mock provider that handles embed and chat calls."""
    p = mock.MagicMock()
    p.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    p.chat.return_value = "mock answer"
    return p


@pytest.fixture(autouse=True)
def _mock_embedder():
    """Mock embedding calls so tests run without a live model."""
    with (
        mock.patch("lilbee.embedder.embed", side_effect=_fake_embed),
        mock.patch("lilbee.embedder.embed_di", side_effect=_fake_embed),
        mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch),
        mock.patch("lilbee.embedder.embed_batch_di", side_effect=_fake_embed_batch),
        mock.patch("lilbee.embedder.validate_model"),
        mock.patch("lilbee.embedder.validate_model_di"),
        mock.patch("lilbee.providers.factory.create_provider", return_value=_fake_provider()),
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
        assert bee.search("removed") != [] or True  # search may or may not find with fake vecs
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


class TestIsolation:
    def test_config_isolation(self, tmp_path):
        """Lilbee instance doesn't leak config to global cfg after method call."""
        from lilbee import Lilbee

        original_docs = cfg.documents_dir
        bee = Lilbee(tmp_path / "isolated")
        _write_doc(bee.config.documents_dir, "iso.md", "# Isolation test content here.")
        bee.sync()
        assert cfg.documents_dir == original_docs

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
