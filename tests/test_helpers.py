"""Tests for CLI helper functions."""

from unittest import mock

import pytest
from rich.console import Console

from lilbee.cli.helpers import register_paths
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all helper tests."""
    snapshot = cfg.model_copy()

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.linked_roots = {}

    yield tmp_path

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestRegisterPaths:
    def test_returns_registered_labels(self, tmp_path):
        src = tmp_path / "corpus"
        src.mkdir()
        (src / "doc.txt").write_text("content")
        con = Console()

        registered = register_paths([src], con)

        assert registered == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(src.resolve())}

    def test_prints_warning_for_skipped(self, tmp_path):
        # A different corpus already holds the "corpus" label; a second one with
        # the same basename is skipped without --force and the user is warned.
        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        cfg.linked_roots = {"corpus": str(one)}
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)
        con = Console(quiet=True)

        with mock.patch.object(con, "print") as mock_print:
            registered = register_paths([two], con)

        assert registered == []
        mock_print.assert_called_once()
        assert "already exists" in str(mock_print.call_args)
