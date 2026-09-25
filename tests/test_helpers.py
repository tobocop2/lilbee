"""Tests for CLI helper functions."""

from unittest import mock

import pytest
from rich.console import Console

from lilbee.cli.helpers import print_prefixed, register_paths
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

        result = register_paths([src], con)

        assert result.registered == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(src.resolve())}

    def test_prints_warning_for_skipped(self, tmp_path):
        # A different corpus already holds the "corpus" label; a second one with
        # the same basename is skipped without --force and the user is warned.
        from lilbee.core import settings

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        settings.set_value(cfg.data_root, "linked_roots", {"corpus": str(one)})
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)
        con = Console(quiet=True)

        with mock.patch.object(con, "print") as mock_print:
            result = register_paths([two], con)

        assert result.registered == []
        assert result.name_taken == ["corpus"]
        mock_print.assert_called_once()
        assert "is taken by another source" in str(mock_print.call_args)

    def test_prints_a_bracketed_name_literally(self, tmp_path):
        """A bracketed source label in the warning must not go through markup."""
        from lilbee.core import settings

        one = tmp_path / "a" / "man[ual]"
        one.mkdir(parents=True)
        settings.set_value(cfg.data_root, "linked_roots", {"man[ual]": str(one)})
        two = tmp_path / "b" / "man[ual]"
        two.mkdir(parents=True)
        con = Console(quiet=True)

        with mock.patch.object(con, "print") as mock_print:
            register_paths([two], con)

        printed = mock_print.call_args.args[0]
        assert "man[ual]" in printed.plain

    def test_re_adding_the_same_path_is_tracked_not_warned(self, tmp_path):
        """--force would change nothing here, so the collision warning must not fire."""
        src = tmp_path / "corpus"
        src.mkdir()
        con = Console(quiet=True)
        register_paths([src], con)

        with mock.patch.object(con, "print") as mock_print:
            result = register_paths([src], con)

        assert result.tracked == ["corpus"]
        assert result.name_taken == []
        mock_print.assert_not_called()


class TestPrintPrefixed:
    def test_prefix_is_styled_and_detail_is_plain_text(self):
        """A bracket or a Windows separator in *detail* must survive verbatim."""
        con = Console(quiet=True)
        with mock.patch.object(con, "print") as mock_print:
            print_prefixed(con, "Error: ", "notes\\[draft].txt", style="bold red")

        printed = mock_print.call_args.args[0]
        assert printed.plain == "Error: notes\\[draft].txt"
        assert mock_print.call_args.kwargs == {"soft_wrap": True}

    def test_detail_need_not_be_a_string(self):
        """An exception is str()-ed, not required to already be text."""
        con = Console(quiet=True)
        with mock.patch.object(con, "print") as mock_print:
            print_prefixed(con, "Error: ", ValueError("bad[value]"), style="bold red")

        assert mock_print.call_args.args[0].plain == "Error: bad[value]"
