"""The README formats table is generated from discovery's format map.

``tools/gen_formats_table.py`` renders it; ``make lint`` fails when the committed
README drifts from what the generator renders.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import get_args

import pytest
from tree_sitter_language_pack import SupportedLanguage

from lilbee.data.ingest.discovery import excluded_extension_reasons, supported_extension_map

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "gen_formats_table.py"
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("gen_formats_table", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[module name].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def block(generator) -> str:
    return generator.render_block()


class TestFormatsTable:
    def test_committed_readme_matches_the_generator(self, generator):
        readme = README.read_text(encoding="utf-8")
        assert readme == generator.render(readme), "README is stale: run `make docs-formats`"

    def test_every_ingestable_extension_has_one_cell(self, block):
        for ext in supported_extension_map():
            assert block.count(f"`{ext}`") == 1, ext

    def test_refused_extensions_are_absent(self, block):
        for ext in excluded_extension_reasons():
            assert f"`{ext}`" not in block, ext

    def test_no_format_falls_into_the_other_row(self, generator):
        assert generator.formats_by_row()["Other"] == []

    def test_archives_carry_the_member_note(self, generator, block):
        archives = generator.formats_by_row()["Archives"]
        assert ".zip" in archives
        assert generator.ARCHIVE_NOTE in block

    def test_code_row_counts_tree_sitter_languages(self, block):
        count = len(get_args(SupportedLanguage))
        assert f"[{count} languages]" in block
        assert "`.py`" in block

    def test_lead_line_carries_live_counts(self, block):
        assert f"ingests {len(supported_extension_map())} file extensions" in block

    def test_render_refuses_a_readme_without_markers(self, generator):
        with pytest.raises(SystemExit, match="no <!-- formats-table:start -->"):
            generator.render("# no markers here\n")

    def test_check_mode_fails_on_a_stale_readme(self, generator, tmp_path, monkeypatch):
        stale = tmp_path / "README.md"
        stale.write_text(
            f"intro\n{generator.START_MARKER}\nstale\n{generator.END_MARKER}\noutro\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(generator, "README", stale)
        monkeypatch.setattr(sys, "argv", ["gen_formats_table.py", "--check"])
        with pytest.raises(SystemExit, match="out of date"):
            generator.main()

    def test_write_mode_regenerates_the_block_in_place(self, generator, tmp_path, monkeypatch):
        readme = tmp_path / "README.md"
        readme.write_text(
            f"intro\n{generator.START_MARKER}\nstale\n{generator.END_MARKER}\noutro\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(generator, "README", readme)
        monkeypatch.setattr(sys, "argv", ["gen_formats_table.py"])
        generator.main()
        written = readme.read_text(encoding="utf-8")
        assert written == f"intro\n{generator.render_block()}\noutro\n"
        monkeypatch.setattr(sys, "argv", ["gen_formats_table.py", "--check"])
        generator.main()
