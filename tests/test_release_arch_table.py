"""Tests for the new-architecture table that pin-bump release notes carry.

A llama.cpp pin bump says only "Bump the bundled llama.cpp engine to <ref>".
The model support it adds is the user-visible payload of that bump and is
invisible in the notes otherwise, so the table is generated from the two tags'
``engine_archs.py`` rather than written by hand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "release_arch_table.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_arch_table", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rat = _load()


def _archs_module(ref: str, commit: str, archs: tuple[str, ...]) -> str:
    entries = "\n".join(f'        "{a}",' for a in archs)
    return (
        f'ENGINE_LLAMA_CPP_REF = "{ref}"\n'
        f'LLAMA_CPP_COMMIT = "{commit}"\n\n'
        "SUPPORTED_ARCHS: frozenset[str] = frozenset(\n"
        "    {\n"
        f"{entries}\n"
        "    }\n"
        ")\n"
    )


_OLD = _archs_module("memory-20260817-085323", "3890631", ("llama", "qwen3", "dots1"))
_NEW = _archs_module(
    "memory-20260828-212210", "cb6b3c7", ("llama", "qwen3", "dots1", "dots3note", "qwen4exp")
)


def test_added_archs_are_the_set_difference() -> None:
    assert rat.added_archs(_OLD, _NEW) == ["dots3note", "qwen4exp"]


def test_added_archs_ignores_removals_and_reordering() -> None:
    shrunk = _archs_module("r", "c", ("qwen3", "llama"))
    assert rat.added_archs(_NEW, shrunk) == []


def test_parses_the_engine_ref_and_commit() -> None:
    assert rat.engine_ref(_NEW) == "memory-20260828-212210"
    assert rat.engine_commit(_NEW) == "cb6b3c7"


def test_table_lists_every_new_arch_with_the_totals() -> None:
    out = rat.render(_OLD, _NEW)
    assert "| `dots3note` |" in out
    assert "| `qwen4exp` |" in out
    # Both totals are counted from the two sets, so a reader sees the size of
    # the bump without opening the diff.
    assert "runs 5 GGUF architectures, up from 3" in out
    assert "adds 2 architectures" in out


def test_table_links_each_arch_to_the_engine_source() -> None:
    out = rat.render(_OLD, _NEW)
    # Without a per-arch upstream PR to cite, the compare view between the two
    # pinned commits is the honest source for what the bump contains.
    assert "3890631" in out and "cb6b3c7" in out


def test_no_table_when_the_pin_did_not_move() -> None:
    assert rat.render(_OLD, _OLD) == ""


def test_no_table_when_the_pin_moved_but_added_no_archs() -> None:
    same_archs = _archs_module("memory-later", "deadbee", ("llama", "qwen3", "dots1"))
    assert rat.render(_OLD, same_archs) == ""


def test_render_tolerates_a_missing_previous_file() -> None:
    # The first release after the generated file appears has no old side.
    assert rat.render("", _NEW) == ""


def test_cli_prints_the_table_for_two_files(tmp_path, capsys) -> None:
    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    old.write_text(_OLD, encoding="utf-8")
    new.write_text(_NEW, encoding="utf-8")
    assert rat.main(["--old-file", str(old), "--new-file", str(new)]) == 0
    assert "dots3note" in capsys.readouterr().out


def test_cli_prints_nothing_when_the_pin_is_unchanged(tmp_path, capsys) -> None:
    same = tmp_path / "same.py"
    same.write_text(_OLD, encoding="utf-8")
    assert rat.main(["--old-file", str(same), "--new-file", str(same)]) == 0
    assert capsys.readouterr().out == ""
