"""The notice of MPL-2.0 packages of a Python pack comes from the tree and the committed list."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from lilbee.providers.fleet import engine_store

_TOOL = Path(__file__).resolve().parents[1] / "tools/wheel-build/mpl_pypi_notice.py"
_SOURCES = Path(__file__).resolve().parents[1] / "packaging/engine-mlx/SOURCE-MPL-2.0"
_TQDM = (
    "https://files.pythonhosted.org/packages/source/t/tqdm/tqdm-4.70.1.tar.gz "
    "cefd0eca11b2a37a3aee776544d4f4ae913f02688135b5556b8788dfa474afc4"
)
_OTHER = (
    "https://files.pythonhosted.org/packages/source/o/other_pkg/other_pkg-2.0.tar.gz " + "a" * 64
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mpl_pypi_notice", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_the_notice_holds_the_line_of_each_package_of_the_tree_and_no_other() -> None:
    text = tool.notice(["tqdm-4.70.1"], [_OTHER, _TQDM])
    assert text == _TQDM + "\n"


@pytest.mark.parametrize("package", ["other-pkg-2.0", "Other_Pkg-2.0", "OTHER.pkg-2.0"])
def test_a_package_name_matches_in_the_form_of_pep_503(package: str) -> None:
    text = tool.notice([package], [_TQDM, _OTHER])
    assert text == _OTHER + "\n"


def test_the_notice_stops_on_a_package_with_no_line() -> None:
    with pytest.raises(SystemExit, match=r"no line for tqdm-4\.71\.0"):
        tool.notice(["tqdm-4.71.0"], [_TQDM])


def test_the_notice_stops_on_a_package_with_a_line_of_another_version() -> None:
    with pytest.raises(SystemExit, match=r"no line for other_pkg-3\.0"):
        tool.notice(["tqdm-4.70.1", "other_pkg-3.0"], [_TQDM, _OTHER])


def test_the_notice_stops_on_a_name_that_is_not_a_package_and_version() -> None:
    with pytest.raises(SystemExit, match="no line for tqdm"):
        tool.notice(["tqdm"], [_TQDM])


def test_a_line_that_is_not_a_url_and_a_sha_is_no_line() -> None:
    with pytest.raises(SystemExit, match=r"no line for tqdm-4\.70\.1"):
        tool.notice(["tqdm-4.70.1"], ["# tqdm-4.70.1", _TQDM.split()[0], "", "x"])


def test_no_mpl_package_gives_no_notice() -> None:
    assert tool.notice([], [_TQDM]) is None


def test_the_notice_of_the_committed_list_is_one_the_store_accepts() -> None:
    sources = _SOURCES.read_text(encoding="utf-8").splitlines()
    text = tool.notice(["tqdm-4.70.1"], sources)
    assert text is not None
    assert engine_store._verified_notices([("SOURCE-MPL-2.0", text)]) == {"SOURCE-MPL-2.0"}


def test_main_writes_the_notice(tmp_path: Path) -> None:
    (tmp_path / "mpl.txt").write_text("tqdm-4.70.1\n", encoding="utf-8")
    (tmp_path / "sources").write_text(_OTHER + "\n" + _TQDM + "\n", encoding="utf-8")
    out = tmp_path / "SOURCE-MPL-2.0"
    tool.main(["x", str(tmp_path / "mpl.txt"), str(tmp_path / "sources"), str(out)])
    assert out.read_text(encoding="utf-8") == _TQDM + "\n"


def test_main_writes_nothing_without_an_mpl_package(tmp_path: Path) -> None:
    (tmp_path / "mpl.txt").write_text("", encoding="utf-8")
    (tmp_path / "sources").write_text(_TQDM + "\n", encoding="utf-8")
    out = tmp_path / "SOURCE-MPL-2.0"
    tool.main(["x", str(tmp_path / "mpl.txt"), str(tmp_path / "sources"), str(out)])
    assert not out.exists()


def test_main_prints_usage_with_the_wrong_argument_count() -> None:
    with pytest.raises(SystemExit, match=r"Usage: mpl_pypi_notice\.py"):
        tool.main(["x"])
