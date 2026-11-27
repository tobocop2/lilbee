"""The Python licence collector copies licence texts, names terms and refuses copyleft trees."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/collect_python_licences.py"
)
_PSF = "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2\n"
_BSD3 = "Redistribution.\nNeither the name of the author nor the names of contributors."
_BSD2 = "Redistribution and use in source and binary forms."


def _package(
    site: pathlib.Path, name: str, *metadata: str, licence: str | None = "text"
) -> pathlib.Path:
    dist = site / f"{name}-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "\n".join(["Metadata-Version: 2.4", f"Name: {name}", *metadata]) + "\n", encoding="utf-8"
    )
    if licence is not None:
        (dist / "licenses").mkdir()
        (dist / "licenses" / "LICENSE").write_text(licence, encoding="utf-8")
    return dist


def _run(
    site: pathlib.Path,
    tmp: pathlib.Path,
    *supplied: pathlib.Path,
    mpl: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(_SCRIPT),
            *(["--mpl", str(mpl)] if mpl else []),
            str(site),
            str(tmp / "out"),
            str(tmp / "terms"),
            *map(str, supplied),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.fixture
def site(tmp_path: pathlib.Path) -> pathlib.Path:
    """A site-packages folder whose parent holds the PSF interpreter licence."""
    folder = tmp_path / "python" / "site-packages"
    folder.mkdir(parents=True)
    (folder.parent / "LICENSE.txt").write_text(_PSF, encoding="utf-8")
    return folder


def test_names_each_term_and_copies_the_texts(site: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _package(site, "numpy", "License-Expression: BSD-3-Clause AND (0BSD OR Zlib)")
    _package(site, "tqdm", "License: MPL-2.0 AND MIT")
    _package(site, "hub", "License: Apache 2.0 License")
    done = _run(site, tmp_path)
    assert done.returncode == 0, done.stderr
    terms = (tmp_path / "terms").read_text(encoding="utf-8").split()
    assert terms == ["0BSD", "Apache-2.0", "BSD-3-Clause", "MIT", "MPL-2.0", "PSF-2.0", "Zlib"]
    assert (tmp_path / "out" / "numpy-1.0" / "LICENSE").read_text(encoding="utf-8") == "text"


def test_reads_the_classifier_and_tells_bsd_clauses_by_their_text(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "three", "Classifier: License :: OSI Approved :: BSD License", licence=_BSD3)
    _package(site, "two", "Classifier: License :: OSI Approved :: BSD License", licence=_BSD2)
    _package(site, "isc", "Classifier: License :: OSI Approved :: ISC License (ISCL)")
    assert _run(site, tmp_path).returncode == 0
    terms = (tmp_path / "terms").read_text(encoding="utf-8").split()
    assert terms == ["BSD-2-Clause", "BSD-3-Clause", "ISC", "PSF-2.0"]


def test_copies_a_licence_file_from_the_top_of_the_dist_info(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    dist = _package(site, "old", "License: MIT", licence=None)
    (dist / "COPYING.txt").write_text("old text", encoding="utf-8")
    assert _run(site, tmp_path).returncode == 0
    assert (tmp_path / "out" / "old-1.0" / "COPYING.txt").read_text(encoding="utf-8") == "old text"


@pytest.mark.parametrize(
    ("metadata", "licence", "message"),
    [
        (["License-Expression: GPL-3.0-only"], "text", "old-1.0: GPL-3.0-only is copyleft"),
        (
            ["Classifier: License :: OSI Approved :: GNU Lesser General Public License v3"],
            "text",
            "is copyleft",
        ),
        (
            ["License-Expression: MIT"],
            "GNU GENERAL PUBLIC LICENSE\nVersion 3",
            "holds a GNU licence text",
        ),
        (["Name: nothing"], "text", "names no licence"),
        (["License-Expression: MIT"], None, "ships no licence file"),
    ],
)
def test_refuses_a_package_that_breaks_a_rule(
    site: pathlib.Path,
    tmp_path: pathlib.Path,
    metadata: list[str],
    licence: str | None,
    message: str,
) -> None:
    _package(site, "old", *metadata, licence=licence)
    done = _run(site, tmp_path)
    assert done.returncode == 1
    assert message in done.stderr


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "ships no LICENSE.txt"),
        ("GNU GENERAL PUBLIC LICENSE\n", "holds a GNU licence text"),
        ("Some other terms\n", "is not the PSF licence"),
    ],
)
def test_refuses_an_interpreter_without_the_psf_licence(
    site: pathlib.Path, tmp_path: pathlib.Path, text: str | None, message: str
) -> None:
    _package(site, "ok", "License: MIT")
    interpreter = site.parent / "LICENSE.txt"
    if text is None:
        interpreter.unlink()
    else:
        interpreter.write_text(text, encoding="utf-8")
    done = _run(site, tmp_path)
    assert done.returncode == 1
    assert message in done.stderr


def _supplied(tmp: pathlib.Path, *lines: str) -> pathlib.Path:
    folder = tmp / "project"
    folder.mkdir(exist_ok=True)
    (folder / "LICENSE-Apache-2.0").write_text("Apache License text", encoding="utf-8")
    path = folder / "supplied.txt"
    path.write_text("\n".join(["# comment", "", *lines]) + "\n", encoding="utf-8")
    return path


def test_takes_the_supplied_file_of_a_package_that_ships_none(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "tokenizers", "License-Expression: Apache-2.0", licence=None)
    supplied = _supplied(tmp_path, "tokenizers-1.0 LICENSE-Apache-2.0")
    done = _run(site, tmp_path, supplied)
    assert done.returncode == 0, done.stderr
    copied = tmp_path / "out" / "tokenizers-1.0" / "LICENSE-Apache-2.0"
    assert copied.read_text(encoding="utf-8") == "Apache License text"
    assert (tmp_path / "terms").read_text(encoding="utf-8").split() == ["Apache-2.0", "PSF-2.0"]


@pytest.mark.parametrize(
    ("ships", "line", "message"),
    [
        ("text", "old-1.0 LICENSE-Apache-2.0", "old-1.0: ships its own licence file"),
        (None, "old-1.0 NO-SUCH-FILE", "the supplied licence file"),
        (None, "old-2.0 LICENSE-Apache-2.0", "old-1.0: ships no licence file"),
    ],
)
def test_refuses_a_supplied_file_that_does_not_fit_the_tree(
    site: pathlib.Path, tmp_path: pathlib.Path, ships: str | None, line: str, message: str
) -> None:
    _package(site, "old", "License-Expression: MIT", licence=ships)
    done = _run(site, tmp_path, _supplied(tmp_path, line))
    assert done.returncode == 1
    assert message in done.stderr


_GCC_SECTION = (
    "\n" * 60 + "Name: GCC runtime\n\n  GNU GENERAL PUBLIC LICENSE\n  Version 3, 29 June 2007\n"
)


def test_refuses_a_licence_file_with_a_gnu_title_past_its_head(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The store reads the whole text, so a title after the first 2000 bytes also refuses."""
    _package(
        site, "numpy", "License-Expression: BSD-3-Clause", licence="BSD" + "x" * 3000 + _GCC_SECTION
    )
    done = _run(site, tmp_path)
    assert done.returncode == 1
    assert "numpy-1.0: LICENSE holds a GNU licence text" in done.stderr


def test_a_replacing_file_is_written_over_the_tree_and_into_the_copy(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    dist = _package(site, "numpy", "License-Expression: BSD-3-Clause", licence="BSD" + _GCC_SECTION)
    (dist / "licenses" / "vendored").mkdir()
    (dist / "licenses" / "vendored" / "LICENSE").write_text("MIT text", encoding="utf-8")
    done = _run(
        site,
        tmp_path,
        _supplied(tmp_path, "numpy-1.0 LICENSE-Apache-2.0 replaces licenses/LICENSE"),
    )
    assert done.returncode == 0, done.stderr
    assert (dist / "licenses" / "LICENSE").read_text(encoding="utf-8") == "Apache License text"
    copy = tmp_path / "out" / "numpy-1.0"
    assert (copy / "LICENSE").read_text(encoding="utf-8") == "Apache License text"
    assert (copy / "vendored" / "LICENSE").read_text(encoding="utf-8") == "MIT text"
    assert not (copy / "LICENSE-Apache-2.0").exists()


def test_refuses_to_replace_a_file_the_dist_info_lacks(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    dist = _package(site, "numpy", "License-Expression: BSD-3-Clause", licence="BSD" + _GCC_SECTION)
    done = _run(
        site,
        tmp_path,
        _supplied(tmp_path, "numpy-1.0 LICENSE-Apache-2.0 replaces licenses/COPYING"),
    )
    assert done.returncode == 1
    assert "numpy-1.0: licenses/COPYING is not a licence file of its dist-info" in done.stderr
    kept = (dist / "licenses" / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in kept


def test_stops_on_a_supplied_line_it_cannot_read(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "ok", "License: MIT")
    done = _run(site, tmp_path, _supplied(tmp_path, "ok-1.0 LICENSE-Apache-2.0 instead LICENSE"))
    assert done.returncode == 1
    assert "cannot read the supplied line" in done.stderr


def test_refuses_a_supplied_file_for_a_package_the_tree_lacks(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "ok", "License: MIT")
    done = _run(site, tmp_path, _supplied(tmp_path, "gone-1.0 LICENSE-Apache-2.0"))
    assert done.returncode == 1
    assert (
        "gone-1.0: a licence file is supplied for a package the tree does not hold" in done.stderr
    )


def test_the_pack_supplies_a_file_that_exists_for_each_named_package() -> None:
    supplied = _SCRIPT.parents[2] / "packaging" / "engine-mlx" / "supplied-licences.txt"
    text = supplied.read_text(encoding="utf-8")
    lines = [line.split() for line in text.splitlines() if line and line[0] != "#"]
    assert lines
    requirements = (supplied.parent / "requirements.txt").read_text(encoding="utf-8")
    for package, path, *replaces in lines:
        assert replaces in ([], ["replaces", "licenses/LICENSE.txt"])
        name, version = package.rsplit("-", 1)
        assert f"\n{name}=={version} \\\n" in requirements
        assert (supplied.parent / path).is_file()


def test_refuses_a_folder_with_no_package(site: pathlib.Path, tmp_path: pathlib.Path) -> None:
    done = _run(site, tmp_path)
    assert done.returncode == 1
    assert "holds no package" in done.stderr


def test_prints_usage_with_the_wrong_argument_count() -> None:
    done = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert done.returncode != 0
    assert "Usage: collect_python_licences.py" in done.stderr


def test_lists_each_package_whose_terms_hold_mpl(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "tqdm", "License: MPL-2.0 AND MIT")
    _package(site, "other", "License-Expression: MPL-2.0")
    _package(site, "plain", "License: MIT")
    mpl = tmp_path / "mpl.txt"
    done = _run(site, tmp_path, mpl=mpl)
    assert done.returncode == 0, done.stderr
    assert mpl.read_text(encoding="utf-8").split() == ["other-1.0", "tqdm-1.0"]


def test_writes_an_empty_mpl_list_when_no_package_uses_mpl(
    site: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _package(site, "plain", "License: MIT")
    mpl = tmp_path / "mpl.txt"
    assert _run(site, tmp_path, mpl=mpl).returncode == 0
    assert mpl.read_text(encoding="utf-8") == ""


_TARGET = "aarch64-apple-darwin"
_BZIP2 = "This program, bzip2, is licensed under a BSD-style licence."
_TCL = "This software is copyrighted by the Regents of the University of California."
_ZLIB = "zlib licence: altered source versions must be plainly marked."


def _extension(
    terms: list[str], paths: list[str], link: str | None = "path_static"
) -> list[dict[str, object]]:
    if link == "system":
        links = [{"name": "lib", "system": True}]
    else:
        links = [{"name": "lib", link: "build/lib/lib.a"}] if link else []
    return [{"variant": "default", "licenses": terms, "license_paths": paths, "links": links}]


def _interpreter(
    tmp: pathlib.Path,
    extensions: dict[str, list[dict[str, object]]],
    texts: dict[str, str],
    *,
    version: str = "3.12.16",
    target: str = _TARGET,
) -> tuple[pathlib.Path, pathlib.Path]:
    """The metadata folder of a full archive and an interpreter tree of the same version."""
    meta = tmp / "pbs" / "python"
    (meta / "licenses").mkdir(parents=True)
    for name, text in texts.items():
        (meta / "licenses" / name).write_text(text, encoding="utf-8")
    info = {
        "python_version": version,
        "target_triple": target,
        "build_info": {"extensions": extensions},
    }
    (meta / "PYTHON.json").write_text(json.dumps(info), encoding="utf-8")
    header = tmp / "tree" / "include" / "python3.12" / "patchlevel.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        '#define PY_RELEASE_LEVEL 0xF\n#define PY_VERSION "3.12.16"\n', encoding="utf-8"
    )
    return meta, tmp / "tree"


def _run_interpreter(
    meta: pathlib.Path, tree: pathlib.Path, tmp: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    (tmp / "terms").write_text("MIT\nPSF-2.0\n", encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(_SCRIPT),
            "--interpreter",
            str(meta),
            str(tree),
            str(tmp / "out"),
            str(tmp / "terms"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_copies_the_libraries_of_the_interpreter_and_adds_their_terms(
    tmp_path: pathlib.Path,
) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {
            "_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"]),
            "_tkinter": _extension(
                ["TCL", "Zlib"], ["licenses/LICENSE.tcl.txt", "licenses/LICENSE.zlib.txt"]
            ),
            "zlib": _extension(["Zlib"], ["licenses/LICENSE.zlib.txt"]),
        },
        {
            "LICENSE.bzip2.txt": _BZIP2,
            "LICENSE.tcl.txt": _TCL,
            "LICENSE.zlib.txt": _ZLIB,
            "LICENSE.unused.txt": "nothing refers to this text",
        },
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 0, done.stderr
    terms = (tmp_path / "terms").read_text(encoding="utf-8").split()
    assert terms == ["MIT", "PSF-2.0", "TCL", "Zlib", "bzip2-1.0.6"]
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "LICENSE.bzip2.txt",
        "LICENSE.tcl.txt",
        "LICENSE.zlib.txt",
    ]
    assert (out / "LICENSE.bzip2.txt").read_text(encoding="utf-8") == _BZIP2


def test_an_extension_that_links_only_the_system_needs_no_licence_text(
    tmp_path: pathlib.Path,
) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {
            "_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"]),
            "_curses": _extension([], [], link=None),
            "_ctypes": _extension([], [], link="system"),
        },
        {"LICENSE.bzip2.txt": _BZIP2},
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("link", ["path_static", "path_dynamic"])
def test_refuses_a_bundled_library_that_has_no_licence_text(
    tmp_path: pathlib.Path, link: str
) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {
            "_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"]),
            "_lzma": _extension(["0BSD"], [], link=link),
        },
        {"LICENSE.bzip2.txt": _BZIP2},
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert "_lzma: links lib and names no licence text" in done.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("texts", "message"),
    [
        ({}, "_bz2: the licence text licenses/LICENSE.bzip2.txt is missing or empty"),
        ({"LICENSE.bzip2.txt": " \n"}, "the licence text licenses/LICENSE.bzip2.txt is missing"),
        (
            {"LICENSE.bzip2.txt": "GNU GENERAL PUBLIC LICENSE\nVersion 3"},
            "_bz2: LICENSE.bzip2.txt holds a GNU licence text",
        ),
        (
            {"LICENSE.bzip2.txt": "bzip2\n" * 600 + "GNU Lesser General Public License version 2"},
            "_bz2: LICENSE.bzip2.txt holds a GNU licence text",
        ),
    ],
)
def test_refuses_a_licence_text_that_is_missing_empty_or_gnu(
    tmp_path: pathlib.Path, texts: dict[str, str], message: str
) -> None:
    meta, tree = _interpreter(
        tmp_path, {"_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"])}, texts
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert message in done.stderr
    assert not (tmp_path / "out").exists()
    assert (tmp_path / "terms").read_text(encoding="utf-8") == "MIT\nPSF-2.0\n"


def test_refuses_a_licence_text_that_names_no_licence(tmp_path: pathlib.Path) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {"_bz2": _extension([], ["licenses/LICENSE.bzip2.txt"])},
        {"LICENSE.bzip2.txt": _BZIP2},
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert "_bz2: names a licence text and no licence" in done.stderr


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": "3.12.17"}, "the tree is Python 3.12.16 and the metadata is for 3.12.17"),
        ({"target": "x86_64-apple-darwin"}, "the metadata is for x86_64-apple-darwin"),
    ],
)
def test_refuses_metadata_that_is_not_for_the_tree(
    tmp_path: pathlib.Path, changes: dict[str, str], message: str
) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {"_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"])},
        {"LICENSE.bzip2.txt": _BZIP2},
        **changes,
    )
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert message in done.stderr


def test_refuses_a_tree_that_has_no_version_header(tmp_path: pathlib.Path) -> None:
    meta, tree = _interpreter(
        tmp_path,
        {"_bz2": _extension(["bzip2-1.0.6"], ["licenses/LICENSE.bzip2.txt"])},
        {"LICENSE.bzip2.txt": _BZIP2},
    )
    (tree / "include" / "python3.12" / "patchlevel.h").write_text("", encoding="utf-8")
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert "the tree is Python None" in done.stderr


def test_refuses_a_folder_with_no_python_json(tmp_path: pathlib.Path) -> None:
    meta, tree = _interpreter(tmp_path, {}, {})
    (meta / "PYTHON.json").unlink()
    done = _run_interpreter(meta, tree, tmp_path)
    assert done.returncode == 1
    assert "holds no PYTHON.json" in done.stderr


def test_prints_usage_with_the_wrong_interpreter_argument_count() -> None:
    done = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), "--interpreter", "one", "two"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert done.returncode != 0
    assert "--interpreter <metadata> <tree> <out> <terms>" in done.stderr
