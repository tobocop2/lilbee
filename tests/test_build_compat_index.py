"""Regression tests for the fork wheel-release PEP 503 index generator.

One script serves lilbee.sh/compat/ from one or more --source specs: the
lancedb fork (tag filter "compat") and, temporarily, the crawlberg fork
(tag filter "+lilbee"). Each source writes its own project directory under
compat/ without touching another source's directory; the root compat/
index.html lists every project directory present afterward. These tests
exercise the parameterization with `gh api` mocked out, so no network call
is made. Byte-identical output for the unmodified lancedb project dir
against the real tobocop2/lancedb releases is proved separately, outside
the test suite (it needs network access).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_compat_index.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_compat_index", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: dataclasses.dataclass resolves its
    # string annotations (from __future__ import annotations, in the script)
    # via sys.modules[cls.__module__], and crashes on a module that is not
    # there yet. This is the standard recipe for loading a module from a
    # file path (see importlib docs, "Importing a source file directly").
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bci = _load()


def test_jq_filter_embeds_the_tag_filter_as_a_literal_substring() -> None:
    # "+lilbee" has a regex metacharacter; contains() must treat it literally,
    # not as "one or more of the preceding token".
    jq = bci._jq_filter("+lilbee")
    assert 'contains("+lilbee")' in jq
    assert 'select(.tag_name | contains("+lilbee"))' in jq


def test_fetch_wheels_passes_repo_and_jq_filter_to_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_check_output(cmd: list[str], text: bool = False) -> str:
        captured["cmd"] = cmd
        return ""

    monkeypatch.setattr(bci.subprocess, "check_output", fake_check_output)

    bci.fetch_wheels("tobocop2/crawlberg", "+lilbee")

    cmd = captured["cmd"]
    assert cmd[:4] == ["gh", "api", "--paginate", "repos/tobocop2/crawlberg/releases"]
    assert cmd[-1] == bci._jq_filter("+lilbee")


def test_fetch_wheels_parses_name_url_and_strips_sha256_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = "\n".join(
        [
            json.dumps({"name": "a.whl", "url": "https://x/a.whl", "digest": "sha256:aaa"}),
            "",
            json.dumps({"name": "b.whl", "url": "https://x/b.whl", "digest": "sha256:bbb"}),
        ]
    )
    monkeypatch.setattr(bci.subprocess, "check_output", lambda cmd, text=False: lines)

    wheels = bci.fetch_wheels("tobocop2/lancedb", "compat")

    assert wheels == [
        ("a.whl", "https://x/a.whl", "aaa"),
        ("b.whl", "https://x/b.whl", "bbb"),
    ]


def test_fetch_wheels_raises_on_missing_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    line = json.dumps({"name": "a.whl", "url": "https://x/a.whl", "digest": None})
    monkeypatch.setattr(bci.subprocess, "check_output", lambda cmd, text=False: line)

    with pytest.raises(SystemExit, match=r"a\.whl has no sha256 digest"):
        bci.fetch_wheels("tobocop2/lancedb", "compat")


def test_parse_source_reads_all_fields() -> None:
    spec = "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn"
    source = bci._parse_source(spec)
    assert source == bci.Source(
        repo="tobocop2/crawlberg",
        tag_filter="+lilbee",
        project="crawlberg",
        on_missing=bci.OnMissing.WARN,
    )
    # on_missing is the enum, not a bare string that merely compares equal to it.
    assert source.on_missing is bci.OnMissing.WARN


def test_parse_source_defaults_on_missing_to_fail() -> None:
    source = bci._parse_source("repo=tobocop2/lancedb,tag-filter=compat,project=lancedb")
    assert source.on_missing is bci.OnMissing.FAIL


def test_parse_source_rejects_a_non_key_value_entry() -> None:
    with pytest.raises(Exception, match=r"not key=value"):
        bci._parse_source("repo=tobocop2/lancedb,oops,project=lancedb,tag-filter=compat")


def test_parse_source_rejects_a_missing_required_field() -> None:
    with pytest.raises(Exception, match=r"missing"):
        bci._parse_source("repo=tobocop2/lancedb,tag-filter=compat")


def test_parse_source_rejects_an_invalid_on_missing_value() -> None:
    with pytest.raises(Exception, match=r"on-missing must be fail or warn"):
        bci._parse_source(
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb,on-missing=ignore"
        )


def test_parse_source_rejects_an_unknown_key() -> None:
    with pytest.raises(Exception, match=r"unknown key 'branch'"):
        bci._parse_source("repo=tobocop2/lancedb,tag-filter=compat,project=lancedb,branch=main")


def test_parse_source_rejects_a_repeated_key() -> None:
    with pytest.raises(Exception, match=r"key 'project' is repeated"):
        bci._parse_source("repo=tobocop2/lancedb,tag-filter=compat,project=lancedb,project=other")


@pytest.mark.parametrize(
    "project",
    [
        pytest.param("<proj>", id="angle-brackets"),  # the exact name that broke Windows CI
        pytest.param("LanceDB", id="uppercase"),
        pytest.param("lance_db", id="underscore"),
        pytest.param("lance.db", id="dot"),
        pytest.param("-lancedb", id="leading-hyphen"),
        pytest.param("lancedb-", id="trailing-hyphen"),
        pytest.param("lance--db", id="doubled-hyphen"),
        pytest.param("lance db", id="space"),
        pytest.param("", id="empty"),
    ],
)
def test_parse_source_rejects_a_project_name_that_is_not_pep503_normalized(project: str) -> None:
    with pytest.raises(Exception, match=r"is not a PEP 503 normalized"):
        bci._parse_source(f"repo=tobocop2/lancedb,tag-filter=compat,project={project}")


@pytest.mark.parametrize("project", ["lancedb", "crawlberg", "a", "a-b-2"])
def test_parse_source_accepts_a_pep503_normalized_project_name(project: str) -> None:
    source = bci._parse_source(f"repo=tobocop2/lancedb,tag-filter=compat,project={project}")
    assert source.project == project


def test_missing_fail_mode_returns_1_without_a_warning_annotation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert bci._missing(bci.OnMissing.FAIL, "crawlberg", "no wheels found") == 1
    err = capsys.readouterr().err
    assert err == "no wheels found\n"
    assert "::warning::" not in err


def test_missing_warn_mode_returns_0_with_a_warning_annotation_naming_the_project(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert bci._missing(bci.OnMissing.WARN, "crawlberg", "no wheels found") == 0
    err = capsys.readouterr().err
    assert err == "::warning::no wheels found; skipping crawlberg\n"


def test_main_requires_at_least_one_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["build_compat_index.py", str(tmp_path)])

    with pytest.raises(SystemExit):
        bci.main()


def test_main_single_source_writes_its_project_dir_and_the_root_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        calls.append((repo, tag_filter))
        return [("b-1.whl", "https://x/b-1.whl", "bbb"), ("a-1.whl", "https://x/a-1.whl", "aaa")]

    monkeypatch.setattr(bci, "fetch_wheels", fake_fetch_wheels)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
        ],
    )

    assert bci.main() == 0
    assert calls == [("tobocop2/lancedb", "compat")]

    project_index = (tmp_path / "compat" / "lancedb" / "index.html").read_text(encoding="utf-8")
    assert "<h1>Links for lancedb</h1>" in project_index
    # Sorted by (name, url, sha): a-1.whl before b-1.whl.
    assert project_index.index("a-1.whl") < project_index.index("b-1.whl")
    assert 'href="https://x/a-1.whl#sha256=aaa"' in project_index

    top_index = (tmp_path / "compat" / "index.html").read_text(encoding="utf-8")
    assert top_index == (
        '<!DOCTYPE html>\n<html><body>\n    <a href="lancedb/">lancedb</a><br/>\n</body></html>\n'
    )


def test_main_two_sources_write_both_project_dirs_and_list_both_on_the_root_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        if "crawlberg" in repo:
            return [("crawlberg-1.8.0.whl", "https://x/crawlberg-1.8.0.whl", "ccc")]
        return [("lancedb-1.whl", "https://x/lancedb-1.whl", "lll")]

    monkeypatch.setattr(bci, "fetch_wheels", fake_fetch_wheels)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn",
        ],
    )

    assert bci.main() == 0

    lancedb_index = (tmp_path / "compat" / "lancedb" / "index.html").read_text(encoding="utf-8")
    assert "lancedb-1.whl" in lancedb_index
    crawlberg_index = (tmp_path / "compat" / "crawlberg" / "index.html").read_text(encoding="utf-8")
    assert "crawlberg-1.8.0.whl" in crawlberg_index

    top_index = (tmp_path / "compat" / "index.html").read_text(encoding="utf-8")
    assert '<a href="crawlberg/">crawlberg</a>' in top_index
    assert '<a href="lancedb/">lancedb</a>' in top_index
    # Sorted: crawlberg before lancedb.
    assert top_index.index("crawlberg/") < top_index.index("lancedb/")


def test_duplicate_project_finds_the_repeated_name() -> None:
    lancedb = bci.Source(repo="tobocop2/lancedb", tag_filter="compat", project="lancedb")
    crawlberg = bci.Source(repo="tobocop2/crawlberg", tag_filter="+lilbee", project="crawlberg")
    again = bci.Source(repo="tobocop2/crawlberg-fork2", tag_filter="x", project="lancedb")

    assert bci._duplicate_project([lancedb, crawlberg]) is None
    assert bci._duplicate_project([lancedb, crawlberg, again]) == "lancedb"


def test_main_rejects_two_sources_with_the_same_project_before_fetching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The design's whole point is that one source can never clobber another's
    project directory. A same-name collision must be rejected outright, not
    left to "whichever source runs last wins"."""
    calls: list[str] = []

    def fake_fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        calls.append(repo)
        return [("x-1.whl", "https://x/x-1.whl", "xxx")]

    monkeypatch.setattr(bci, "fetch_wheels", fake_fetch_wheels)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=lancedb,on-missing=warn",
        ],
    )

    assert bci.main() == 1
    err = capsys.readouterr().err
    assert "--source project 'lancedb' is used by more than one source" in err
    # Rejected before either source is fetched, let alone written.
    assert calls == []
    assert not (tmp_path / "compat").exists()


def test_main_crawlberg_on_missing_warn_leaves_lancedb_alone_and_off_the_root_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        if "crawlberg" in repo:
            return []
        return [("lancedb-1.whl", "https://x/lancedb-1.whl", "lll")]

    monkeypatch.setattr(bci, "fetch_wheels", fake_fetch_wheels)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn",
        ],
    )

    assert bci.main() == 0
    err = capsys.readouterr().err
    assert "::warning::no crawlberg wheels found in tobocop2/crawlberg releases" in err
    assert not (tmp_path / "compat" / "crawlberg").exists()
    assert (tmp_path / "compat" / "lancedb" / "index.html").exists()

    top_index = (tmp_path / "compat" / "index.html").read_text(encoding="utf-8")
    assert "crawlberg" not in top_index
    assert '<a href="lancedb/">lancedb</a>' in top_index


def test_main_lancedb_on_missing_fail_stops_before_the_crawlberg_source_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def fake_fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        calls.append(repo)
        if "lancedb" in repo:
            return []
        return [("crawlberg-1.8.0.whl", "https://x/crawlberg-1.8.0.whl", "ccc")]

    monkeypatch.setattr(bci, "fetch_wheels", fake_fetch_wheels)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn",
        ],
    )

    assert bci.main() == 1
    err = capsys.readouterr().err
    assert "no lancedb wheels found in tobocop2/lancedb releases" in err
    assert "::warning::" not in err
    assert calls == ["tobocop2/lancedb"]
    assert not (tmp_path / "compat" / "crawlberg").exists()
    assert not (tmp_path / "compat" / "index.html").exists()


def test_main_gh_api_failure_fails_by_default_without_a_warning_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_gh_error(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        raise bci.subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(bci, "fetch_wheels", raise_gh_error)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
        ],
    )

    assert bci.main() == 1
    err = capsys.readouterr().err
    assert "gh api failed for tobocop2/lancedb" in err
    assert "::warning::" not in err


def test_main_on_missing_warn_skips_the_project_dir_when_gh_api_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_gh_error(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        raise bci.subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(bci, "fetch_wheels", raise_gh_error)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn",
        ],
    )

    assert bci.main() == 0
    err = capsys.readouterr().err
    assert "::warning::gh api failed for tobocop2/crawlberg" in err
    assert not (tmp_path / "compat" / "crawlberg").exists()


def test_main_on_missing_warn_does_not_swallow_a_missing_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--on-missing warn only degrades 'no release'/'gh failed'; a broken release
    (an asset with no digest) is a worse failure and still aborts the build."""

    def raise_missing_digest(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
        raise SystemExit("a.whl has no sha256 digest from the API")

    monkeypatch.setattr(bci, "fetch_wheels", raise_missing_digest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/crawlberg,tag-filter=+lilbee,project=crawlberg,on-missing=warn",
        ],
    )

    with pytest.raises(SystemExit, match=r"a\.whl has no sha256 digest"):
        bci.main()


def test_main_escapes_html_in_wheel_name_and_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # project is validated to a PEP 503 normalized name (see the
    # _parse_source rejection tests below), so it can never carry HTML --
    # name and url come straight from the fork's GitHub releases, with no
    # such validation, and are the fields that still need escaping.
    wheel = ("<script>a.whl", 'https://x/a.whl?x="y"', "aaa")
    monkeypatch.setattr(bci, "fetch_wheels", lambda repo, tag_filter: [wheel])
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
        ],
    )

    assert bci.main() == 0

    project_index = (tmp_path / "compat" / "lancedb" / "index.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;a.whl" in project_index
    assert "<script>a.whl" not in project_index
    assert "x=&quot;y&quot;" in project_index
    assert '?x="y"' not in project_index


def test_main_returns_1_and_warns_when_no_wheels_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bci, "fetch_wheels", lambda repo, tag_filter: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compat_index.py",
            str(tmp_path),
            "--source",
            "repo=tobocop2/lancedb,tag-filter=compat,project=lancedb",
        ],
    )

    assert bci.main() == 1
    expected = "no lancedb wheels found in tobocop2/lancedb releases"
    assert expected in capsys.readouterr().err
