"""Tests for the diff-scoped code-smell gate in ``scripts/check_style_rules.py``.

The script lives outside the ``lilbee`` package (not import-installed), so it is
loaded by path. Only the pure functions are exercised here; the git plumbing
(`_smell_base_ref`, `_git_diff_src`) is environment-dependent and covered by
running ``make lint`` itself.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_style_rules.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_style_rules", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csr = _load_module()


class TestParseAddedLines:
    def test_tracks_path_and_new_line_numbers(self):
        diff = (
            "diff --git a/src/x.py b/src/x.py\n"
            "--- a/src/x.py\n"
            "+++ b/src/x.py\n"
            "@@ -10,0 +11,2 @@ def f():\n"
            "+    first = 1\n"
            "+    second = 2\n"
        )
        assert list(csr._parse_added_lines(diff)) == [
            ("src/x.py", 11, "    first = 1"),
            ("src/x.py", 12, "    second = 2"),
        ]

    def test_minus_lines_do_not_advance_counter(self):
        diff = "+++ b/src/x.py\n@@ -20,2 +25,1 @@\n-old one\n-old two\n+replacement\n"
        assert list(csr._parse_added_lines(diff)) == [("src/x.py", 25, "replacement")]

    def test_deletion_target_is_skipped(self):
        diff = "+++ /dev/null\n@@ -1,1 +0,0 @@\n+ignored\n"
        assert list(csr._parse_added_lines(diff)) == []

    def test_single_line_hunk_header_without_count(self):
        diff = "+++ b/src/x.py\n@@ -3 +7 @@\n+added\n"
        assert list(csr._parse_added_lines(diff)) == [("src/x.py", 7, "added")]


class TestCheckCodeSmells:
    def test_flags_getattr_with_default(self):
        added = [("src/x.py", 11, '    content = getattr(result, "content", None)')]
        findings = list(csr._check_code_smells(added))
        assert len(findings) == 1
        assert "src/x.py:11" in findings[0]
        assert "getattr-with-default" in findings[0]

    def test_skips_dunder_getattr(self):
        added = [("src/x.py", 5, '    klass = getattr(obj, "__class__", None)')]
        assert list(csr._check_code_smells(added)) == []

    def test_flags_getattr_self(self):
        added = [("src/x.py", 1, '        self._flag = getattr(self, "_flag", False)')]
        findings = list(csr._check_code_smells(added))
        assert len(findings) == 1
        assert "getattr-by-name" in findings[0]

    def test_flags_type_ignore_attr_defined(self):
        added = [("src/x.py", 2, "        self.app.task_bar  # type: ignore[attr-defined]")]
        findings = list(csr._check_code_smells(added))
        assert "type-ignore on an owned attribute" in findings[0]

    def test_flags_host_narrowing(self):
        added = [("src/x.py", 3, "        if isinstance(self.app, LilbeeApp):")]
        findings = list(csr._check_code_smells(added))
        assert "production host-narrowing" in findings[0]

    @pytest.mark.parametrize("field", ["task", "kind", "role", "event_type", "status", "mode"])
    def test_flags_string_typed_closed_set(self, field):
        added = [("src/x.py", 4, f"    {field}: str = default")]
        findings = list(csr._check_code_smells(added))
        assert "string-typed closed set" in findings[0]

    def test_flags_module_level_global(self):
        added = [("src/x.py", 6, "    global _cache")]
        findings = list(csr._check_code_smells(added))
        assert "module-level mutable global" in findings[0]

    def test_allow_smell_tag_opts_out(self):
        added = [
            (
                "src/x.py",
                7,
                '    x = getattr(row, "field", None)  # style-check: allow-smell',
            )
        ]
        assert list(csr._check_code_smells(added)) == []

    def test_non_python_added_lines_ignored(self):
        added = [("docs/x.md", 1, '    getattr(result, "content", None)')]
        assert list(csr._check_code_smells(added)) == []

    def test_clean_line_yields_nothing(self):
        added = [("src/x.py", 1, "    return result.content")]
        assert list(csr._check_code_smells(added)) == []


class TestUnspecifiedEncoding:
    """Text file I/O must name its encoding, or it decodes as the locale's.

    The bug is invisible on macOS and Linux (UTF-8 locales) and raises
    UnicodeDecodeError on Windows, so only one CI cell in nine ever goes red.
    """

    def _findings(self, tmp_path: Path, source: str) -> list[str]:
        target = tmp_path / "sample.py"
        target.write_text(source, encoding="utf-8")
        return list(csr._check_unspecified_encoding([target]))

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("Path('a').read_text()", id="read_text"),
            pytest.param("Path('a').write_text('x')", id="write_text"),
            pytest.param("Path('a').open()", id="path_open"),
            pytest.param("open('a')", id="builtin_open"),
            pytest.param("open('a', 'w')", id="builtin_open_text_mode"),
            pytest.param("tempfile.NamedTemporaryFile(mode='w')", id="named_temporary_file"),
        ],
    )
    def test_flags_text_io_without_encoding(self, tmp_path: Path, source: str) -> None:
        assert self._findings(tmp_path, source), f"should have flagged: {source}"

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("Path('a').read_text(encoding='utf-8')", id="read_text_encoded"),
            pytest.param("open('a', encoding='utf-8')", id="open_encoded"),
            pytest.param("Path('a').read_bytes()", id="read_bytes"),
            pytest.param("Path('a').write_bytes(b'x')", id="write_bytes"),
            pytest.param("open('a', 'rb')", id="binary_positional"),
            pytest.param("open('a', mode='wb')", id="binary_keyword"),
            pytest.param("tempfile.NamedTemporaryFile(mode='wb')", id="temp_binary"),
            pytest.param("socket.open()", id="unrelated_open_receiver"),
        ],
    )
    def test_leaves_encoded_and_binary_io_alone(self, tmp_path: Path, source: str) -> None:
        assert not self._findings(tmp_path, source), f"should not have flagged: {source}"

    def test_reports_path_and_line(self, tmp_path: Path) -> None:
        findings = self._findings(tmp_path, "x = 1\ny = Path('a').read_text()\n")
        assert len(findings) == 1
        assert ":2:" in findings[0]
        assert "encoding" in findings[0]

    def test_a_syntactically_invalid_file_is_skipped_not_crashed(self, tmp_path: Path) -> None:
        # The checker runs over the whole tree in make lint; one unparsable file
        # must not take the entire gate down with a SyntaxError.
        assert self._findings(tmp_path, "def broken(:\n") == []

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("os.open(os.devnull, os.O_WRONLY)", id="os_open_fd"),
            pytest.param("webbrowser.open('https://example.com')", id="webbrowser_open"),
            pytest.param("zf.open(name)", id="zipfile_member"),
        ],
    )
    def test_leaves_open_homonyms_alone(self, tmp_path: Path, source: str) -> None:
        """These take no encoding at all, so flagging them is unresolvable."""
        assert not self._findings(tmp_path, source), f"should not have flagged: {source}"

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("log_path.open('a')", id="path_variable_text_append"),
            pytest.param("fault_path.open('w')", id="path_variable_text_write"),
        ],
    )
    def test_flags_a_path_variable_opened_in_a_text_mode(self, tmp_path: Path, source: str) -> None:
        """The mode string is what identifies a file open on an unresolved name."""
        assert self._findings(tmp_path, source), f"should have flagged: {source}"


class TestDiffScopedEncodingCheck:
    """The check runs on added lines, so it must survive paths that are gone.

    A branch that adds a file and then deletes it still has added lines in the
    diff while the path no longer exists on disk, and make lint must not die on
    it. ``scripts/`` is outside the coverage gate's scope, so nothing else would
    catch a regression here.
    """

    @pytest.mark.parametrize(
        "rel_path",
        [
            pytest.param("src/lilbee/deleted_on_this_branch.py", id="path_removed"),
            pytest.param("src", id="path_is_a_directory"),
        ],
    )
    def test_an_unreadable_path_yields_nothing_instead_of_raising(self, rel_path: str) -> None:
        assert list(csr._check_new_unspecified_encoding([(rel_path, 1, "x")])) == []

    def test_it_reports_only_the_lines_the_branch_added(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two offences in the file, one of them added: only that one is reported."""
        monkeypatch.setattr(csr, "REPO_ROOT", tmp_path)
        (tmp_path / "sample.py").write_text(
            "Path('a').read_text()\nPath('b').read_text()\n", encoding="utf-8"
        )

        findings = list(csr._check_new_unspecified_encoding([("sample.py", 2, "x")]))

        assert len(findings) == 1
        assert ":2:" in findings[0]
