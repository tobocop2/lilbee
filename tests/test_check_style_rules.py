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
