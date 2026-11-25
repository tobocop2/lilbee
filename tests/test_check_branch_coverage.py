"""Tests for the coverage gate in ``scripts/check_branch_coverage.py``.

The script lives outside the ``lilbee`` package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_branch_coverage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_branch_coverage", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cbc = _load_module()


def _file(lines=(), branches=()):
    return {"missing_lines": list(lines), "missing_branches": [list(b) for b in branches]}


class TestViolations:
    def test_clean_report_has_none(self):
        report = {"files": {"src/lilbee/a.py": _file(), "src/lilbee/cli/tui/app.py": _file()}}
        assert cbc.violations(report) == []

    def test_missed_branch_outside_line_only_packages_fails(self):
        report = {"files": {"src/lilbee/a.py": _file(branches=[(10, 12), (20, -3)])}}
        assert cbc.violations(report) == ["src/lilbee/a.py: missed branches 10->12, 20->-3"]

    def test_missed_branch_in_tui_is_ignored(self):
        report = {"files": {"src/lilbee/cli/tui/app.py": _file(branches=[(10, 12)])}}
        assert cbc.violations(report) == []

    def test_missed_branch_in_fleet_is_ignored(self):
        report = {"files": {"src/lilbee/providers/fleet/client.py": _file(branches=[(1, 2)])}}
        assert cbc.violations(report) == []

    def test_missed_line_fails_everywhere(self):
        report = {
            "files": {
                "src/lilbee/a.py": _file(lines=[4]),
                "src/lilbee/cli/tui/app.py": _file(lines=[7, 8]),
            }
        }
        assert cbc.violations(report) == [
            "src/lilbee/a.py: missed lines [4]",
            "src/lilbee/cli/tui/app.py: missed lines [7, 8]",
        ]

    def test_windows_separators_match_line_only_packages(self):
        report = {"files": {"src\\lilbee\\cli\\tui\\app.py": _file(branches=[(1, 2)])}}
        assert cbc.violations(report) == []

    def test_report_without_branch_data_is_an_error(self):
        report = {"files": {"src/lilbee/a.py": {"missing_lines": []}}}
        with pytest.raises(KeyError):
            cbc.violations(report)


class TestMain:
    def _run(self, tmp_path, report, capsys):
        path = tmp_path / "coverage.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        code = cbc.main(["check_branch_coverage.py", str(path)])
        return code, capsys.readouterr().out

    def test_passes_on_a_clean_report(self, tmp_path, capsys):
        code, out = self._run(tmp_path, {"files": {"src/lilbee/a.py": _file()}}, capsys)
        assert code == 0
        assert "every branch covered outside" in out

    def test_fails_and_names_the_file_on_a_dropped_branch(self, tmp_path, capsys):
        report = {"files": {"src/lilbee/a.py": _file(branches=[(3, 5)])}}
        code, out = self._run(tmp_path, report, capsys)
        assert code == 1
        assert "FAIL src/lilbee/a.py: missed branches 3->5" in out
        assert "1 file(s) below 100%" in out
