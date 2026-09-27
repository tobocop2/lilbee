"""Tests for scripts/check_community_profiles.py, the ``make lint`` gate over
profiles/community/."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_community_profiles.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_community_profiles", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ccp = _load_module()


def test_empty_folder_has_no_findings(tmp_path):
    assert ccp.check(tmp_path) == []


def test_a_bad_file_is_reported_with_its_path(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[values]\nchunk_size = 900\n", encoding="utf-8")
    assert ccp.check(tmp_path) == [f"{bad}: A community profile needs tested_on"]


def test_a_good_file_has_no_findings(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text(
        '[profile]\ntested_on = "100 files"\n[values]\nchunk_size = 900\n', encoding="utf-8"
    )
    assert ccp.check(tmp_path) == []


def test_main_exits_1_on_a_bad_file_and_0_on_an_empty_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ccp, "COMMUNITY_DIR", tmp_path)
    assert ccp.main() == 0
    bad = tmp_path / "bad.toml"
    bad.write_text("[values]\nchunk_size = 900\n", encoding="utf-8")
    assert ccp.main() == 1
    out = capsys.readouterr().out
    assert f"{bad}: A community profile needs tested_on" in out
