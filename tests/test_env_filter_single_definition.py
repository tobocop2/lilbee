"""The LILBEE_ environment filter has one definition, keyed on the config prefix."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests import conftest

TESTS_DIR = Path(__file__).parent
INLINE_FILTER = re.compile(r"if not \w+\.startswith\(\s*[\"']LILBEE_[\"']\s*\)")


def test_filter_follows_config_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the config prefix changes what the helper strips."""
    monkeypatch.setenv("LILBEE_X_KEEP", "1")
    monkeypatch.setenv("OTHERPFX_X_DROP", "1")
    monkeypatch.setattr(conftest, "ENV_PREFIX", "OTHERPFX_", raising=False)
    env = conftest.without_lilbee_env()
    assert "OTHERPFX_X_DROP" not in env
    assert env["LILBEE_X_KEEP"] == "1"


def test_clean_env_strips_prefixed_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILBEE_X_DROP", "1")
    monkeypatch.setenv("UNRELATED_X_KEEP", "1")
    env = conftest.clean_env()
    assert "LILBEE_X_DROP" not in env
    assert env["UNRELATED_X_KEEP"] == "1"


def test_no_inline_copy_of_the_filter() -> None:
    """Only the shared helper may spell the filter out."""
    offenders = [
        str(path.relative_to(TESTS_DIR))
        for path in TESTS_DIR.rglob("*.py")
        if path.name not in {"conftest.py", Path(__file__).name}
        and INLINE_FILTER.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
