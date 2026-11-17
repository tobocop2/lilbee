"""The helper that finds the engine-packs workflow fails in CI and skips elsewhere."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.outcomes import Failed, Skipped

from tests._pack_workflow import WORKFLOW_ENV, pack_workflow


def test_the_helper_returns_the_named_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKFLOW_ENV, "/packs/build-packs.yml")
    monkeypatch.setenv("CI", "true")
    assert pack_workflow() == Path("/packs/build-packs.yml")


@pytest.mark.parametrize("value", [None, ""])
def test_the_helper_fails_in_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.setenv("CI", "true")
    if value is None:
        monkeypatch.delenv(WORKFLOW_ENV, raising=False)
    else:
        monkeypatch.setenv(WORKFLOW_ENV, value)
    with pytest.raises(Failed, match=WORKFLOW_ENV):
        pack_workflow()


def test_the_helper_skips_outside_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(WORKFLOW_ENV, raising=False)
    with pytest.raises(Skipped, match=WORKFLOW_ENV):
        pack_workflow()
