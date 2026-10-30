"""The requirement table in code, the golden files and docs/engine-integration.md say the same thing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conformance.harness import GOLDEN_DIR, REQUIREMENTS, golden

DOC = Path(__file__).resolve().parents[2] / "docs" / "engine-integration.md"
SPIKE_ROWS = range(1, 56)


def _section(text: str, requirement: str) -> str:
    match = re.search(rf"^### {requirement}\b.*?(?=^### |^## |\Z)", text, re.M | re.S)
    assert match, f"no section for {requirement}"
    return match.group(0)


def test_each_golden_file_belongs_to_a_requirement() -> None:
    stems = sorted(p.stem for p in GOLDEN_DIR.glob("*.json"))
    assert stems
    assert stems == sorted(REQUIREMENTS)


@pytest.mark.parametrize("requirement", sorted(REQUIREMENTS))
def test_golden_file_names_its_requirement_and_states_an_assertion(requirement: str) -> None:
    data = golden(requirement)
    assert data["id"] == requirement
    assert data["assert"]
    assert all(isinstance(row, int) and row in SPIKE_ROWS for row in data["rows"])


def test_tiers_follow_the_id_prefix() -> None:
    tiers = {key: req.tier for key, req in REQUIREMENTS.items()}
    assert tiers
    assert all(tier == (1 if key.startswith("E") else 2) for key, tier in tiers.items())


@pytest.mark.parametrize("requirement", sorted(REQUIREMENTS))
def test_document_cites_the_rows_of_each_golden_file(requirement: str) -> None:
    section = _section(DOC.read_text(encoding="utf-8"), requirement)
    evidence = next(line for line in section.splitlines() if line.startswith("Evidence:"))
    cited = {int(n) for n in re.findall(r"\brow (\d+)\b", evidence)}
    assert cited == set(golden(requirement)["rows"])


def test_document_maps_every_spike_row() -> None:
    text = DOC.read_text(encoding="utf-8")
    missing = [n for n in SPIKE_ROWS if not re.search(rf"^\|\s*{n}\s*\|", text, re.M)]
    assert missing == []


def test_document_names_the_ci_flag_of_each_requirement() -> None:
    text = DOC.read_text(encoding="utf-8")
    real_only = {key for key, req in REQUIREMENTS.items() if not req.ci}
    assert real_only == {"E11"}
    assert "real engine only" in _section(text, "E11")
