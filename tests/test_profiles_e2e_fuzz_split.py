"""Each hostile profile input is its own test, so no one test runs them all under one timeout."""

from __future__ import annotations

from typing import Any

import pytest

import tests.test_profiles_e2e_fuzz as fuzz

# generated test name -> the hostile inputs it ran as explicit examples before the split
EXPECTED_CASES: dict[str, tuple[Any, ...]] = {
    "test_new_names_through_the_cli": fuzz.HOSTILE_NAMES,
    "test_new_names_through_http": fuzz.HOSTILE_NAMES,
    "test_new_names_through_mcp": fuzz.HOSTILE_NAMES,
    "test_profile_files_through_the_cli": (
        *fuzz.HOSTILE_BYTES,
        *(text.encode("utf-8") for text in fuzz.HOSTILE_TEXTS),
    ),
    "test_profile_uploads_through_http": (
        *((fuzz.VALID_TEXT, name) for name in fuzz.HOSTILE_FILENAMES),
        *((text, "upload.toml") for text in fuzz.HOSTILE_TEXTS),
    ),
    "test_profile_uploads_through_mcp": (
        *((fuzz.VALID_TEXT, name) for name in fuzz.HOSTILE_FILENAMES),
        *((text, "upload.toml") for text in fuzz.HOSTILE_TEXTS),
    ),
    "test_analyze_folders_through_the_cli": fuzz.HOSTILE_FOLDERS,
    "test_analyze_folders_through_http": fuzz.HOSTILE_FOLDERS,
    "test_analyze_folders_through_mcp": fuzz.HOSTILE_FOLDERS,
}


def _parametrized_cases(test: Any) -> list[Any]:
    (mark,) = [m for m in test.pytestmark if m.name == "parametrize"]
    names, values = mark.args
    return [value if isinstance(names, str) else tuple(value) for value in values]


@pytest.mark.parametrize("generated", sorted(EXPECTED_CASES))
def test_every_hostile_input_is_a_parametrized_case(generated: str) -> None:
    hostile = getattr(fuzz, generated.replace("test_", "test_hostile_", 1))
    assert _parametrized_cases(hostile) == list(EXPECTED_CASES[generated])


@pytest.mark.parametrize("generated", sorted(EXPECTED_CASES))
def test_the_generated_test_keeps_no_hostile_example(generated: str) -> None:
    assert getattr(getattr(fuzz, generated), "hypothesis_explicit_examples", []) == []
