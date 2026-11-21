"""The integration target runs the directory of tests that need the client library."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_integration_target_runs_the_integration_directory() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^test-integration:\n\t(.+)$", makefile, re.MULTILINE)
    assert match, "Makefile has no test-integration target"
    assert "tests/integration/" in match.group(1).split()


@pytest.mark.parametrize(
    "parser_test", ["tests/test_estimator_real_binary.py", "tests/test_gguf_meta.py"]
)
def test_integration_target_runs_the_tests_that_need_the_real_parser(parser_test: str) -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^test-integration:\n\t(.+)$", makefile, re.MULTILINE)
    assert match, "Makefile has no test-integration target"
    assert parser_test in match.group(1).split()
    assert (ROOT / parser_test).is_file()


def test_the_integration_job_requires_the_parser_it_builds() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["integration"]["steps"]
    run_step = next(step for step in steps if step.get("name") == "Run integration tests")
    assert run_step["run"] == "make test-integration"
    assert run_step["env"]["LILBEE_REQUIRE_GGUF_PARSER"] == "1"


def test_check_runs_pytest_on_the_preset_replay() -> None:
    replay = "tests/integration/test_preset_fixtures.py"
    dry_run = subprocess.run(
        ["make", "-n", "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    commands = [line.split() for line in dry_run.stdout.splitlines()]
    assert any(words[:3] == ["uv", "run", "pytest"] and replay in words for words in commands), (
        "make check does not run pytest on the preset replay"
    )
    assert (ROOT / replay).is_file()
