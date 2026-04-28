"""QA matrix pytest configuration.

Fixtures here are the contract between scenarios and the runner. Lifecycles
are documented inline; load-bearing for cross-worker isolation under xdist.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from drivers.tui import TuiSession, lilbee_env

_DEFAULT_CHAT_MODEL = "smollm2:135m"
_LANE_ENV_VAR = "LILBEE_QA_LANE"
_BIN_ENV_VAR = "LILBEE_QA_BIN"
_CHAT_MODEL_ENV_VAR = "LILBEE_QA_CHAT_MODEL"


@dataclass(frozen=True)
class Lane:
    """The artifact under test for this run."""

    name: str
    lilbee_bin: str

    @property
    def is_binary(self) -> bool:
        return self.name == "l2-binary"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Assign xdist groups: writers serialize, others group by file.

    Enforces the invariant that writer-marked tests live in dedicated files. A
    file mixing writer and non-writer tests would fork `lilbee serve` twice for
    one file (writer group + file group), defeating the file-scoped fixture.
    """
    files_with_writers = {item.path for item in items if "writer" in item.keywords}
    files_with_non_writers = {item.path for item in items if "writer" not in item.keywords}
    mixed = files_with_writers & files_with_non_writers
    if mixed:
        names = sorted(p.name for p in mixed)
        raise pytest.UsageError(
            f"writer-marked tests must live in dedicated files; found mixed: {names}"
        )

    for item in items:
        if "writer" in item.keywords:
            item.add_marker(pytest.mark.xdist_group("writers"))
        else:
            item.add_marker(pytest.mark.xdist_group(item.path.name))


@pytest.fixture(scope="session")
def qa_chat_model() -> str:
    return os.environ.get(_CHAT_MODEL_ENV_VAR, _DEFAULT_CHAT_MODEL)


@pytest.fixture(scope="session")
def lane() -> Lane:
    name = os.environ.get(_LANE_ENV_VAR, "l1-source")
    explicit = os.environ.get(_BIN_ENV_VAR)
    if explicit:
        bin_path = explicit
    else:
        discovered = shutil.which("lilbee")
        if not discovered:
            pytest.skip(f"lilbee binary not found; set {_BIN_ENV_VAR} or install lilbee")
        bin_path = discovered
    return Lane(name=name, lilbee_bin=bin_path)


@pytest.fixture
def lilbee_data(tmp_path: Path) -> Path:
    """Per-test data directory; isolates LanceDB across xdist workers."""
    data = tmp_path / "lilbee-data"
    data.mkdir()
    return data


def run_lilbee(
    lane: Lane,
    args: list[str],
    *,
    data_dir: Path,
    timeout: float = 60.0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a lilbee CLI command and capture stdout/stderr."""
    return subprocess.run(
        [lane.lilbee_bin, *args],
        env=lilbee_env(data_dir, extra=extra_env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def tui(lane: Lane, lilbee_data: Path) -> Iterator[TuiSession]:
    """Spawn `lilbee` as a TUI in a PTY; tear down on exit."""
    session = TuiSession([lane.lilbee_bin], env=lilbee_env(lilbee_data))
    try:
        yield session
    finally:
        session.close()
