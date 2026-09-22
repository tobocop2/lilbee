"""Every release cell installs the dependency set uv.lock records.

Cells that run `uv sync` get that for free. The Intel Mac cell cannot run
`uv sync`, because the locked lancedb ships no macosx_x86_64 wheel, so it
installs with `uv pip install` and needs a constraints file exported from the
lock. Without one it resolved fresh from PyPI on every run: eleven packages
moved between two releases with no change to the repository.

Every install in that cell has to carry the constraints, not just the one that
names the release extra. An unconstrained install of lancedb alone still drags
numpy, pyarrow, pydantic and tqdm in at whatever PyPI serves that day.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yml"

# lancedb is resolved from the +compat index instead, because the version the
# lock records is the one with no wheel for this platform. lilbee-engine is a
# local path, which uv rejects as a constraint.
_UNPINNED_PACKAGES = {"lancedb"}
_UNNAMED_PACKAGES = {"lilbee-engine"}


def _run_scripts() -> list[str]:
    workflow = yaml.safe_load(_RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    return [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    ]


def _commands(script: str) -> list[list[str]]:
    """Split a run block into commands, folding shell line continuations."""
    folded = script.replace("\\\n", " ")
    return [
        part.split() for line in folded.splitlines() for part in line.split("&&") if part.strip()
    ]


def _release_extra_scripts() -> list[str]:
    return [
        script for script in _run_scripts() if "uv pip install" in script and ".[release]" in script
    ]


def _flag_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def _export(script: str) -> list[str]:
    exports = [c for c in _commands(script) if c[:2] == ["uv", "export"]]
    assert len(exports) == 1, "the cell needs exactly one export of the lock"
    return exports[0]


def _installs(script: str) -> list[list[str]]:
    return [c for c in _commands(script) if c[:3] == ["uv", "pip", "install"]]


def test_the_release_extra_is_installed_from_a_constraints_file() -> None:
    """An unconstrained `uv pip install '.[release]'` resolves from PyPI."""
    scripts = _release_extra_scripts()

    assert scripts, "no cell installs the release extra with uv pip install"
    for script in scripts:
        constraints = _flag_value(_export(script), "-o")

        assert constraints is not None, "the export writes no constraints file"
        named = [c for c in _installs(script) if any(".[release]" in w for w in c)]
        assert named, "no install names the release extra"
        for command in named:
            assert _flag_value(command, "--constraint") == constraints


def test_every_install_in_the_cell_carries_the_constraints() -> None:
    """lancedb's own closure drifts if it is installed on its own."""
    for script in _release_extra_scripts():
        constraints = _flag_value(_export(script), "-o")

        for command in _installs(script):
            assert _flag_value(command, "--constraint") == constraints, (
                f"{' '.join(command)} installs without the lock's constraints; "
                "every package it pulls in resolves from PyPI"
            )


def test_the_export_covers_the_extra_the_cell_installs() -> None:
    """Constraints exported without the extra miss most of what is installed."""
    for script in _release_extra_scripts():
        assert _flag_value(_export(script), "--extra") == "release"


def test_lancedb_is_the_only_package_held_out_of_the_constraints() -> None:
    """A package dropped from the export is a package free to drift."""
    for script in _release_extra_scripts():
        export = _export(script)
        excluded = {
            export[index + 1]
            for index, word in enumerate(export[:-1])
            if word == "--no-emit-package"
        }

        assert excluded - _UNNAMED_PACKAGES == _UNPINNED_PACKAGES, (
            f"the export drops {sorted(excluded)} from the lock; every name "
            "here resolves from PyPI and needs a reason in the workflow comment"
        )
