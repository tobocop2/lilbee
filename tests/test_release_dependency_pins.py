"""Every release cell installs the dependency set uv.lock records.

Cells that run `uv sync` get that for free. The Intel Mac cell cannot run
`uv sync`, because the locked lancedb ships no macosx_x86_64 wheel, so it
installs with `uv pip install` and needs a constraints file exported from the
lock. Without one it resolved fresh from PyPI on every run: eleven packages
moved between two releases with no change to the repository.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yml"

# lancedb is resolved from the +compat index instead, because the version the
# lock records is the one with no wheel for this platform.
_UNPINNED_PACKAGES = {"lancedb"}


def _run_scripts() -> list[str]:
    workflow = yaml.safe_load(_RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    return [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    ]


def _project_installs() -> list[str]:
    return [
        script for script in _run_scripts() if "uv pip install" in script and ".[release]" in script
    ]


def test_every_pip_installed_cell_constrains_the_release_extra_to_the_lock() -> None:
    """An unconstrained `uv pip install '.[release]'` resolves from PyPI."""
    installs = _project_installs()

    assert installs, "no cell installs the release extra with uv pip install"
    for script in installs:
        assert "uv export --frozen" in script, (
            "a cell installs the release extra without exporting uv.lock first; "
            "it will resolve fresh from PyPI and drift from every other cell"
        )
        assert "--constraint intel-constraints.txt" in script
        assert "-o intel-constraints.txt" in script


def test_lancedb_is_the_only_package_held_out_of_the_constraints() -> None:
    """A package dropped from the export is a package free to drift."""
    for script in _project_installs():
        words = script.split()
        excluded = {
            words[index + 1] for index, word in enumerate(words[:-1]) if word == "--no-emit-package"
        }

        assert excluded - {"lilbee-engine"} == _UNPINNED_PACKAGES, (
            f"the export drops {sorted(excluded)} from the lock; every name "
            "here resolves from PyPI and needs a reason in the workflow comment"
        )
