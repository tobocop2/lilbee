"""The compat lancedb index is an explicit uv index bound to lancedb alone.

A flat extra index lets every package in a resolution fall through to it.
An explicit index serves only the packages whose source names it, so the
safety of the install no longer depends on that index serving one project.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_WORKFLOWS = sorted((_ROOT / ".github" / "workflows").glob("*.yml"))
_COMPAT_URL = "https://lilbee.sh/compat/"


def _compat_indexes(pyproject_text: str) -> list[dict]:
    uv = tomllib.loads(pyproject_text)["tool"]["uv"]
    return [index for index in uv.get("index", []) if index.get("url") == _COMPAT_URL]


def _packages_sourced_from(pyproject_text: str, index_name: str) -> set[str]:
    sources = tomllib.loads(pyproject_text)["tool"]["uv"].get("sources", {})
    named = set()
    for package, source in sources.items():
        entries = source if isinstance(source, list) else [source]
        if any(entry.get("index") == index_name for entry in entries):
            named.add(package)
    return named


def _flat_compat_flags(workflow_text: str) -> list[str]:
    """Return every step command that offers the compat index to a whole resolution."""
    workflow = yaml.safe_load(workflow_text)
    flagged = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            words = step.get("run", "").replace("\\\n", " ").split()
            for flag in ("--extra-index-url", "--index", "--find-links", "-f"):
                for position, word in enumerate(words[:-1]):
                    if word == flag and words[position + 1].rstrip("/") == _COMPAT_URL.rstrip("/"):
                        flagged.append(f"{flag} {words[position + 1]}")
    return flagged


def test_the_compat_index_is_explicit() -> None:
    indexes = _compat_indexes(_PYPROJECT.read_text(encoding="utf-8"))

    assert len(indexes) == 1
    assert indexes[0]["explicit"] is True


def test_only_lancedb_names_the_compat_index() -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")
    name = _compat_indexes(text)[0]["name"]

    assert _packages_sourced_from(text, name) == {"lancedb"}


def test_the_lancedb_source_is_limited_to_intel_macs() -> None:
    """Every other platform keeps the stock wheel from PyPI."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    source = tomllib.loads(text)["tool"]["uv"]["sources"]["lancedb"]

    assert source["marker"] == "sys_platform == 'darwin' and platform_machine == 'x86_64'"


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda path: path.name)
def test_no_workflow_offers_the_compat_index_as_an_extra_index(workflow: Path) -> None:
    assert _flat_compat_flags(workflow.read_text(encoding="utf-8")) == []


def test_the_lock_resolves_the_compat_lancedb_only_from_the_compat_index() -> None:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    sources = {
        package["name"]: package["source"]["registry"]
        for package in lock["package"]
        if package["source"].get("registry") == _COMPAT_URL
    }

    assert sources == {"lancedb": _COMPAT_URL}


# Revert arms: the checks above must fail on the configuration before the change.


def test_a_flat_extra_index_is_flagged() -> None:
    workflow = (
        "jobs:\n  build:\n    steps:\n      - run: |\n"
        "          uv pip install \\\n"
        "            --extra-index-url https://lilbee.sh/compat/ \\\n"
        "            lancedb==0.40.0+compat\n"
    )

    assert _flat_compat_flags(workflow) == ["--extra-index-url https://lilbee.sh/compat/"]


def test_a_flat_index_in_pyproject_is_not_explicit() -> None:
    text = '[tool.uv]\nindex = [{ name = "c", url = "https://lilbee.sh/compat/" }]\n'

    assert "explicit" not in _compat_indexes(text)[0]


def test_a_second_package_on_the_compat_index_is_found() -> None:
    text = '[tool.uv.sources]\nlancedb = { index = "c" }\nnumpy = { index = "c" }\n'

    assert _packages_sourced_from(text, "c") == {"lancedb", "numpy"}
