"""Every lilbee name that a script under scripts/ or tools/ imports exists."""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRS = ("scripts", "tools")
SCRIPT_SUFFIXES = (".py", ".sh")

LILBEE_MODULE = r"lilbee(?:\.\w+)*"
SHELL_FROM_IMPORT = re.compile(rf"\bfrom\s+({LILBEE_MODULE})\s+import\s+(\([^)]*\)|[^\n;\"']+)")
SHELL_IMPORT = re.compile(rf"\bimport\s+({LILBEE_MODULE})(?!\w)")
COMMENT = re.compile(r"#[^\n]*")


@dataclass(frozen=True)
class ImportRef:
    """One lilbee module, and optionally one name taken from it, that a script imports."""

    path: str
    line: int
    module: str
    name: str | None

    @property
    def label(self) -> str:
        target = self.module if self.name is None else f"{self.module}:{self.name}"
        return f"{self.path}:{self.line}:{target}"


def _is_lilbee(module: str) -> bool:
    return module == "lilbee" or module.startswith("lilbee.")


def _python_refs(path: str, source: str) -> set[ImportRef]:
    refs: set[ImportRef] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            refs.update(
                ImportRef(path, node.lineno, alias.name, None)
                for alias in node.names
                if _is_lilbee(alias.name)
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            refs.update(
                ImportRef(path, node.lineno, node.module, alias.name)
                for alias in node.names
                if _is_lilbee(node.module) and alias.name != "*"
            )
    return refs


def _clause_names(clause: str) -> list[str]:
    body = COMMENT.sub("", clause).strip("() \n\t").replace("\n", " ")
    return [part.split()[0] for part in body.split(",") if part.strip()]


def _shell_refs(path: str, source: str) -> set[ImportRef]:
    refs: set[ImportRef] = set()
    for match in SHELL_FROM_IMPORT.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        refs.update(
            ImportRef(path, line, match.group(1), name) for name in _clause_names(match.group(2))
        )
    for match in SHELL_IMPORT.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        refs.add(ImportRef(path, line, match.group(1), None))
    return refs


def collect_refs(root: Path) -> list[ImportRef]:
    """Return every lilbee import in the .py and .sh files under root's script directories."""
    refs: set[ImportRef] = set()
    for directory in SCRIPT_DIRS:
        for script in (root / directory).rglob("*"):
            if script.suffix not in SCRIPT_SUFFIXES or "__pycache__" in script.parts:
                continue
            rel = script.relative_to(root).as_posix()
            source = script.read_text(encoding="utf-8")
            finder = _python_refs if script.suffix == ".py" else _shell_refs
            refs.update(finder(rel, source))
    return sorted(refs, key=lambda ref: (ref.path, ref.line, ref.module, ref.name or ""))


def resolve_problem(ref: ImportRef) -> str | None:
    """Return why the import does not resolve, or None when it does."""
    try:
        module = importlib.import_module(ref.module)
    except ImportError as exc:
        return f"{ref.module} does not import: {exc}"
    if ref.name is None or hasattr(module, ref.name):
        return None
    try:
        importlib.import_module(f"{ref.module}.{ref.name}")
    except ImportError:
        return f"{ref.module} has no name {ref.name}"
    return None


REFS = collect_refs(REPO_ROOT)


@pytest.mark.parametrize("ref", REFS, ids=lambda ref: ref.label)
def test_script_import_resolves(ref: ImportRef) -> None:
    assert resolve_problem(ref) is None


def test_scripts_name_lilbee_imports_in_python_and_shell() -> None:
    suffixes = {Path(ref.path).suffix for ref in REFS}
    assert suffixes == {".py", ".sh"}


def test_collect_refs_reads_python_imports_of_any_depth(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "qa" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import lilbee.core.config as config\n"
        "import os\n"
        "from lilbee.app import placement\n"
        "from lilbee.core.config import *\n"
        "from . import sibling\n"
        "from lilbee_engine_llama_cpp import bin_dir\n"
        "\n"
        "def run():\n"
        "    from lilbee.providers.roles import WorkerRole, ROLE_REGISTRY as REG\n",
        encoding="utf-8",
    )

    refs = collect_refs(tmp_path)

    assert [(ref.module, ref.name, ref.line) for ref in refs] == [
        ("lilbee.core.config", None, 1),
        ("lilbee.app", "placement", 3),
        ("lilbee.providers.roles", "ROLE_REGISTRY", 9),
        ("lilbee.providers.roles", "WorkerRole", 9),
    ]


def test_collect_refs_reads_shell_imports(tmp_path: Path) -> None:
    script = tmp_path / "tools" / "run.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/bin/sh\n"
        "python -c 'import lilbee.mcp_server'\n"
        "python -c 'import lilbee_engine_llama_cpp'\n"
        "python - <<'PY'\n"
        "from lilbee.core.config import (  # noqa: E402\n"
        "    Config,\n"
        "    cfg as settings,\n"
        ")\n"
        "from lilbee.app.agent_configs.opencode import opencode_config\n"
        "PY\n",
        encoding="utf-8",
    )

    refs = collect_refs(tmp_path)

    assert [(ref.module, ref.name, ref.line) for ref in refs] == [
        ("lilbee.mcp_server", None, 2),
        ("lilbee.core.config", "Config", 5),
        ("lilbee.core.config", "cfg", 5),
        ("lilbee.app.agent_configs.opencode", "opencode_config", 9),
    ]


@pytest.mark.parametrize(
    ("module", "name", "reason"),
    [
        ("lilbee.cli.agent_configs.opencode", "opencode_config", "does not import"),
        ("lilbee.core.config", "no_such_name", "has no name no_such_name"),
    ],
)
def test_resolve_problem_names_the_missing_part(module: str, name: str, reason: str) -> None:
    problem = resolve_problem(ImportRef("probe.py", 1, module, name))

    assert problem is not None
    assert reason in problem


def test_resolve_problem_accepts_a_submodule_named_in_from_import() -> None:
    assert resolve_problem(ImportRef("probe.py", 1, "lilbee.providers.fleet", "planning")) is None
