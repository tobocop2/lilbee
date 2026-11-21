"""Every lilbee name that a script under scripts/ or tools/ imports or names in a string exists."""

from __future__ import annotations

import ast
import importlib
import pkgutil
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


@dataclass(frozen=True)
class DottedRef:
    """One dotted path, naming a module or an attribute of one, that a script puts in a string."""

    path: str
    line: int
    dotted: str
    origin: str

    @property
    def label(self) -> str:
        return f"{self.path}:{self.line}:{self.origin}:{self.dotted}"


Ref = ImportRef | DottedRef

LILBEE_DOTTED = re.compile(r"lilbee(?:\.\w+)*")
DYNAMIC_IMPORT_CALLS = frozenset({"import_module", "find_spec"})


def _is_lilbee(module: str) -> bool:
    return module == "lilbee" or module.startswith("lilbee.")


def _callee(node: ast.Call) -> list[str]:
    """Return the dotted parts of the callee, such as ["mock", "patch", "object"]."""
    parts: list[str] = []
    func: ast.expr = node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return parts[::-1]


def _string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _creates_missing_attribute(node: ast.Call) -> bool:
    """Whether the patch call passes create=True, so the attribute may be absent."""
    return any(
        kw.arg == "create" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in node.keywords
    )


def _lilbee_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Map each local name bound by a lilbee import to the dotted paths it can name."""
    bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_lilbee(alias.name) and alias.asname:
                    bindings.setdefault(alias.asname, set()).add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if _is_lilbee(node.module) and alias.name != "*":
                    bindings.setdefault(alias.asname or alias.name, set()).add(
                        f"{node.module}.{alias.name}"
                    )
    return bindings


def _call_refs(path: str, node: ast.Call, bindings: dict[str, set[str]]) -> tuple[set[Ref], bool]:
    """Return the string-named refs in one call, and whether it imports a module by a variable."""
    callee = _callee(node)
    first = node.args[0] if node.args else None
    refs: set[Ref] = set()
    if callee[-2:] == ["patch", "object"] and len(node.args) >= 2:
        attr = _string(node.args[1])
        if isinstance(first, ast.Name) and attr and not _creates_missing_attribute(node):
            refs.update(
                DottedRef(path, node.lineno, f"{bound}.{attr}", "patch.object")
                for bound in bindings.get(first.id, ())
            )
    elif callee[-1:] == ["patch"]:
        target = _string(first)
        if target and _is_lilbee(target) and not _creates_missing_attribute(node):
            refs.add(DottedRef(path, node.lineno, target, "patch"))
    elif callee[-1:] and callee[-1] in DYNAMIC_IMPORT_CALLS:
        module = _string(first)
        if module is None:
            return refs, True
        if _is_lilbee(module):
            refs.add(ImportRef(path, node.lineno, module, None))
    return refs, False


def _pair_refs(path: str, tree: ast.AST) -> set[ImportRef]:
    """Return the ("lilbee.module", "Name") string pairs that a dynamic import reads."""
    refs: set[ImportRef] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            module, name = _string(node.elts[0]), _string(node.elts[1])
            if module and name and LILBEE_DOTTED.fullmatch(module) and name.isidentifier():
                refs.add(ImportRef(path, node.lineno, module, name))
    return refs


def _python_refs(path: str, source: str) -> set[Ref]:
    tree = ast.parse(source)
    bindings = _lilbee_bindings(tree)
    refs: set[Ref] = set()
    dynamic_import = False
    for node in ast.walk(tree):
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
        elif isinstance(node, ast.Call):
            call_refs, by_variable = _call_refs(path, node, bindings)
            refs.update(call_refs)
            dynamic_import = dynamic_import or by_variable
    if dynamic_import:
        refs.update(_pair_refs(path, tree))
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


def collect_refs(root: Path) -> list[Ref]:
    """Return every lilbee import or string target in root's script directories."""
    refs: set[Ref] = set()
    for directory in SCRIPT_DIRS:
        for script in (root / directory).rglob("*"):
            if script.suffix not in SCRIPT_SUFFIXES or "__pycache__" in script.parts:
                continue
            rel = script.relative_to(root).as_posix()
            source = script.read_text(encoding="utf-8")
            finder = _python_refs if script.suffix == ".py" else _shell_refs
            refs.update(finder(rel, source))
    return sorted(refs, key=lambda ref: (ref.path, ref.line, ref.label))


def resolve_problem(ref: Ref) -> str | None:
    """Return why the import or string target does not resolve, or None when it does."""
    if isinstance(ref, DottedRef):
        try:
            pkgutil.resolve_name(ref.dotted)
        except (ImportError, AttributeError) as exc:
            return f"{ref.dotted} does not resolve: {exc}"
        return None
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
def test_script_import_resolves(ref: Ref) -> None:
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


STRING_TARGET_SCRIPT = (
    "import importlib\n"
    "import importlib.util\n"
    "from unittest import mock\n"
    "from unittest.mock import patch\n"
    "from lilbee.providers.fleet import planning as planning_mod\n"
    "import lilbee.core.config as config_mod\n"
    "\n"
    "SURFACES = {'chat': ('lilbee.cli.tui.screens.chat', 'ChatScreen'), 'x': ('a', 'B')}\n"
    "\n"
    "def run(module, other):\n"
    "    importlib.import_module('lilbee.core.config')\n"
    "    importlib.import_module('json')\n"
    "    importlib.util.find_spec('lilbee.app.placement')\n"
    "    importlib.import_module(module)\n"
    "    with (\n"
    "        mock.patch(\n"
    "            'lilbee.core.config.Config', return_value=1\n"
    "        ),\n"
    "        patch('lilbee.core.config.Missing', create=True),\n"
    "        patch('os.path.join'),\n"
    "        patch(other),\n"
    "        mock.patch.object(planning_mod, 'plan_all_launches', None),\n"
    "        mock.patch.object(config_mod, 'cfg', None),\n"
    "        mock.patch.object(config_mod, 'absent', None, create=True),\n"
    "        mock.patch.object(other, 'attr', None),\n"
    "        mock.patch.object(),\n"
    "    ):\n"
    "        pass\n"
)


def test_collect_refs_reads_string_targets(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "qa" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(STRING_TARGET_SCRIPT, encoding="utf-8")

    refs = [ref for ref in collect_refs(tmp_path) if ref.line > 6]

    assert [(type(ref), ref.line, ref.label.split(":", 2)[2]) for ref in refs] == [
        (ImportRef, 8, "lilbee.cli.tui.screens.chat:ChatScreen"),
        (ImportRef, 11, "lilbee.core.config"),
        (ImportRef, 13, "lilbee.app.placement"),
        (DottedRef, 16, "patch:lilbee.core.config.Config"),
        (DottedRef, 22, "patch.object:lilbee.providers.fleet.planning.plan_all_launches"),
        (DottedRef, 23, "patch.object:lilbee.core.config.cfg"),
    ]


def test_collect_refs_ignores_unrelated_string_calls(tmp_path: Path) -> None:
    script = tmp_path / "tools" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "def run(f, g):\n    f('lilbee.core.config.Missing')\n    g()('lilbee.nope')\n",
        encoding="utf-8",
    )

    assert collect_refs(tmp_path) == []


@pytest.mark.parametrize(
    ("dotted", "reason"),
    [
        ("lilbee.cli.tui.app.chat_ready", None),
        ("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", None),
        ("lilbee.cli.tui.gone.chat_ready", "does not resolve"),
        ("lilbee.cli.tui.app.chat_ready_renamed", "has no attribute 'chat_ready_renamed'"),
    ],
)
def test_resolve_problem_checks_a_dotted_target(dotted: str, reason: str | None) -> None:
    problem = resolve_problem(DottedRef("probe.py", 1, dotted, "patch"))

    if reason is None:
        assert problem is None
    else:
        assert problem is not None
        assert reason in problem


def test_a_renamed_patch_target_in_a_script_fails_the_check(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "qa" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from unittest import mock\n"
        "mock.patch('lilbee.cli.tui.app.chat_ready')\n"
        "mock.patch('lilbee.cli.tui.app.chat_ready_renamed')\n",
        encoding="utf-8",
    )

    refs = collect_refs(tmp_path)

    assert [resolve_problem(ref) is None for ref in refs] == [True, False]


def test_the_real_scripts_name_string_targets_of_every_kind() -> None:
    origins = {ref.origin for ref in REFS if isinstance(ref, DottedRef)}
    pair_paths = {ref.path for ref in REFS if isinstance(ref, ImportRef) and ref.name}

    assert origins == {"patch", "patch.object"}
    assert "scripts/qa/tui_render_sweep.py" in pair_paths
