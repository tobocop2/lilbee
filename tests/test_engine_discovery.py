"""The engine adapters are found by module: a module of the engines package exports one."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from lilbee.providers.fleet import engines as engines_pkg
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS, discover_adapters
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.engines.tei import TEI

_EXTRA_MODULE = """\
from tests.conformance.fake_adapter import FakeAdapter

ADAPTER = FakeAdapter()
ADAPTER.name = {name!r}
"""


def _add_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, stem: str) -> None:
    """Put a module named *stem* beside the adapters; its adapter is called *name*."""
    (tmp_path / f"{stem}.py").write_text(_EXTRA_MODULE.format(name=str(name)), encoding="utf-8")
    monkeypatch.setattr(engines_pkg, "__path__", [*engines_pkg.__path__, str(tmp_path)])


def test_the_shipped_adapters_are_found() -> None:
    found = discover_adapters()
    assert found == {
        EngineName.LLAMA: LLAMA,
        EngineName.IK_LLAMA: IK_LLAMA,
        EngineName.MISTRAL: MISTRAL,
        EngineName.TEI: TEI,
        EngineName.MLX: MLX,
    }


def test_the_adapter_table_is_the_discovered_set() -> None:
    assert ADAPTERS
    assert dict(ADAPTERS) == discover_adapters()


def test_the_order_is_the_order_of_the_engine_names() -> None:
    names = list(discover_adapters())
    assert names == sorted(names, key=list(EngineName).index)


def test_a_new_module_is_found_without_an_edit_to_the_choice_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_module(monkeypatch, tmp_path, EngineName.OPENVINO, "extra_openvino_engine")
    found = discover_adapters()
    assert EngineName.OPENVINO in found
    assert found[EngineName.OPENVINO].name == EngineName.OPENVINO
    assert list(found) == sorted(found, key=list(EngineName).index)


def test_a_module_that_exports_no_adapter_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "helper_only.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(engines_pkg, "__path__", [*engines_pkg.__path__, str(tmp_path)])
    assert discover_adapters() == dict(ADAPTERS)


def test_two_modules_for_one_engine_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_module(monkeypatch, tmp_path, EngineName.LLAMA, "second_llama_engine")
    with pytest.raises(
        ValueError, match=re.escape("llama.cpp is exported by more than one module")
    ):
        discover_adapters()


def test_an_adapter_with_an_unknown_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_module(monkeypatch, tmp_path, "no-such-engine", "unnamed_engine")
    with pytest.raises(ValueError, match="no-such-engine"):
        discover_adapters()


_SHARED_MODULES = frozenset({"base", "host_gate"})


def _engine_module_imports(source: str) -> list[str]:
    """Return every import in *source* that reaches a module of the engines package."""
    package = engines_pkg.__name__
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith(f"{package}.")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # `from . import tei` or `from .tei import X` inside the engines package
                found += [f".{module}" if module else f".{a.name}" for a in node.names]
            elif module.startswith(f"{package}."):
                found.append(module)
            elif module == package:
                found += [f"{package}.{a.name}" for a in node.names]
    return [name for name in found if name.rsplit(".", 1)[-1] not in _SHARED_MODULES]


@pytest.mark.parametrize(
    "line",
    [
        "from lilbee.providers.fleet.engines.tei import TEI",
        "from lilbee.providers.fleet.engines import tei",
        "from . import tei",
        "from .tei import TEI",
        "import lilbee.providers.fleet.engines.tei",
        "import lilbee.providers.fleet.engines.tei as t",
    ],
)
def test_the_import_check_sees_every_spelling_of_an_engine_import(line: str) -> None:
    assert _engine_module_imports(line) != []


def test_the_import_check_allows_the_shared_base_module() -> None:
    line = "from lilbee.providers.fleet.engines.base import EngineName"
    assert _engine_module_imports(line) == []


def test_the_choice_module_imports_no_engine_module() -> None:
    source = Path(choice.__file__).read_text(encoding="utf-8")
    assert _engine_module_imports(source) == []
