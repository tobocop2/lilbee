"""Dataclass field defaults must be values Python 3.11 accepts (it rejects unhashable defaults).

The census imports every module under ``lilbee`` and reads each dataclass field's real default,
so no spelling of the default (alias, ``field(default=...)``, module constant) can hide it.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import pkgutil
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest

import lilbee

# Modules the walk does not import: running ``python -m lilbee`` is the CLI entry point.
_SKIPPED_MODULES = frozenset({"lilbee.__main__"})
# Optional extras (pyproject.toml): a missing one skips only the module that needs it.
_OPTIONAL_DEPENDENCIES = frozenset(
    {"crawlberg", "playwright", "ada_url", "spacy", "graspologic_native", "liter_llm"}
)


def _is_unhashable(value: object) -> bool:
    """Return True when Python 3.11 would reject ``value`` as a dataclass default.

    3.11 rejects a default whose type sets ``__hash__`` to None. A mapping proxy over a dict
    defines ``__hash__`` on 3.12+ but fails when called, so a failing ``hash`` counts too.
    """
    if type(value).__hash__ is None:
        return True
    try:
        hash(value)
    except TypeError:
        return True
    return False


def _classes_in(
    owner: ModuleType | type, module_name: str, seen: set[type] | None = None
) -> list[type]:
    """Return the classes defined in ``module_name`` under ``owner``, nested ones too."""
    seen = set() if seen is None else seen
    found: list[type] = []
    for value in vars(owner).values():
        if inspect.isclass(value) and value.__module__ == module_name and value not in seen:
            seen.add(value)
            found.append(value)
            found.extend(_classes_in(value, module_name, seen))
    return found


def _dataclasses_in(module: ModuleType) -> list[type]:
    return [c for c in _classes_in(module, module.__name__) if dataclasses.is_dataclass(c)]


def _offenders(classes: list[type]) -> list[str]:
    return [
        f"{cls.__module__}.{cls.__qualname__}.{f.name}"
        for cls in classes
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING and _is_unhashable(f.default)
    ]


def _import_package_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for info in pkgutil.walk_packages(lilbee.__path__, "lilbee."):
        if info.name in _SKIPPED_MODULES:
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except ModuleNotFoundError as exc:
            if exc.name is None or exc.name.split(".")[0] not in _OPTIONAL_DEPENDENCIES:
                raise
    return modules


def _import_scratch(tmp_path: Path, source: str) -> ModuleType:
    path = tmp_path / "scratch_plant.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("scratch_plant", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


_PLANTS = {
    "field_default_proxy": (
        "from dataclasses import dataclass, field\n"
        "from types import MappingProxyType\n"
        "@dataclass(frozen=True)\n"
        "class Plant:\n"
        "    m: dict[str, int] = field(default=MappingProxyType({}))\n"
    ),
    "module_constant": (
        "from dataclasses import dataclass\n"
        "from types import MappingProxyType\n"
        "_EMPTY = MappingProxyType({})\n"
        "@dataclass(frozen=True)\n"
        "class Plant:\n"
        "    m: dict[str, int] = _EMPTY\n"
    ),
    "aliased_import": (
        "from dataclasses import dataclass as dc\n"
        "from types import MappingProxyType as Proxy\n"
        "@dc(frozen=True)\n"
        "class Plant:\n"
        "    m: dict[str, int] = Proxy({})\n"
    ),
}


@pytest.mark.parametrize("shape", sorted(_PLANTS))
def test_census_flags_a_planted_default(shape: str, tmp_path: Path) -> None:
    if sys.version_info < (3, 12):
        # 3.11 rejects the plant while the class is created, before any census runs.
        with pytest.raises(ValueError, match="mutable default"):
            _import_scratch(tmp_path, _PLANTS[shape])
        return
    module = _import_scratch(tmp_path, _PLANTS[shape])
    assert _offenders(_dataclasses_in(module)) == ["scratch_plant.Plant.m"]


def test_census_passes_hashable_and_factory_defaults(tmp_path: Path) -> None:
    source = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class Fine:\n"
        "    a: int = 0\n"
        "    b: tuple[int, ...] = ()\n"
        "    c: dict[str, int] = field(default_factory=dict)\n"
        "    d: int = field(default=3)\n"
        "    class Inner:\n"
        "        pass\n"
        "@dataclass\n"
        "class Outer:\n"
        "    @dataclass\n"
        "    class Nested:\n"
        "        z: int = 1\n"
        "    e: str = 'x'\n"
    )
    module = _import_scratch(tmp_path, source)
    names = sorted(c.__qualname__ for c in _dataclasses_in(module))
    assert names == ["Fine", "Outer", "Outer.Nested"]
    assert _offenders(_dataclasses_in(module)) == []


def test_census_reads_nested_dataclass_defaults(tmp_path: Path) -> None:
    if sys.version_info < (3, 12):
        pytest.skip("3.11 rejects the plant at class creation")
    source = (
        "from dataclasses import dataclass\n"
        "from types import MappingProxyType\n"
        "class Holder:\n"
        "    @dataclass\n"
        "    class Inner:\n"
        "        m: dict[str, int] = MappingProxyType({})\n"
    )
    module = _import_scratch(tmp_path, source)
    assert _offenders(_dataclasses_in(module)) == ["scratch_plant.Holder.Inner.m"]


def test_every_dataclass_default_is_hashable() -> None:
    modules = _import_package_modules()
    classes = [c for module in modules for c in _dataclasses_in(module)]
    assert len(modules) > 100
    assert len(classes) > 200
    assert _offenders(classes) == []


def test_unhashable_check_matches_python_311() -> None:
    assert _is_unhashable({}) and _is_unhashable([]) and _is_unhashable(set())
    assert _is_unhashable(types.MappingProxyType({}))
    assert not (_is_unhashable(0) or _is_unhashable(()) or _is_unhashable(None))
