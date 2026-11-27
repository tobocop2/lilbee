"""Tests for the licence gate of the engine pack release (tools/check_pack_licences.py)."""

from __future__ import annotations

import importlib.util
import json
import re
import stat
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from tests._pack_workflow import pack_cells, pack_workflow

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_pack_licences.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pack_licences", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpl = _load()

_IK_VERSION = ADAPTERS[EngineName.IK_LLAMA].pin_record()["version"]
_LLAMA_VERSION = ADAPTERS[EngineName.LLAMA].pin_record()["version"]
_LINUX = "py3-none-manylinux_2_34_x86_64.whl"


def _wheel(
    folder: Path,
    name: str,
    *,
    licence: str = "MIT",
    extras: tuple[tuple[str, str], ...] = (),
) -> Path:
    """A wheel named *name* whose package holds an executable llama-server."""
    path = folder / name
    package = name.split("-", 1)[0]
    with zipfile.ZipFile(path, "w") as archive:
        server = zipfile.ZipInfo(f"{package}/bin/llama-server")
        server.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(server, "#!/bin/sh\n")
        archive.writestr(f"{package}-1.dist-info/METADATA", f"License-Expression: {licence}\n")
        for entry, text in extras:
            archive.writestr(entry, text)
    return path


def _ik(folder: Path, **kwargs: Any) -> Path:
    name = f"lilbee_engine_ik_llama_cpp-{_IK_VERSION}-4.cpu-{_LINUX}"
    return _wheel(folder, name, **kwargs)


def _plan(tmp_path: Path, plan: dict[str, list[dict[str, Any]]]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_every_declared_term_that_the_install_permits_passes() -> None:
    plan = {
        "llama": [
            {"cell": "linux-cpu", "license": "MIT"},
            {"cell": "linux-cuda", "license": "MIT AND LicenseRef-NVIDIA-CUDA-EULA"},
        ],
        "tei": [],
    }
    assert cpl.refused_terms(plan) == []


def test_a_gpl_term_is_refused_and_the_line_names_the_cell_and_the_terms() -> None:
    plan = {"ik": [{"cell": "linux-cpu", "license": "MIT AND GPL-3.0-only"}]}
    (line,) = cpl.refused_terms(plan)
    assert line.startswith("ik cell linux-cpu: ")
    assert "'MIT AND GPL-3.0-only'" in line


def test_an_empty_licence_is_refused() -> None:
    assert len(cpl.refused_terms({"llama": [{"cell": "c", "license": ""}]})) == 1


def test_a_cell_that_declares_no_licence_is_left_to_the_wheel_check() -> None:
    assert cpl.refused_terms({"mistralrs": [{"cell": "mistralrs-linux-x86_64-cpu"}]}) == []


def test_the_screen_runs_with_every_notice_present() -> None:
    assert {rule.file for rule in cpl.NOTICE_RULES.values()} == cpl.EVERY_NOTICE
    assert cpl.EVERY_NOTICE


def test_every_licence_that_the_cells_file_declares_passes_the_screen() -> None:
    engines = ("llama", "mistralrs", "tei", "ik")
    plan = {engine: pack_cells(engine) for engine in engines}
    assert any("license" in cell for cells in plan.values() for cell in cells)
    assert cpl.refused_terms(plan) == []


def test_a_wheel_that_the_install_accepts_passes(tmp_path: Path) -> None:
    _ik(tmp_path)
    assert cpl.refused_wheels(tmp_path) == []


def test_the_lock_comes_from_the_shipped_adapter(tmp_path: Path) -> None:
    lock = cpl.wheel_lock(_ik(tmp_path))
    pin = ADAPTERS[EngineName.IK_LLAMA].pin_record()
    assert (lock.engine, lock.version, lock.tool) == (
        EngineName.IK_LLAMA.value,
        _IK_VERSION,
        pin["tool"],
    )
    assert lock.backend == "cpu"


def test_a_cuda_backend_with_a_dash_names_the_lock(tmp_path: Path) -> None:
    name = f"lilbee_engine_ik_llama_cpp-{_IK_VERSION}-3.cuda_sm80-{_LINUX}"
    assert cpl.wheel_lock(tmp_path / name).backend == "cuda-sm80"


def test_a_wheel_with_a_gpl_file_is_refused_by_name(tmp_path: Path) -> None:
    wheel = _ik(tmp_path, extras=(("lilbee_engine_ik_llama_cpp/COPYING.GPL", "text"),))
    (line,) = cpl.refused_wheels(tmp_path)
    assert line.startswith(f"{wheel.name}: ")
    assert "GPL, AGPL or LGPL file" in line


def test_a_wheel_that_declares_a_gpl_expression_is_refused(tmp_path: Path) -> None:
    _ik(tmp_path, licence="GPL-3.0-only")
    (line,) = cpl.refused_wheels(tmp_path)
    assert "'GPL-3.0-only'" in line


def test_a_wheel_without_a_library_that_the_adapter_names_is_refused(tmp_path: Path) -> None:
    libraries = ADAPTERS[EngineName.LLAMA].pin_record()["libraries"]
    assert libraries
    name = f"lilbee_engine_llama_cpp-{_LLAMA_VERSION}-3.cpu-{_LINUX}"
    bare, full = tmp_path / "bare", tmp_path / "full"
    bare.mkdir()
    full.mkdir()
    _wheel(bare, name)
    (line,) = cpl.refused_wheels(bare)
    assert f"has no library file for {libraries[0]!r}" in line
    package = name.split("-", 1)[0]
    entries = tuple((f"{package}/lib/lib{library}.so", "x") for library in libraries)
    _wheel(full, name, extras=entries)
    assert cpl.refused_wheels(full) == []


def test_a_version_that_the_adapter_does_not_pin_is_refused(tmp_path: Path) -> None:
    name = f"lilbee_engine_ik_llama_cpp-0+git000000000000-4.cpu-{_LINUX}"
    _wheel(tmp_path, name)
    (line,) = cpl.refused_wheels(tmp_path)
    assert f"pins version {_IK_VERSION}, not 0+git000000000000" in line


def test_a_wheel_of_an_engine_with_no_shipped_adapter_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpl, "shipped_adapter", lambda _name: None)
    _ik(tmp_path)
    (line,) = cpl.refused_wheels(tmp_path)
    assert "lilbee ships no adapter for the engine ik_llama.cpp" in line


def test_a_wheel_name_of_no_engine_project_is_refused(tmp_path: Path) -> None:
    _wheel(tmp_path, f"lilbee_engine_nothing-1-1.cpu-{_LINUX}")
    (line,) = cpl.refused_wheels(tmp_path)
    assert "is not an engine project" in line


def test_a_folder_without_a_wheel_is_refused(tmp_path: Path) -> None:
    (tmp_path / "x.whl.sha256").write_text("x", encoding="utf-8")
    assert cpl.refused_wheels(tmp_path) == [f"{tmp_path}: no lilbee_engine_*.whl file to check."]


def test_main_terms_exits_1_and_prints_an_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan(tmp_path, {"ik": [{"cell": "linux-cpu", "license": "GPL-3.0-only"}]})
    assert cpl.main(["terms", str(plan)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("::error::ik cell linux-cpu: ")


def test_main_terms_exits_0_when_every_cell_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan(tmp_path, {"ik": [{"cell": "linux-cpu", "license": "MIT"}]})
    assert cpl.main(["terms", str(plan)]) == 0
    assert "lilbee accepts every pack" in capsys.readouterr().out


def test_main_wheels_checks_the_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ik(tmp_path)
    assert cpl.main(["wheels", str(tmp_path)]) == 0
    _ik(tmp_path, licence="AGPL-3.0-only")
    assert cpl.main(["wheels", str(tmp_path)]) == 1
    assert "'AGPL-3.0-only'" in capsys.readouterr().err


def test_main_needs_a_check() -> None:
    with pytest.raises(SystemExit):
        cpl.main([])


_CALL = re.compile(r"tools/check_pack_licences\.py (\w+) (\S+)")


def test_the_pack_workflow_calls_each_check_of_this_tool() -> None:
    workflow = yaml.safe_load(pack_workflow().read_text(encoding="utf-8"))
    calls = {
        job: [m.groups() for s in spec["steps"] for m in _CALL.finditer(s.get("run", ""))]
        for job, spec in workflow["jobs"].items()
    }
    assert calls["plan"] == [("terms", "plan.json")]
    assert calls["publish"] == [("wheels", "packs")]
