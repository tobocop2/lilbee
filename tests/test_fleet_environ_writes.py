"""No fleet thread writes the process environment.

``setenv`` reallocates the C ``environ`` array. A native thread reading the
environment at the same moment (lancedb's runtime, the Vulkan loader) can walk
freed memory and segfault. The GPU variables belong in the environment handed to
a child at spawn, never in this process's own.
"""

from __future__ import annotations

import ast
import os
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import cuda_runtime, gpu_env, gpu_select, planning

FLEET_SOURCE = Path(planning.__file__).parent

_WRITE_METHODS = {"setdefault", "update", "pop", "popitem", "clear"}
_WRITE_FUNCTIONS = {"putenv", "unsetenv"}


class _RecordingEnviron(MutableMapping[str, str]):
    """An ``os.environ`` stand-in that records every write and the thread that made it."""

    def __init__(self, initial: dict[str, str]) -> None:
        self._data = dict(initial)
        self.writes: list[tuple[str, str, str]] = []

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.writes.append((threading.current_thread().name, "set", key))
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        self.writes.append((threading.current_thread().name, "del", key))
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def copy(self) -> dict[str, str]:
        return dict(self._data)


def _is_environ(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _writes_in(source: str) -> list[int]:
    """Line numbers where *source* assigns to, deletes from, or mutates ``os.environ``."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript) and _is_environ(node.value):
            if isinstance(node.ctx, ast.Store | ast.Del):
                lines.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func
            mutates_environ = _is_environ(target.value) and target.attr in _WRITE_METHODS
            if mutates_environ or target.attr in _WRITE_FUNCTIONS:
                lines.append(node.lineno)
    return lines


def test_no_fleet_module_writes_the_process_environment() -> None:
    offenders = {
        f"{path.relative_to(FLEET_SOURCE)}:{line}"
        for path in sorted(FLEET_SOURCE.rglob("*.py"))
        for line in _writes_in(path.read_text(encoding="utf-8"))
    }
    assert not offenders, f"fleet code writes os.environ at {sorted(offenders)}"


def test_the_scan_sees_each_kind_of_environment_write() -> None:
    source = (
        "import os\n"
        "os.environ['A'] = '1'\n"
        "del os.environ['A']\n"
        "os.environ.setdefault('B', '1')\n"
        "os.environ.update({'C': '1'})\n"
        "os.environ.pop('C', None)\n"
        "os.putenv('D', '1')\n"
        "os.unsetenv('D')\n"
        "os.environ.get('E')\n"
        "dict(os.environ)\n"
    )
    assert _writes_in(source) == [2, 3, 4, 5, 6, 7, 8]


def test_planning_on_a_worker_thread_writes_no_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The warm-up thread plans every launch; that pass must leave the environment alone."""
    for name in gpu_env._GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("VK_LOADER_LAYERS_DISABLE", raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "0")
    monkeypatch.setattr(gpu_env.sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [tmp_path / "wheel"])
    monkeypatch.setattr(gpu_select, "disable_conflicting_vulkan_icds", lambda: "icd.json")
    binary = tmp_path / "llama-server"
    monkeypatch.setattr(planning, "resolve_llama_server", lambda: binary)
    monkeypatch.setattr(planning, "_plan_devices", lambda _binary: [])
    monkeypatch.setattr(planning, "_launched_roles", lambda: ())
    monkeypatch.setattr(planning, "plan_launches", lambda *_a, **_k: "plan")

    recording = _RecordingEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", recording)
    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.append(planning.plan_all_launches()), name="fleet-warm-up"
    )
    thread.start()
    thread.join(timeout=30)

    assert result == ["plan"]
    assert recording.writes == []


_ICD_GLOB = "amd_icd.json,intel_icd.json"


@pytest.mark.parametrize(
    ("icd_glob", "pin"),
    [(_ICD_GLOB, None), (None, None), (_ICD_GLOB, "0"), (None, "0")],
    ids=["dual-vendor", "single-vendor", "dual-vendor-pinned", "single-vendor-pinned"],
)
def test_the_probe_child_gets_the_gpu_variables_through_its_own_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, icd_glob: str | None, pin: str | None
) -> None:
    from lilbee.providers.fleet import devices

    for name in (*gpu_env._GPU_VISIBLE_ENV_VARS, "LD_LIBRARY_PATH", "VK_LOADER_DRIVERS_DISABLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VK_LOADER_LAYERS_DISABLE", raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", pin)
    monkeypatch.setattr(gpu_env.sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    wheel = tmp_path / "wheel"
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
    monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: True)
    monkeypatch.setattr(cuda_runtime, "_ships_its_own_cuda_runtime", lambda _b: False)
    monkeypatch.setattr(gpu_select, "disable_conflicting_vulkan_icds", lambda: icd_glob)

    env = devices._probe_env(tmp_path / "llama-server")

    assert env["VK_LOADER_LAYERS_DISABLE"] == gpu_env._VK_LOADER_LAYERS_DISABLE_VALUE
    assert env.get("VK_LOADER_DRIVERS_DISABLE") == icd_glob
    assert str(wheel) in env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    for name in ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        assert env.get(name) == pin
    assert "ROCR_VISIBLE_DEVICES" not in env
    for name in ("VK_LOADER_LAYERS_DISABLE", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"):
        assert name not in os.environ


def test_a_user_set_loader_variable_beats_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_env.sys, "platform", "linux")
    monkeypatch.setattr(gpu_select, "disable_conflicting_vulkan_icds", lambda: None)
    monkeypatch.setenv("VK_LOADER_LAYERS_DISABLE", "mine")
    assert gpu_env.engine_environ()["VK_LOADER_LAYERS_DISABLE"] == "mine"


def test_the_vulkan_enumeration_child_runs_under_the_loader_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.providers.fleet import proc

    seen: dict[str, object] = {}

    def _run(argv: list[str], **kwargs: object) -> tuple[str, int]:
        seen.update(kwargs)
        return "[]", 0

    monkeypatch.setattr(proc, "run_bounded", _run)
    monkeypatch.setattr(gpu_env, "engine_environ", lambda: {"MARK": "engine-env"})
    gpu_select._run_probe_child()
    assert seen["env"] == {"MARK": "engine-env"}


def test_the_engine_help_run_gets_the_engine_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess
    from types import SimpleNamespace

    from lilbee.providers.fleet import binary

    seen: dict[str, object] = {}

    def _run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(stdout="usage", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(gpu_env, "engine_environ", lambda: {"MARK": "engine-env"})
    binary.engine_help.cache_clear()
    assert binary.engine_help(tmp_path / "unique-help-binary") == "usage\n"
    assert seen["env"] == {"MARK": "engine-env"}
    binary.engine_help.cache_clear()
