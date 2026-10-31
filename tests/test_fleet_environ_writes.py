"""No lilbee thread writes the process environment after startup.

``setenv`` reallocates the C ``environ`` array. A native thread reading the
environment at the same moment (lancedb's runtime, the Vulkan loader) can walk
freed memory and segfault. A value a child needs belongs in the environment
handed to that child at spawn. A value a library reads in this process is set
once at startup, on the main thread, before lilbee starts any thread. A library
can already own native threads then (importing pyarrow starts three), so the set
of startup writes is small and named.
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
from lilbee.providers.fleet.engines import LLAMA

SOURCE = Path(planning.__file__).parents[2]

# Writes allowed at startup: the main thread runs them while lilbee has started
# no thread of its own. A library may already own native threads (pyarrow starts
# three when it is imported), which is why a variable that a child reads is set
# here once and never again. Each entry is a file and the function that holds
# the write ("<module>" is top-level code), so a new write in the same file
# fails the scan.
_STARTUP_WRITES = {
    ("__init__.py", "<module>"): "import-time default, set on the importing main thread",
    ("__init__.py", "_disable_xet_where_it_stalls"): "called once at import, main thread",
    ("__main__.py", "_isolate_vendored_openssl"): "entry-point default before any command runs",
    ("crawler/__init__.py", "<module>"): "import-time default, importing main thread",
    ("core/config/model.py", "<module>"): "import-time export of the data root and marks",
    ("core/config/model.py", "export_download_environment"): "startup export, main thread",
    ("cli/app.py", "_apply_data_root"): "the CLI callback exports the data root on the main thread",
}

_WRITE_METHODS = {
    "setdefault",
    "update",
    "pop",
    "popitem",
    "clear",
    "__setitem__",
    "__delitem__",
    "__ior__",
}
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


def _is_environ(node: ast.expr, aliases: set[str]) -> bool:
    """True for ``os.environ`` or a name bound to the environment mapping."""
    if isinstance(node, ast.Attribute):
        return node.attr == "environ"
    return isinstance(node, ast.Name) and (node.id == "environ" or node.id in aliases)


def _environ_aliases(tree: ast.AST) -> set[str]:
    """Names bound to the environment mapping: ``env = os.environ``, ``import environ as e``."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_environ(node.value, aliases):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            aliases.update(a.asname or a.name for a in node.names if a.name == "environ")
    return aliases


def _is_write(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Subscript):
        return _is_environ(node.value, aliases) and isinstance(node.ctx, ast.Store | ast.Del)
    if isinstance(node, ast.AugAssign):
        return _is_environ(node.target, aliases)
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _WRITE_FUNCTIONS or (
            _is_environ(func.value, aliases) and func.attr in _WRITE_METHODS
        )
    return isinstance(func, ast.Name) and func.id in _WRITE_FUNCTIONS


def _writes_in(source: str) -> list[tuple[int, str]]:
    """``(line, function)`` for each place *source* mutates the environment.

    The function is the dotted name of the innermost enclosing function, or
    ``<module>`` for top-level code.
    """
    tree = ast.parse(source)
    aliases = _environ_aliases(tree)
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = child.name if scope == "<module>" else f"{scope}.{child.name}"
                visit(child, name)
                continue
            if _is_write(child, aliases):
                found.append((child.lineno, scope))
            visit(child, scope)

    visit(tree, "<module>")
    return sorted(found)


def _offenders(sources: dict[str, str]) -> set[str]:
    return {
        f"{rel}:{line} in {scope}"
        for rel, text in sources.items()
        for line, scope in _writes_in(text)
        if (rel, scope) not in _STARTUP_WRITES
    }


def _tree_sources() -> dict[str, str]:
    return {
        path.relative_to(SOURCE).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(SOURCE.rglob("*.py"))
    }


def test_no_module_writes_the_process_environment_after_startup() -> None:
    offenders = _offenders(_tree_sources())
    assert not offenders, f"code writes os.environ at {sorted(offenders)}"


def test_every_startup_write_entry_still_has_a_write() -> None:
    """A stale entry would let a future write back in unnoticed."""
    live = {(rel, scope) for rel, text in _tree_sources().items() for _, scope in _writes_in(text)}
    assert set(_STARTUP_WRITES) <= live


def test_a_new_write_in_an_allowed_file_goes_red() -> None:
    sources = _tree_sources()
    assert not _offenders(sources)
    # The rest of the tree is clean (asserted above), so only the probed file is rescanned.
    for rel, _scope in _STARTUP_WRITES:
        probed = sources[rel] + "\n\ndef _from_a_thread() -> None:\n    os.environ['X'] = '1'\n"
        assert _offenders({rel: probed}) == {f"{rel}:{probed.count(chr(10))} in _from_a_thread"}


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
        "from os import environ, putenv\n"
        "environ['F'] = '1'\n"
        "putenv('G', '1')\n"
        "env = os.environ\n"
        "env['H'] = '1'\n"
        "env.setdefault('I', '1')\n"
        "os.environ |= {'J': '1'}\n"
        "os.environ.__setitem__('K', '1')\n"
        "from os import environ as aliased\n"
        "aliased['L'] = '1'\n"
        "def inner():\n"
        "    os.environ['M'] = '1'\n"
        "class C:\n"
        "    def method(self):\n"
        "        os.environ['N'] = '1'\n"
    )
    assert _writes_in(source) == [
        (2, "<module>"),
        (3, "<module>"),
        (4, "<module>"),
        (5, "<module>"),
        (6, "<module>"),
        (7, "<module>"),
        (8, "<module>"),
        (12, "<module>"),
        (13, "<module>"),
        (15, "<module>"),
        (16, "<module>"),
        (17, "<module>"),
        (18, "<module>"),
        (20, "<module>"),
        (22, "inner"),
        (25, "method"),
    ]


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
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: binary)
    monkeypatch.setattr(planning, "_plan_devices", lambda _binary, engine: [])
    monkeypatch.setattr(planning, "_launched_roles", lambda engine: ())
    monkeypatch.setattr(
        planning, "plan_launches", lambda *_a, **_k: planning.FleetPlan(launches=())
    )

    recording = _RecordingEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", recording)
    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.append(
            planning.plan_all_launches(engines=planning.engines_for_all_roles(LLAMA))
        ),
        name="fleet-warm-up",
    )
    thread.start()
    thread.join(timeout=30)

    assert [plan.launches for plan in result] == [()]
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


def test_a_load_warning_logs_once_without_touching_the_environment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warned-once memory lives in the process, not in a variable."""
    from lilbee.core.config.load_warnings import warn_on_load

    recording = _RecordingEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", recording)

    with caplog.at_level("WARNING"):
        warn_on_load("bad value")
        warn_on_load("bad value")
        warn_on_load("bad value", "v2")

    assert [r.getMessage() for r in caplog.records] == ["bad value", "bad value"]
    assert recording.writes == []


def test_a_shard_worker_is_pinned_without_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The readers take the card, the engine dir and the CPU share from the slice state."""
    from lilbee.runtime import cpu, engine_lock
    from lilbee.runtime.shard_process import ShardProcessState

    state = ShardProcessState()
    for module in (cpu, engine_lock, gpu_env):
        monkeypatch.setattr(module, "shard_process", state)
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "99")
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "other"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    recording = _RecordingEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", recording)

    state.bind(tmp_path / "slot", 4, {"CUDA_VISIBLE_DEVICES": "2"})

    assert cpu.cpu_quota() == 4
    assert engine_lock.machine_engine_dir() == tmp_path / "slot"
    assert gpu_env.gpu_environ()["CUDA_VISIBLE_DEVICES"] == "2"
    assert recording.writes == []
