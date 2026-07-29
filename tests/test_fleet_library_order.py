"""The engine's own libraries must win over anything a wheel installed."""

from __future__ import annotations

import os

import pytest

from lilbee.providers.fleet import cuda_runtime


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)


class TestTheEngineDirectoryComesFirst:
    """$ORIGIN lands in DT_RUNPATH, which is searched after LD_LIBRARY_PATH.

    So prepending a wheel directory shadows the libraries the bundled engine
    ships beside itself, and a host that merely has torch installed silently
    swapped out the engine's CUDA runtime for torch's.
    """

    def test_the_binary_s_directory_precedes_every_wheel_dir(self, monkeypatch, tmp_path) -> None:
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        engine = tmp_path / "engine"
        engine.mkdir()
        binary = engine / "llama-server"
        binary.touch()
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: True)
        monkeypatch.setattr(cuda_runtime, "_ships_its_own_cuda_runtime", lambda _b: False)

        path = cuda_runtime.cuda_runtime_env(binary)["LD_LIBRARY_PATH"].split(os.pathsep)

        assert path.index(str(engine)) < path.index(str(wheel))

    def test_a_bundled_engine_with_its_own_runtime_gets_no_wheel_dirs(
        self, monkeypatch, tmp_path
    ) -> None:
        # Its libraries are already beside it; adding a wheel dir can only shadow them.
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        engine = tmp_path / "engine"
        engine.mkdir()
        binary = engine / "llama-server"
        binary.touch()
        (engine / "libcudart.so.12").touch()
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: True)

        path = cuda_runtime.cuda_runtime_env(binary)["LD_LIBRARY_PATH"].split(os.pathsep)

        assert str(engine) in path
        assert str(wheel) not in path

    def test_a_non_cuda_binary_gets_no_wheel_dirs(self, monkeypatch, tmp_path) -> None:
        # A Vulkan or CPU build has no use for them, and putting them on its path
        # only gives an unrelated torch install a way to interfere.
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        binary = tmp_path / "llama-server"
        binary.touch()
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: False)

        path = cuda_runtime.cuda_runtime_env(binary).get("LD_LIBRARY_PATH", "").split(os.pathsep)

        assert str(wheel) not in path

    def test_the_callers_existing_path_is_kept_behind_both(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/mine")
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        engine = tmp_path / "engine"
        engine.mkdir()
        binary = engine / "llama-server"
        binary.touch()
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: True)
        monkeypatch.setattr(cuda_runtime, "_ships_its_own_cuda_runtime", lambda _b: False)

        path = cuda_runtime.cuda_runtime_env(binary)["LD_LIBRARY_PATH"].split(os.pathsep)

        assert path == [str(engine), str(wheel), "/opt/mine"]

    def test_without_a_binary_the_old_wheel_only_answer_stands(self, monkeypatch, tmp_path) -> None:
        # Callers that have no binary to reason about still get the wheel dirs.
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        assert cuda_runtime.cuda_runtime_env()["LD_LIBRARY_PATH"].startswith(str(wheel))


class TestAnUnreadableEngineDirectory:
    """A directory the process cannot list is not a reason to fail a launch."""

    def test_it_reads_as_shipping_no_runtime_of_its_own(self, monkeypatch, tmp_path) -> None:
        missing = tmp_path / "gone" / "llama-server"
        assert cuda_runtime._ships_its_own_cuda_runtime(missing) is False

    def test_the_engine_dir_is_still_placed_first(self, monkeypatch, tmp_path) -> None:
        # Unreadable or not, its own directory belongs ahead of any wheel.
        wheel = tmp_path / "wheel-lib"
        wheel.mkdir()
        binary = tmp_path / "gone" / "llama-server"
        monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: True)
        path = cuda_runtime.cuda_runtime_env(binary)["LD_LIBRARY_PATH"].split(os.pathsep)
        assert path[0] == str(binary.parent)
