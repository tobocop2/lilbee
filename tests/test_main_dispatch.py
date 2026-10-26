"""Tests for the frozen-binary dispatchers in `lilbee.__main__`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee import __main__ as main_mod

_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Windows has no resource tracker")


class TestIsolateVendoredOpenssl:
    """Default OPENSSL_CONF to the empty config only inside Flatpak sandboxes."""

    def test_sets_devnull_inside_flatpak(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        with (
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", marker),
            mock.patch.dict(os.environ, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert os.environ["OPENSSL_CONF"] == os.devnull

    def test_preserves_explicit_openssl_conf(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        with (
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", marker),
            mock.patch.dict(os.environ, {"OPENSSL_CONF": "/etc/custom.cnf"}, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert os.environ["OPENSSL_CONF"] == "/etc/custom.cnf"

    def test_noop_outside_flatpak(self, tmp_path: Path) -> None:
        with (
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", tmp_path / "absent"),
            mock.patch.dict(os.environ, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert "OPENSSL_CONF" not in os.environ

    def test_noop_on_other_platforms(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        with (
            mock.patch.object(main_mod.sys, "platform", "darwin"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", marker),
            mock.patch.dict(os.environ, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert "OPENSSL_CONF" not in os.environ


class TestMultiprocessingChildCode:
    """Extract the `-c "from multiprocessing..."` payload from a frozen reinvocation."""

    def test_returns_payload_for_multiprocessing_marker(self) -> None:
        argv = [
            "bin",
            "-B",
            "-s",
            "-E",
            "-c",
            "from multiprocessing.resource_tracker import main;main(7)",
        ]
        assert (
            main_mod._multiprocessing_child_code(argv)
            == "from multiprocessing.resource_tracker import main;main(7)"
        )

    def test_returns_payload_for_spawn_main_marker(self) -> None:
        argv = [
            "bin",
            "-c",
            "from multiprocessing.spawn import spawn_main;spawn_main(parent_pid=1)",
        ]
        assert main_mod._multiprocessing_child_code(argv) is not None

    def test_returns_none_when_no_dash_c(self) -> None:
        assert main_mod._multiprocessing_child_code(["bin", "--version"]) is None

    def test_returns_none_when_dash_c_at_end(self) -> None:
        assert main_mod._multiprocessing_child_code(["bin", "-c"]) is None

    def test_returns_none_when_payload_lacks_marker(self) -> None:
        assert main_mod._multiprocessing_child_code(["bin", "-c", "print('hi')"]) is None


class TestIsFrozen:
    """Detect frozen builds via sys.frozen (PyInstaller) or __compiled__ (Nuitka)."""

    def test_true_when_sys_frozen_set(self) -> None:
        with mock.patch.object(main_mod.sys, "frozen", True, create=True):
            assert main_mod._is_frozen() is True

    def test_true_when_nuitka_compiled_marker_present(self) -> None:
        # Nuitka injects __compiled__ into every module but never sets sys.frozen.
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod, "__compiled__", object(), create=True),
        ):
            assert main_mod._is_frozen() is True

    def test_false_in_plain_interpreter(self) -> None:
        with mock.patch.object(main_mod.sys, "frozen", False, create=True):
            assert main_mod._is_frozen() is False


class TestSharedIsFrozen:
    """lilbee._frozen.is_frozen is the canonical helper for package-level code."""

    def test_true_when_sys_frozen_set(self) -> None:
        from lilbee import _frozen

        with mock.patch.object(_frozen.sys, "frozen", True, create=True):
            assert _frozen.is_frozen() is True

    def test_true_when_nuitka_compiled_marker_present(self) -> None:
        from lilbee import _frozen

        with (
            mock.patch.object(_frozen.sys, "frozen", False, create=True),
            mock.patch.object(_frozen, "__compiled__", object(), create=True),
        ):
            assert _frozen.is_frozen() is True

    def test_false_in_plain_interpreter(self) -> None:
        from lilbee import _frozen

        with mock.patch.object(_frozen.sys, "frozen", False, create=True):
            assert _frozen.is_frozen() is False


class TestDispatchFrozenChild:
    """Run multiprocessing `-c` payloads inside the frozen exe."""

    def test_returns_false_when_not_frozen(self) -> None:
        argv = ["bin", "-c", "from multiprocessing.x import main;main()"]
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod.sys, "argv", argv),
        ):
            assert main_mod._dispatch_frozen_child() is False

    def test_execs_payload_under_nuitka_without_sys_frozen(self, tmp_path: Path) -> None:
        # Regression: Nuitka onefile never sets sys.frozen, so gating the
        # dispatcher on it leaks the resource_tracker reinvocation into typer
        # ("No such command '3'"). __compiled__ is the Nuitka frozen marker.
        marker = tmp_path / "marker.txt"
        payload = (
            f"from multiprocessing.resource_tracker import main\n"
            f"open({str(marker)!r}, 'w').write('ran')"
        )
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod, "__compiled__", object(), create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "-c", payload]),
        ):
            assert main_mod._dispatch_frozen_child() is True
        assert marker.read_text() == "ran"

    def test_returns_false_when_no_payload(self) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "--version"]),
        ):
            assert main_mod._dispatch_frozen_child() is False

    def test_execs_payload_and_returns_true(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        payload = (
            f"from multiprocessing.spawn import spawn_main\nopen({str(marker)!r}, 'w').write('ran')"
        )
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "-c", payload]),
        ):
            assert main_mod._dispatch_frozen_child() is True
        assert marker.read_text() == "ran"


class TestDispatchModuleInvocation:
    """Route `[bin, -m, lilbee.<module>, ...]` to the module's main() before typer."""

    def test_returns_false_when_not_frozen(self) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "-m", "lilbee.core.system"]),
        ):
            assert main_mod._dispatch_module_invocation() is False

    def test_returns_false_when_argv_too_short(self) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "-m"]),
        ):
            assert main_mod._dispatch_module_invocation() is False

    def test_returns_false_when_no_dash_m(self) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "--version", "extra"]),
        ):
            assert main_mod._dispatch_module_invocation() is False

    def test_returns_false_for_non_lilbee_module(self) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", ["bin", "-m", "os", "extra"]),
        ):
            assert main_mod._dispatch_module_invocation() is False

    def test_routes_under_nuitka_without_sys_frozen(self) -> None:
        # Regression: same Nuitka frozen-detection gap as the mp-child dispatcher.
        argv_in = ["bin", "-m", "lilbee.core.system", "extra"]
        module = mock.Mock()
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod, "__compiled__", object(), create=True),
            mock.patch.object(main_mod.sys, "argv", argv_in),
            mock.patch.object(main_mod.importlib, "import_module", return_value=module) as imp,
        ):
            assert main_mod._dispatch_module_invocation() is True
        imp.assert_called_once_with("lilbee.core.system")
        module.main.assert_called_once_with()

    def test_runs_module_main_with_rewritten_argv(self) -> None:
        # The reinvocation calls the module's main() (never runpy, which fails
        # under Nuitka: its loader has no get_code). argv is stripped to the
        # module name plus the original trailing args so main() reads its fd.
        argv_in = ["bin", "-m", "lilbee.runtime._splash_runner", "999"]
        captured: dict[str, object] = {}
        module = mock.Mock()
        module.main.side_effect = lambda: captured.update(argv_at_call=list(main_mod.sys.argv))

        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", argv_in),
            mock.patch.object(main_mod.importlib, "import_module", return_value=module) as imp,
        ):
            assert main_mod._dispatch_module_invocation() is True

        imp.assert_called_once_with("lilbee.runtime._splash_runner")
        module.main.assert_called_once_with()
        assert captured["argv_at_call"] == ["lilbee.runtime._splash_runner", "999"]


_TRACKER_PID = "import multiprocessing.resource_tracker as rt; "

_FAKE_TEXTUAL_STDERR = """
import io, sys

class _NoFd(io.StringIO):
    def fileno(self):
        return -1

{prestart}
sys.stderr = _NoFd()
import multiprocessing
try:
    multiprocessing.get_context("spawn").Event()
except ValueError as exc:
    print("CRASH", exc)
else:
    print("OK")
"""


class TestImportStartsNoProcess:
    """Importing the package leaves the multiprocessing resource tracker unstarted."""

    def test_import_leaves_tracker_pid_none(self) -> None:
        code = (
            _TRACKER_PID
            + "before = rt._resource_tracker._pid; import lilbee; "
            + "print(before, rt._resource_tracker._pid)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=120,
        )
        assert out.stdout.strip() == "None None"

    @_POSIX_ONLY
    def test_tracker_pid_probe_sees_a_started_tracker(self) -> None:
        code = _TRACKER_PID + "rt.ensure_running(); print(rt._resource_tracker._pid is not None)"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=120,
        )
        assert out.stdout.strip() == "True"


class TestPrestartMpResourceTracker:
    """The TUI start path prestarts the tracker on POSIX, including frozen Nuitka builds."""

    def test_prestarts_under_nuitka_without_sys_frozen(self) -> None:
        # Regression (the chat "bad value(s) in fds_to_keep" crash): the prestart
        # was disabled in frozen builds, so the resource tracker launched lazily
        # at worker-spawn time after Textual swapped stderr (fileno -1). It must
        # run in frozen builds too; the tracker's -c reinvocation is intercepted
        # by __main__._dispatch_frozen_child.
        from multiprocessing import resource_tracker

        from lilbee import _frozen
        from lilbee.cli import tui

        called: list[int] = []
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(_frozen, "__compiled__", object(), create=True),
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(resource_tracker, "ensure_running", lambda: called.append(1)),
        ):
            tui._prestart_mp_resource_tracker()
        assert called == [1]

    def test_prestarts_on_posix_when_frozen(self) -> None:
        from multiprocessing import resource_tracker

        from lilbee.cli import tui

        called: list[int] = []
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(resource_tracker, "ensure_running", lambda: called.append(1)),
        ):
            tui._prestart_mp_resource_tracker()
        assert called == [1]

    def test_skips_on_windows(self) -> None:
        from multiprocessing import resource_tracker

        from lilbee.cli import tui

        called: list[int] = []
        with (
            mock.patch.object(main_mod.sys, "platform", "win32"),
            mock.patch.object(resource_tracker, "ensure_running", lambda: called.append(1)),
        ):
            tui._prestart_mp_resource_tracker()
        assert called == []

    def test_a_tracker_that_cannot_start_is_left_alone(self) -> None:
        from multiprocessing import resource_tracker

        from lilbee.cli import tui

        with (
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(
                resource_tracker, "ensure_running", side_effect=RuntimeError("simulated crash")
            ),
        ):
            tui._prestart_mp_resource_tracker()

    def test_run_tui_prestarts_before_stderr_is_redirected_and_the_app_runs(self) -> None:
        from lilbee.cli import tui
        from lilbee.cli.tui.app import LilbeeApp

        events: list[str] = []
        with (
            mock.patch.object(
                tui, "_prestart_mp_resource_tracker", lambda: events.append("prestart")
            ),
            mock.patch.object(
                tui, "_redirect_native_stderr_to", lambda _p: events.append("redirect")
            ),
            mock.patch.object(LilbeeApp, "run", lambda _self: events.append("run")),
            mock.patch.object(LilbeeApp, "__init__", lambda _self, **_kw: None),
            mock.patch.object(LilbeeApp, "task_bar", create=True),
            mock.patch.object(tui, "_restore_native_stderr"),
            mock.patch("lilbee.cli.sync.shutdown_executor"),
            mock.patch.object(tui, "reset_services_on_exit"),
            mock.patch.object(tui, "exit_when_stragglers_would_hang"),
        ):
            tui.run_tui()
        assert events == ["prestart", "redirect", "run"]

    @_POSIX_ONLY
    def test_spawn_primitive_survives_a_stderr_without_a_descriptor(self, tmp_path: Path) -> None:
        # Control: with no prestart the same stream crashes the tracker launch.
        def _run(prestart: str) -> str:
            script = tmp_path / "child.py"
            script.write_text(_FAKE_TEXTUAL_STDERR.format(prestart=prestart), encoding="utf-8")
            done = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=120,
            )
            return done.stdout.strip()

        assert _run("pass").startswith("CRASH bad value(s) in fds_to_keep")
        assert _run("from lilbee.cli.tui import _prestart_mp_resource_tracker as p; p()") == "OK"
