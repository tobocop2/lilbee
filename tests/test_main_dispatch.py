"""Tests for the frozen-binary dispatchers in `lilbee.__main__`."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from lilbee import __main__ as main_mod


class TestIsolateVendoredOpenssl:
    """Default OPENSSL_CONF to the empty config only in frozen Flatpak runs."""

    def test_sets_devnull_when_frozen_in_flatpak(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
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
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", marker),
            mock.patch.dict(os.environ, {"OPENSSL_CONF": "/etc/custom.cnf"}, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert os.environ["OPENSSL_CONF"] == "/etc/custom.cnf"

    def test_noop_outside_flatpak(self, tmp_path: Path) -> None:
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", tmp_path / "absent"),
            mock.patch.dict(os.environ, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert "OPENSSL_CONF" not in os.environ

    def test_noop_when_not_frozen(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        # sys.frozen does not exist in a regular interpreter; no patching needed.
        with (
            mock.patch.object(main_mod.sys, "platform", "linux"),
            mock.patch.object(main_mod, "_FLATPAK_INFO", marker),
            mock.patch.dict(os.environ, clear=True),
        ):
            main_mod._isolate_vendored_openssl()
            assert "OPENSSL_CONF" not in os.environ

    def test_noop_on_other_platforms(self, tmp_path: Path) -> None:
        marker = tmp_path / ".flatpak-info"
        marker.touch()
        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
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


class TestDispatchFrozenChild:
    """Run multiprocessing `-c` payloads inside the frozen exe."""

    def test_returns_false_when_not_frozen(self) -> None:
        argv = ["bin", "-c", "from multiprocessing.x import main;main()"]
        with (
            mock.patch.object(main_mod.sys, "frozen", False, create=True),
            mock.patch.object(main_mod.sys, "argv", argv),
        ):
            assert main_mod._dispatch_frozen_child() is False

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
    """Route `[bin, -m, lilbee.<module>, ...]` to runpy before typer sees it."""

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

    def test_routes_lilbee_module_through_runpy(self) -> None:
        argv_in = ["bin", "-m", "lilbee.core.system", "extra"]
        captured: dict[str, object] = {}

        def fake_run_module(name: str, *, run_name: str, alter_sys: bool) -> None:
            captured["name"] = name
            captured["run_name"] = run_name
            captured["alter_sys"] = alter_sys
            captured["argv_at_call"] = list(main_mod.sys.argv)

        with (
            mock.patch.object(main_mod.sys, "frozen", True, create=True),
            mock.patch.object(main_mod.sys, "argv", argv_in),
            mock.patch("runpy.run_module", side_effect=fake_run_module),
        ):
            assert main_mod._dispatch_module_invocation() is True

        assert captured["name"] == "lilbee.core.system"
        assert captured["run_name"] == "__main__"
        assert captured["alter_sys"] is True
        assert captured["argv_at_call"] == ["lilbee.core.system", "extra"]
