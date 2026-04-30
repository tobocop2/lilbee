"""Tests for the frozen-binary dispatchers in `lilbee.__main__`."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from lilbee import __main__ as main_mod


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
            mock.patch.object(main_mod.sys, "argv", ["bin", "-m", "lilbee.system"]),
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
        argv_in = ["bin", "-m", "lilbee.system", "extra"]
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

        assert captured["name"] == "lilbee.system"
        assert captured["run_name"] == "__main__"
        assert captured["alter_sys"] is True
        assert captured["argv_at_call"] == ["lilbee.system", "extra"]
