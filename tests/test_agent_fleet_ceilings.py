"""Ceilings that cap how many agents one daemon serves, short of the hardware."""

from __future__ import annotations

import types
from unittest import mock

from tests._sys_modules import inject_modules

# The resource module is POSIX-only, so the file-descriptor tests run against a
# stand-in rather than importing it: that keeps them (and the coverage of the
# warning's body) working on Windows, where the real module is absent.
_UNLIMITED = -1


def _fake_resource(*, soft: int, hard: int) -> dict[str, types.SimpleNamespace]:
    mod = types.SimpleNamespace(
        RLIMIT_NOFILE=7, RLIM_INFINITY=_UNLIMITED, getrlimit=lambda _res: (soft, hard)
    )
    return {"resource": mod}


class TestThreadPoolCeiling:
    """Synchronous work is offloaded off the event loop, and anyio's default pool
    holds 40 threads.

    That is the real ceiling on how many agents one daemon serves: past it,
    retrieval calls queue while the disk and CPU sit idle.
    """

    async def test_the_pool_is_resized_to_the_setting(self, monkeypatch) -> None:
        import anyio.to_thread

        from lilbee.core.config import cfg
        from lilbee.server import app as app_mod

        limiter = anyio.to_thread.current_default_thread_limiter()
        original = limiter.total_tokens
        monkeypatch.setattr(cfg, "mcp_tool_threads", 200)
        try:
            app_mod._raise_thread_pool_ceiling()
            assert limiter.total_tokens == 200
        finally:
            limiter.total_tokens = original

    async def test_resizing_anyios_own_limiter_lifts_every_offload(self, monkeypatch) -> None:
        """A private limiter would raise the ceiling only for lilbee's handlers
        and leave Litestar's and the MCP SDK's pinned at the default."""
        import anyio.to_thread

        from lilbee.core.config import cfg
        from lilbee.server import app as app_mod

        limiter = anyio.to_thread.current_default_thread_limiter()
        original = limiter.total_tokens
        monkeypatch.setattr(cfg, "mcp_tool_threads", 77)
        try:
            app_mod._raise_thread_pool_ceiling()
            assert anyio.to_thread.current_default_thread_limiter().total_tokens == 77
        finally:
            limiter.total_tokens = original

    async def test_a_pool_already_the_right_size_is_left_alone(self, monkeypatch) -> None:
        import anyio.to_thread

        from lilbee.core.config import cfg
        from lilbee.server import app as app_mod

        limiter = anyio.to_thread.current_default_thread_limiter()
        original = limiter.total_tokens
        monkeypatch.setattr(cfg, "mcp_tool_threads", int(original))
        try:
            app_mod._raise_thread_pool_ceiling()
            assert limiter.total_tokens == original
        finally:
            limiter.total_tokens = original

    def test_the_default_keeps_anyios_ceiling(self) -> None:
        """Unset, nothing changes: the pool is the size it has always been."""
        from lilbee.core.config.model import Config

        assert Config.model_fields["mcp_tool_threads"].default == 40

    def test_reapply_is_a_noop_without_a_running_server(self, monkeypatch) -> None:
        from lilbee.server import app as app_mod

        holder = app_mod._ServerLoop()  # no loop set
        monkeypatch.setattr(app_mod, "_server_loop", holder)

        app_mod.reapply_thread_pool_ceiling()  # no server loop: nothing to resize, no raise

    def test_reapply_marshals_the_resize_onto_the_server_loop(self, monkeypatch) -> None:
        from lilbee.server import app as app_mod

        scheduled: list[object] = []
        loop = mock.MagicMock()
        loop.is_closed.return_value = False
        loop.call_soon_threadsafe.side_effect = scheduled.append
        holder = app_mod._ServerLoop()
        holder.set(loop)
        monkeypatch.setattr(app_mod, "_server_loop", holder)

        app_mod.reapply_thread_pool_ceiling()

        assert scheduled == [app_mod._raise_thread_pool_ceiling]

    def test_reapply_ignores_a_closed_loop(self, monkeypatch) -> None:
        from lilbee.server import app as app_mod

        loop = mock.MagicMock()
        loop.is_closed.return_value = True
        holder = app_mod._ServerLoop()
        holder.set(loop)
        monkeypatch.setattr(app_mod, "_server_loop", holder)

        app_mod.reapply_thread_pool_ceiling()

        loop.call_soon_threadsafe.assert_not_called()

    def test_changing_the_thread_count_reapplies_the_ceiling(self, monkeypatch) -> None:
        from lilbee.app import settings as settings_mod
        from lilbee.server import app as app_mod

        called: list[bool] = []
        monkeypatch.setattr(app_mod, "reapply_thread_pool_ceiling", lambda: called.append(True))

        settings_mod._invalidate_caches({"mcp_tool_threads"})

        assert called == [True]

    async def test_a_sync_handler_is_still_offloaded(self, monkeypatch) -> None:
        """The offload itself is unchanged; only the pool it lands in is sized."""
        from lilbee import mcp_server

        offloaded = mcp_server._offload_sync(lambda: "done")

        assert await offloaded() == "done"

    def test_an_async_handler_is_left_alone(self) -> None:
        """It already yields; wrapping it would buy a thread for nothing."""
        from lilbee import mcp_server

        async def _handler() -> str:
            return "x"

        assert mcp_server._offload_sync(_handler) is _handler


class TestFileDescriptorNudge:
    """Each connected agent holds a socket; macOS still defaults to 256."""

    def _run(self, caplog, *, soft: int, hard: int = _UNLIMITED) -> str:
        from lilbee.server import app as app_mod

        with inject_modules(_fake_resource(soft=soft, hard=hard)), caplog.at_level(
            "INFO", logger=app_mod.__name__
        ):
            app_mod._warn_if_few_file_descriptors()
        return caplog.text

    def test_a_low_limit_is_named_with_the_command_to_raise_it(self, caplog) -> None:
        text = self._run(caplog, soft=256)

        assert "256" in text
        assert "ulimit -n" in text

    def test_a_generous_limit_says_nothing(self, caplog) -> None:
        assert self._run(caplog, soft=65536) == ""

    def test_an_unlimited_soft_limit_says_nothing(self, caplog) -> None:
        assert self._run(caplog, soft=_UNLIMITED) == ""

    def test_the_hard_limit_is_reported_so_the_operator_knows_the_ceiling(self, caplog) -> None:
        text = self._run(caplog, soft=256, hard=10240)

        assert "10240" in text

    def test_an_unlimited_hard_limit_reads_as_unlimited(self, caplog) -> None:
        text = self._run(caplog, soft=256, hard=_UNLIMITED)

        assert "unlimited" in text

    def test_a_platform_without_the_resource_module_is_silent(self, monkeypatch, caplog) -> None:
        """Windows has no resource module, and the daemon still has to start."""
        import builtins

        from lilbee.server import app as app_mod

        real_import = builtins.__import__

        def _no_resource(name, *args, **kwargs):
            if name == "resource":
                raise ImportError("no resource module on this platform")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_resource)
        with caplog.at_level("INFO", logger=app_mod.__name__):
            app_mod._warn_if_few_file_descriptors()

        assert caplog.text == ""
