"""Ceilings that cap how many agents one daemon serves, short of the hardware."""

from __future__ import annotations

import resource


class TestToolThreadLimiter:
    """Sync MCP tool handlers are offloaded off the event loop into a pool.

    anyio's default holds 40 threads, so a fleet larger than that queues
    retrieval calls while the disk and CPU sit idle.
    """

    def _limiter(self, monkeypatch, threads: int):
        from lilbee import mcp_server
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "mcp_tool_threads", threads)
        mcp_server._tool_thread_limiter.cache_clear()
        try:
            return mcp_server._tool_thread_limiter()
        finally:
            mcp_server._tool_thread_limiter.cache_clear()

    def test_the_pool_is_sized_by_the_setting(self, monkeypatch) -> None:
        assert self._limiter(monkeypatch, 200).total_tokens == 200

    def test_the_default_keeps_anyios_ceiling(self, monkeypatch) -> None:
        """Unset, nothing changes: the pool is the size it has always been."""
        from lilbee.core.config.model import Config

        assert Config.model_fields["mcp_tool_threads"].default == 40

    def test_every_handler_shares_one_pool(self, monkeypatch) -> None:
        """A limiter per call would be no limit at all."""
        from lilbee import mcp_server
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "mcp_tool_threads", 12)
        mcp_server._tool_thread_limiter.cache_clear()
        try:
            assert mcp_server._tool_thread_limiter() is mcp_server._tool_thread_limiter()
        finally:
            mcp_server._tool_thread_limiter.cache_clear()

    async def test_a_sync_handler_runs_against_that_pool(self, monkeypatch) -> None:
        """The limiter has to reach the offload call, not merely exist."""
        from lilbee import mcp_server

        seen: dict[str, object] = {}

        async def _fake_run_sync(func, *, limiter=None):
            seen["limiter"] = limiter
            return func()

        monkeypatch.setattr(mcp_server.anyio.to_thread, "run_sync", _fake_run_sync)
        offloaded = mcp_server._offload_sync(lambda: "done")

        assert await offloaded() == "done"
        assert seen["limiter"] is mcp_server._tool_thread_limiter()

    def test_an_async_handler_is_left_alone(self) -> None:
        """It already yields; wrapping it would buy a thread for nothing."""
        from lilbee import mcp_server

        async def _handler() -> str:
            return "x"

        assert mcp_server._offload_sync(_handler) is _handler


class TestFileDescriptorNudge:
    """Each connected agent holds a socket; macOS still defaults to 256."""

    def _run(self, monkeypatch, caplog, *, soft: int, hard: int = resource.RLIM_INFINITY):
        from lilbee.server import app as app_mod

        monkeypatch.setattr(resource, "getrlimit", lambda _res: (soft, hard))
        with caplog.at_level("INFO", logger=app_mod.__name__):
            app_mod._warn_if_few_file_descriptors()
        return caplog.text

    def test_a_low_limit_is_named_with_the_command_to_raise_it(self, monkeypatch, caplog) -> None:
        text = self._run(monkeypatch, caplog, soft=256)

        assert "256" in text
        assert "ulimit -n" in text

    def test_a_generous_limit_says_nothing(self, monkeypatch, caplog) -> None:
        assert self._run(monkeypatch, caplog, soft=65536) == ""

    def test_an_unlimited_soft_limit_says_nothing(self, monkeypatch, caplog) -> None:
        assert self._run(monkeypatch, caplog, soft=resource.RLIM_INFINITY) == ""

    def test_the_hard_limit_is_reported_so_the_operator_knows_the_ceiling(
        self, monkeypatch, caplog
    ) -> None:
        text = self._run(monkeypatch, caplog, soft=256, hard=10240)

        assert "10240" in text

    def test_an_unlimited_hard_limit_reads_as_unlimited(self, monkeypatch, caplog) -> None:
        text = self._run(monkeypatch, caplog, soft=256, hard=resource.RLIM_INFINITY)

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
