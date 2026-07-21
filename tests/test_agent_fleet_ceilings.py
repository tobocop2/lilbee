"""Ceilings that cap how many agents one daemon serves, short of the hardware."""

from __future__ import annotations

import resource


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
