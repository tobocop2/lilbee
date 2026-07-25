"""Tests for attach-only engine access (ingest workers must never build a fleet)."""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.guest import (
    NO_ENGINE_TO_ATTACH,
    bind_only_active,
    bind_only_engine,
)


class TestBindOnlyFlag:
    def test_off_by_default_so_normal_startup_still_builds(self):
        assert bind_only_active() is False

    def test_scoped_to_the_block(self):
        with bind_only_engine():
            assert bind_only_active() is True
        assert bind_only_active() is False

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError), bind_only_engine():
            raise RuntimeError("boom")
        assert bind_only_active() is False

    def test_a_plain_thread_sees_the_flag(self):
        """warm_up_pool runs the acquisition ladder on a threading.Thread it spawns.

        A new thread starts with a fresh context, so a ContextVar-backed flag read
        its default there and the worker built a fleet anyway -- the exact bug this
        guard exists to stop.
        """
        import threading

        seen: list[bool] = []
        with bind_only_engine():
            t = threading.Thread(target=lambda: seen.append(bind_only_active()))
            t.start()
            t.join()
        assert seen == [True]

    def test_nesting_does_not_clear_the_outer_scope(self):
        with bind_only_engine():
            with bind_only_engine():
                assert bind_only_active() is True
            assert bind_only_active() is True
        assert bind_only_active() is False

    async def test_reaches_the_ingest_offload_pool(self):
        """Workers embed via to_ingest_thread, so the guard must hold on that thread too."""
        from lilbee.data.ingest.offload import to_ingest_thread

        with bind_only_engine():
            assert await to_ingest_thread(bind_only_active) is True
        assert await to_ingest_thread(bind_only_active) is False


class TestAcquireRefusesToBuild:
    """The gate sits after the bind attempt, so a failed bind cannot fall through."""

    @staticmethod
    def _provider_with_failed_bind(monkeypatch):
        from lilbee.providers.fleet import provider as prov

        monkeypatch.setattr(prov, "_healthy_states", lambda d: {})
        monkeypatch.setattr(prov, "build_lock", lambda d: __import__("contextlib").nullcontext())
        instance = object.__new__(prov.FleetProvider)
        monkeypatch.setattr(
            prov.FleetProvider,
            "_bind_all_in_dir",
            lambda self, *a, **k: False,
        )
        return instance, prov

    def test_raises_instead_of_building_when_bind_only(self, monkeypatch, tmp_path):
        instance, prov = self._provider_with_failed_bind(monkeypatch)

        def _must_not_run(*args, **kwargs):
            raise AssertionError("a bind-only process must never build an engine")

        monkeypatch.setattr(prov, "live_users_exist", _must_not_run)

        with bind_only_engine(), pytest.raises(ProviderError) as excinfo:
            instance._acquire_in_dir(tmp_path, "pin", {("embed", "m")}, is_overflow=False)

        assert excinfo.value.kind is ProviderErrorKind.SERVER
        assert str(excinfo.value) == NO_ENGINE_TO_ATTACH

    def test_error_names_the_problem_without_leaking_internals(self):
        lowered = NO_ENGINE_TO_ATTACH.lower()
        for leak in ("dispatch", "resolve", "_acquire", "bind_only", "routed", "contextvar"):
            assert leak not in lowered

    def test_without_the_guard_the_build_path_is_reached(self, monkeypatch, tmp_path):
        """Pins that the gate is what stops it, not an unrelated early return."""
        instance, prov = self._provider_with_failed_bind(monkeypatch)
        reached = []
        monkeypatch.setattr(prov, "live_users_exist", lambda d: reached.append(d) or False)
        monkeypatch.setattr(prov, "_healthy_groups_ours", lambda *a, **k: False)
        monkeypatch.setattr(prov, "_can_build_engine", lambda wanted: False)

        instance._acquire_in_dir(tmp_path, "pin", {("embed", "m")}, is_overflow=False)

        assert reached == [tmp_path]
