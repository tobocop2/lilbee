"""Tests for the refcounted ingest bracket in ``lilbee.providers.fleet.ingest_scale``."""

from __future__ import annotations

import contextlib
from unittest import mock

from lilbee.providers.fleet import ingest_scale


def test_release_only_after_last_ingest(monkeypatch):
    """``release_ingest_pool`` is called exactly once, when the outermost bracket exits."""
    released = {"n": 0}
    prov = mock.Mock()
    prov.release_ingest_pool.side_effect = lambda: released.__setitem__("n", released["n"] + 1)
    monkeypatch.setattr(ingest_scale, "_provider", lambda: prov)

    with ingest_scale.ingest_scale():
        with ingest_scale.ingest_scale():
            assert released["n"] == 0  # inner exit does not release
        assert released["n"] == 0  # still one active
    assert released["n"] == 1  # last exit releases once


def test_release_runs_even_on_exception(monkeypatch):
    """``release_ingest_pool`` is called even when the body raises."""
    prov = mock.Mock()
    monkeypatch.setattr(ingest_scale, "_provider", lambda: prov)
    with contextlib.suppress(RuntimeError), ingest_scale.ingest_scale():
        raise RuntimeError("boom")
    prov.release_ingest_pool.assert_called_once()


def test_release_not_called_on_non_last_exit(monkeypatch):
    """Inner bracket exit does not call ``release_ingest_pool``."""
    prov = mock.Mock()
    monkeypatch.setattr(ingest_scale, "_provider", lambda: prov)
    with ingest_scale.ingest_scale():
        with ingest_scale.ingest_scale():
            pass
        prov.release_ingest_pool.assert_not_called()
    prov.release_ingest_pool.assert_called_once()


def test_routing_provider_release_ingest_pool_forwards_to_local():
    """``RoutingProvider.release_ingest_pool`` forwards to the inner engine."""
    from lilbee.providers.routing_provider import RoutingProvider

    rp = RoutingProvider()
    local = mock.MagicMock()
    rp._local = local
    rp.release_ingest_pool()
    local.release_ingest_pool.assert_called_once_with()


def test_routing_provider_release_ingest_pool_noop_without_local():
    """``RoutingProvider.release_ingest_pool`` is a no-op when ``_local`` is None."""
    from lilbee.providers.routing_provider import RoutingProvider

    rp = RoutingProvider()  # _local is None
    rp.release_ingest_pool()  # must not raise


def test_sdk_llm_provider_release_ingest_pool_is_callable_noop():
    """``SdkLLMProvider.release_ingest_pool`` is a callable no-op."""
    from lilbee.providers.sdk_backend import LlmSdkBackend
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = mock.MagicMock(spec=LlmSdkBackend)
    provider = SdkLLMProvider(backend)
    provider.release_ingest_pool()  # must not raise
