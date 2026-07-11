"""A warm is in flight from the moment it is requested, not from its first phase."""

from __future__ import annotations

import threading
from unittest import mock

from lilbee.providers.base import LLMProvider


def test_base_provider_reports_no_pending_warm():
    """Providers without managed servers have nothing to wait for."""
    assert LLMProvider.warm_pending(mock.MagicMock()) is False


def test_fleet_reports_a_pending_warm_before_any_phase_is_stamped():
    """Regression: warm_up_pool spawns the fleet before the tracker stamps STARTING.

    A surface that gates on warm_progress() alone sees None during the spawn and
    concludes nothing is warming, which is how the startup gate stepped aside
    while llama-server had not even been launched.
    """
    from lilbee.providers.fleet.provider import FleetProvider

    provider = FleetProvider.__new__(FleetProvider)
    provider._lock = threading.RLock()
    provider._swaps = {}
    provider._warming = False
    provider._warm_tracker = mock.MagicMock()
    provider._warm_tracker.snapshot.return_value = None

    assert provider.warm_pending() is False

    released = threading.Event()
    with mock.patch.object(FleetProvider, "_warm_up_blocking", lambda self: released.wait(5)):
        provider.warm_up_pool()
        # The tracker has stamped nothing yet, exactly as in production.
        assert provider.warm_progress() is None
        assert provider.warm_pending() is True, "a requested warm must read as pending"
        released.set()


def test_routing_provider_forwards_pending_warm_to_the_local_engine():
    """The routing wrapper must not hide the native side's pending warm."""
    from lilbee.providers.routing_provider import RoutingProvider

    provider = RoutingProvider.__new__(RoutingProvider)
    local = mock.MagicMock()
    local.warm_pending.return_value = True
    with mock.patch.object(RoutingProvider, "_get_local", return_value=local):
        assert provider.warm_pending() is True
