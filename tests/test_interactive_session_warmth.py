"""An interactive session (the TUI) holds its fleet resident for the session.

The intent is recorded on the services state before anything builds the
container, threaded through the provider factory, and ends up as provider
instance state -- so the ttl decision reads the provider it belongs to rather
than a process-wide flag.
"""

from __future__ import annotations

from lilbee.providers.fleet.provider import FleetProvider


def test_provider_defaults_to_not_holding_warm() -> None:
    assert FleetProvider()._hold_warm_for_session is False


def test_provider_carries_the_session_hold() -> None:
    assert FleetProvider(hold_warm=True)._hold_warm_for_session is True


def test_routing_provider_hands_the_hold_to_its_fleet() -> None:
    from lilbee.providers.routing_provider import RoutingProvider

    assert RoutingProvider(hold_warm=True)._get_local()._hold_warm_for_session is True
    assert RoutingProvider()._get_local()._hold_warm_for_session is False


def test_factory_threads_the_hold_from_the_container(monkeypatch) -> None:
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import LlmProvider
    from lilbee.providers.factory import create_provider

    monkeypatch.setattr(cfg, "llm_provider", LlmProvider.AUTO)
    assert create_provider(cfg, hold_warm=True)._get_local()._hold_warm_for_session is True


def test_mark_interactive_session_records_build_intent(monkeypatch) -> None:
    from lilbee.app import services as services_mod

    monkeypatch.setattr(services_mod._state, "interactive", False)
    services_mod.mark_interactive_session()
    assert services_mod._state.interactive is True
