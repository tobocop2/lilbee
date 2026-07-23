def test_active_chat_warm_progress_skips_the_probe_when_nothing_is_warming():
    """The task bar polls this every tick, so an idle TUI must not fire an HTTP
    readiness probe forever -- the free in-process snapshot gates it."""
    from unittest import mock

    from lilbee.app.placement import active_chat_warm_progress
    from lilbee.app.services import set_services

    services = mock.MagicMock()
    services.provider.warm_progress.return_value = None  # nothing warming
    set_services(services)
    try:
        assert active_chat_warm_progress() is None
        services.provider.role_ready.assert_not_called()
    finally:
        set_services(None)
