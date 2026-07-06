"""Verify ``create_app`` mounts the v1 chat-completions router."""

from __future__ import annotations

from lilbee.server.app import create_app


def _route_paths() -> set[str]:
    return {route.path for route in create_app().routes}


class TestV1RouterRegistration:
    def test_lists_v1_models(self) -> None:
        assert "/v1/models" in _route_paths()

    def test_lists_v1_chat_completions(self) -> None:
        assert "/v1/chat/completions" in _route_paths()
