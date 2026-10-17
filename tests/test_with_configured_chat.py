"""The listing helper puts a configured chat model the server serves first."""

from unittest.mock import patch

import pytest

from lilbee.app.models import with_configured_chat
from lilbee.app.services import set_services
from lilbee.modelhub.model_manager import discovery
from lilbee.modelhub.registry import ModelRegistry
from tests.conftest import make_mock_services


@pytest.fixture
def served(tmp_path):
    """Real registry and known-model lookup; one loose GGUF file and one missing path."""
    gguf = tmp_path / "loose" / "model-Q4.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    with (
        patch.object(discovery, "classify_all_remote_models", return_value=[]),
        patch.object(discovery, "discover_api_models", return_value={}),
    ):
        set_services(
            make_mock_services(
                registry=ModelRegistry(tmp_path / "models"),
                known_models=discovery.KnownModelCache(),
            )
        )
        yield gguf
    set_services(None)


class TestWithConfiguredChat:
    def test_remote_configured_chat_model_is_prepended(self, served) -> None:
        assert with_configured_chat(["a/b/c.gguf"], "ollama/qwen3:8b") == [
            "ollama/qwen3:8b",
            "a/b/c.gguf",
        ]

    def test_a_model_set_by_path_is_prepended(self, served) -> None:
        assert with_configured_chat(["a/b/c.gguf"], str(served)) == [str(served), "a/b/c.gguf"]

    def test_a_path_to_no_file_is_not_listed(self, served) -> None:
        missing = str(served.with_name("absent.gguf"))
        assert with_configured_chat(["a/b/c.gguf"], missing) == ["a/b/c.gguf"]

    def test_an_installed_model_is_not_duplicated(self, served) -> None:
        assert with_configured_chat(["a/b/c.gguf"], "a/b/c.gguf") == ["a/b/c.gguf"]
        assert with_configured_chat([str(served)], str(served)) == [str(served)]

    def test_an_unconfigured_role_adds_nothing(self, served) -> None:
        assert with_configured_chat(["a/b/c.gguf"], "") == ["a/b/c.gguf"]
