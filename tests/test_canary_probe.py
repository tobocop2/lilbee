"""The canary probe: which embedder a store names, and how it embeds canaries."""

from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store.canary import CANARY_TEXTS
from lilbee.retrieval import canary_probe as probe_module
from lilbee.retrieval.canary_probe import canary_probe, embedder_trigger


@pytest.fixture()
def registry():
    return mock.MagicMock()


def _config(model):
    return cfg.model_copy(update={"embedding_model": model})


class TestEmbedderTrigger:
    def test_no_model_names_no_embedder(self, registry):
        assert embedder_trigger(_config(""), registry) is None

    def test_a_remote_model_is_named_by_the_process(self, registry):
        trigger = embedder_trigger(_config("ollama/nomic-embed-text"), registry)

        assert trigger is not None
        assert trigger.startswith("process:")
        assert trigger.endswith(":ollama/nomic-embed-text")
        assert trigger == embedder_trigger(_config("ollama/nomic-embed-text"), registry)

    def test_a_local_model_is_named_by_the_engine_build_and_the_file(self, registry, tmp_path):
        model = tmp_path / "embed.gguf"
        model.write_bytes(b"12345")
        with (
            mock.patch.object(probe_module, "resolve_model_path", return_value=model),
            mock.patch.object(probe_module, "engine_build_id", return_value="build-1"),
        ):
            first = embedder_trigger(_config(str(model)), registry)
            model.write_bytes(b"123456")
            second = embedder_trigger(_config(str(model)), registry)

        assert first is not None
        assert first.startswith(f"build-1|{model}@5-")
        assert second != first

    def test_another_engine_build_is_another_trigger(self, registry, tmp_path):
        model = tmp_path / "embed.gguf"
        model.write_bytes(b"1")
        with mock.patch.object(probe_module, "resolve_model_path", return_value=model):
            with mock.patch.object(probe_module, "engine_build_id", return_value="build-1"):
                first = embedder_trigger(_config(str(model)), registry)
            with mock.patch.object(probe_module, "engine_build_id", return_value="build-2"):
                second = embedder_trigger(_config(str(model)), registry)

        assert first != second

    def test_a_model_file_that_is_gone_names_no_embedder(self, registry, tmp_path):
        with mock.patch.object(
            probe_module, "resolve_model_path", return_value=tmp_path / "gone.gguf"
        ):
            assert embedder_trigger(_config(str(tmp_path / "gone.gguf")), registry) is None

    def test_a_model_that_does_not_resolve_names_no_embedder(self, registry):
        from lilbee.providers.base import ProviderError

        with mock.patch.object(probe_module, "resolve_model_path", side_effect=ProviderError("x")):
            assert embedder_trigger(_config("org/repo-GGUF/m.gguf"), registry) is None


class TestCanaryProbe:
    def test_it_embeds_the_canaries_with_the_embedder_of_the_store(self, registry):
        embedder = mock.MagicMock()
        embedder.embed_batch.return_value = [[1.0]] * len(CANARY_TEXTS)

        probe = canary_probe(_config("ollama/x"), embedder, registry)

        assert probe.embed(list(CANARY_TEXTS)) == [[1.0]] * len(CANARY_TEXTS)
        embedder.embed_batch.assert_called_once_with(list(CANARY_TEXTS))

    def test_its_trigger_follows_the_config(self, registry):
        config = _config("ollama/x")
        probe = canary_probe(config, mock.MagicMock(), registry)
        first = probe.trigger()

        config.embedding_model = "ollama/y"

        second = probe.trigger()
        assert first is not None and "ollama/x" in first
        assert second is not None and "ollama/y" in second
