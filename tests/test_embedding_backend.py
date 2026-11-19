"""Tests for lilbee's per-provider xberg embedding backend (semantic chunk boundaries)."""

from __future__ import annotations

import gc
import threading
import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from tests.conftest import make_mock_services

_DIM = 8
_TOPICS = [
    "Solar panels convert sunlight into electricity via photovoltaic cells.",
    "The FDA approved a new clinical trial for a diabetes treatment.",
    "Quantum computers use qubits and entanglement for parallel computation.",
    "Ancient Roman aqueducts used gravity to transport water across cities.",
]
_TEXT = "\n\n".join(_TOPICS[i % len(_TOPICS)] + f" Variant {i}." for i in range(60))


def _vector(text: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    return rng.random(_DIM).astype(np.float32)


def _provider() -> MagicMock:
    """A provider whose embedder answers every text with a vector derived from the text."""
    provider = make_mock_services().provider
    provider.embed.side_effect = lambda texts: [_vector(t) for t in texts]
    return provider


@pytest.fixture(autouse=True)
def semantic_chunking(monkeypatch):
    monkeypatch.setattr(cfg, "semantic_chunking", True)
    monkeypatch.setattr(cfg, "embedding_dim", _DIM)
    yield
    svc_mod.set_services(None)


def _chunk_as(provider: MagicMock) -> list[str]:
    """Chunk the sample text semantically while *provider* is the current provider."""
    from lilbee.data.extract.chunk import chunk_text

    svc_mod.set_services(make_mock_services(provider=provider))
    return chunk_text(_TEXT)


class TestEachProviderEmbedsItsOwnBoundaries:
    def test_two_providers_bound_in_turn_each_chunk_with_their_own_embedder(self):
        from lilbee.data.extract.backends import sync_xberg_backends

        first, second = _provider(), _provider()
        sync_xberg_backends(first)
        sync_xberg_backends(second)

        for provider, other in ((first, second), (second, first), (first, second)):
            before = provider.embed.call_count
            other_before = other.embed.call_count
            assert _chunk_as(provider)
            assert provider.embed.call_count > before
            assert other.embed.call_count == other_before

    def test_a_services_build_registers_no_embedding_backend(self, monkeypatch):
        import xberg

        from lilbee.data.extract.backends import sync_xberg_backends

        register = MagicMock()
        monkeypatch.setattr(xberg, "register_embedding_backend", register)
        sync_xberg_backends(_provider())
        register.assert_not_called()


class TestARegisteringProviderLeavesNoWindow:
    def test_chunking_during_another_providers_registration_returns_its_chunks(self, monkeypatch):
        import xberg

        from lilbee.data.extract.backends import sync_xberg_backends
        from lilbee.data.extract.chunk import chunk_text

        first, second = _provider(), _provider()
        sync_xberg_backends(first)
        reference = _chunk_as(first)
        assert reference

        entered, release = threading.Event(), threading.Event()
        binder: list[threading.Thread] = []
        real_register = xberg.register_embedding_backend

        def held_register(backend):
            if threading.current_thread() in binder:
                entered.set()
                assert release.wait(60)
            real_register(backend)

        monkeypatch.setattr(xberg, "register_embedding_backend", held_register)

        def bind_second() -> None:
            sync_xberg_backends(second)
            chunk_text(_TEXT)

        svc_mod.set_services(make_mock_services(provider=second))
        thread = threading.Thread(target=bind_second)
        binder.append(thread)
        thread.start()
        try:
            assert entered.wait(60)
            during = _chunk_as(first)
        finally:
            release.set()
            thread.join(60)
        assert during == reference
        assert not thread.is_alive()


class TestProviderEmbedders:
    def test_a_provider_registers_one_backend_however_often_it_is_named(self, monkeypatch):
        import xberg

        from lilbee.data.extract.backends.embedding import _ProviderEmbedders

        register = MagicMock()
        monkeypatch.setattr(xberg, "register_embedding_backend", register)
        embedders = _ProviderEmbedders()
        provider = MagicMock()
        names = {embedders.name_for(provider) for _ in range(5)}
        assert len(names) == 1
        register.assert_called_once()
        assert register.call_args.args[0].name() == names.pop()

    def test_two_providers_get_two_names_and_two_backends(self, monkeypatch):
        import xberg

        from lilbee.data.extract.backends.embedding import _ProviderEmbedders

        register = MagicMock()
        monkeypatch.setattr(xberg, "register_embedding_backend", register)
        embedders = _ProviderEmbedders()
        first, second = MagicMock(), MagicMock()
        assert embedders.name_for(first) != embedders.name_for(second)
        assert register.call_count == 2
        assert embedders.name_for(first) == register.call_args_list[0].args[0].name()

    def test_the_backend_routes_to_its_providers_embed_and_reports_the_dimension(
        self, monkeypatch
    ):
        import xberg

        from lilbee.data.extract.backends.embedding import _ProviderEmbedders

        register = MagicMock()
        monkeypatch.setattr(xberg, "register_embedding_backend", register)
        monkeypatch.setattr(cfg, "embedding_dim", 7)
        provider = MagicMock()
        vectors = [np.zeros(7, dtype=np.float32)]
        provider.embed.return_value = vectors
        _ProviderEmbedders().name_for(provider)
        backend = register.call_args.args[0]
        assert backend.dimensions() == 7
        assert backend.embed(["hello"]) is vectors
        provider.embed.assert_called_once_with(["hello"])
        assert backend.initialize() is None
        assert backend.shutdown() is None

    def test_a_backend_does_not_keep_its_provider_alive_and_fails_once_it_is_gone(
        self, monkeypatch
    ):
        import xberg

        from lilbee.data.extract.backends.embedding import _ProviderEmbedders

        register = MagicMock()
        monkeypatch.setattr(xberg, "register_embedding_backend", register)
        embedders = _ProviderEmbedders()
        provider = MagicMock()
        embedders.name_for(provider)
        backend = register.call_args.args[0]
        del provider
        gc.collect()
        with pytest.raises(ReferenceError, match="no longer exists"):
            backend.embed(["hello"])
