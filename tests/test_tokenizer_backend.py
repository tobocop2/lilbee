"""Tests for lilbee's xberg tokenizer backend (token-budgeted chunk sizing)."""

from __future__ import annotations

import threading

import pytest
from xberg import ChunkingConfig, ChunkSizing, ExtractedDocument, ExtractionConfig

from lilbee.data.extract.backends import tokenizer as tokenizer_module
from lilbee.data.extract.backends.tokenizer import (
    _XBERG_PROBE_TEXT,
    LilbeeTokenizerBackend,
    TokenCountError,
    _estimate_tokens,
    _ProviderTokenizers,
    provider_tokenizers,
)
from lilbee.data.extract.chunk import build_chunking_config, chunk_text
from lilbee.data.extract.xberg import BatchItem, aextract_batch, extract_document
from lilbee.data.types import TokenizerBackendName
from lilbee.providers.base import ProviderError, ProviderErrorKind
from tests.conftest import make_mock_services

_NOTE = b"A note about turbines and pumps. " * 20
_HOLD_S = 5.0


def test_name_is_lilbee():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.name() == TokenizerBackendName.LILBEE


def test_initialize_and_shutdown_are_noops():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.initialize() is None
    assert backend.shutdown() is None


def test_a_probe_answer_taken_by_another_thread_still_returns_a_count():
    """The answer goes to one reader; a reader that finds it gone counts, and never gets None."""

    class _TakenAfterFirstRead(LilbeeTokenizerBackend):
        reads = 0

        @property
        def _probe_count(self) -> int | None:
            self.reads += 1
            return 5 if self.reads == 1 else None

        @_probe_count.setter
        def _probe_count(self, value: int | None) -> None: ...

    backend = _TakenAfterFirstRead(count_fn=lambda _t: 9)
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 5
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 9


def test_exact_count_is_returned():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 7)
    assert backend.count_tokens("hello world") == 7


def test_probe_raises_a_count_failure():
    def count_fn(_text: str) -> int:
        raise RuntimeError("engine stopped")

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    with pytest.raises(RuntimeError, match="engine stopped"):
        backend.probe()


def test_probe_answers_only_the_probe_text_without_the_provider():
    counts = iter([3, 9])
    calls: list[str] = []

    def count_fn(text: str) -> int:
        calls.append(text)
        return next(counts)

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    backend.probe()
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 3
    assert backend.count_tokens("hello world") == 9
    assert calls == [_XBERG_PROBE_TEXT, "hello world"]


def test_probe_answer_is_used_once_then_the_provider_counts_again():
    counts = iter([3, 7])
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: next(counts))
    backend.probe()
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 3
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 7


def test_probe_text_routes_to_the_provider_before_a_probe():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 4)
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 4


def test_empty_text_is_zero_without_calling_count_fn():
    calls: list[str] = []

    def count_fn(text: str) -> int:
        calls.append(text)
        return 5

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    assert backend.count_tokens("") == 0
    assert calls == []


def test_not_implemented_falls_back_to_char_estimate():
    """SDK embedders expose no local tokenizer, so the count degrades to a guess."""

    def boom(_text: str) -> int:
        raise NotImplementedError("SDK backends have no local tokenizer")

    backend = LilbeeTokenizerBackend(count_fn=boom)
    text = "x" * 9
    assert backend.count_tokens(text) == _estimate_tokens(text)


def test_unexpected_error_raises_instead_of_estimating():
    """A count that fails after the embed-grade wait raises to the caller."""

    def boom(_text: str) -> int:
        raise RuntimeError("embedder unreachable")

    backend = LilbeeTokenizerBackend(count_fn=boom)
    with pytest.raises(RuntimeError, match="embedder unreachable"):
        backend.count_tokens("x" * 9)


def test_zero_count_for_non_empty_text_falls_back_to_estimate():
    """xberg rejects a zero count for non-empty text; degrade to the estimate."""
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 0)
    text = "x" * 9
    assert backend.count_tokens(text) == _estimate_tokens(text)


class _Provider:
    """A provider with no embed cap that counts one token per word."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_token_cap(self) -> int | None:
        return None

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        return max(1, len(text.split()))


class _WordCounter(_Provider):
    """A provider whose counts all succeed."""


class _StopsAfterTheProbe(_Provider):
    """A provider whose count route fails after its first count."""

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        if self.calls > 1:
            raise ProviderError("The embedding server stopped")
        return 1


@pytest.fixture
def services_with():
    """Set services around one provider for a test, and clear them after."""
    from lilbee.app.services import set_services

    def _set(provider: object) -> None:
        set_services(make_mock_services(provider=provider))

    yield _set
    set_services(None)


class TestProviderTokenizers:
    def test_a_new_registry_numbers_its_names_and_counts_failures_from_zero(self, monkeypatch):
        registered: list[LilbeeTokenizerBackend] = []
        monkeypatch.setattr(tokenizer_module, "_register", registered.append)
        tokenizers = _ProviderTokenizers()
        first, second = _StopsAfterTheProbe(), _WordCounter()
        first_name, second_name = (
            f"{TokenizerBackendName.LILBEE}-1",
            f"{TokenizerBackendName.LILBEE}-2",
        )
        assert tokenizers.failures(first_name) is None
        assert tokenizers.name_for(first) == first_name
        assert tokenizers.name_for(second) == second_name
        assert [b.name() for b in registered] == [first_name, second_name]
        assert tokenizers.failures(first_name) == 0
        with pytest.raises(ProviderError, match="The embedding server stopped"):
            registered[0].count_tokens("two words")
        assert tokenizers.failures(first_name) == 1
        assert tokenizers.failures(second_name) == 0

    def test_each_provider_registers_one_name_of_its_own(self):
        import xberg

        first, second = _WordCounter(), _WordCounter()
        name = provider_tokenizers.name_for(first)
        assert provider_tokenizers.name_for(first) == name
        other = provider_tokenizers.name_for(second)
        assert other != name
        assert {name, other} <= set(xberg.list_tokenizer_backends())
        assert first.calls == 1  # the probe, once

    def test_a_registered_backend_counts_through_its_provider(self):
        import xberg

        provider = _WordCounter()
        name = provider_tokenizers.name_for(provider)
        chunking = ChunkingConfig(
            max_characters=4, overlap=0, sizing=ChunkSizing(type="tokenizer", model=name)
        )
        words = [f"word{i}" for i in range(40)]
        doc = extract_document(
            " ".join(words).encode(), "text/plain", config=ExtractionConfig(chunking=chunking)
        )
        pieces = [c.content.split() for c in doc.chunks]
        assert len(pieces) > 1
        assert max(len(p) for p in pieces) <= 4
        assert [w for p in pieces for w in p] == words
        assert provider.calls > 1
        assert name in xberg.list_tokenizer_backends()

    def test_a_probe_failure_raises_the_real_count_error(self):
        class _NoModel(_Provider):
            def count_tokens(self, text: str) -> int:
                raise ProviderError("No embedding model is configured")

        with pytest.raises(ProviderError, match="No embedding model is configured"):
            provider_tokenizers.name_for(_NoModel())

    def test_a_provider_that_stops_after_the_probe_still_registers(self):
        """xberg's own probe is answered from the first count, not from the stopped route."""
        import xberg

        provider = _StopsAfterTheProbe()
        name = provider_tokenizers.name_for(provider)
        assert name in xberg.list_tokenizer_backends()
        assert provider.calls == 1

    def test_an_sdk_provider_registers_on_its_estimate_without_a_failure(self):
        class _Sdk(_Provider):
            def count_tokens(self, text: str) -> int:
                raise NotImplementedError("no local tokenizer")

        provider = _Sdk()
        assert provider_tokenizers.failures(provider_tokenizers.name_for(provider)) == 0

    def test_a_registered_backend_does_not_keep_its_provider_alive(self):
        import gc
        import weakref

        provider = _WordCounter()
        provider_tokenizers.name_for(provider)
        ref = weakref.ref(provider)
        del provider
        gc.collect()
        assert ref() is None

    def test_a_count_after_the_provider_is_gone_fails_and_is_recorded(self):
        import gc

        keeper, provider = _WordCounter(), _WordCounter()
        name = provider_tokenizers.name_for(keeper)
        counter = provider_tokenizers._counter(provider, name)
        del provider
        gc.collect()
        with pytest.raises(ReferenceError, match="no longer exists"):
            counter("text")
        assert provider_tokenizers.failures(name) == 1

    def test_a_new_registration_never_unregisters_an_older_name(self, services_with):
        import xberg

        older = _WordCounter()
        services_with(older)
        config = ExtractionConfig(chunking=build_chunking_config())
        name = provider_tokenizers.name_for(older)
        services_with(_WordCounter())
        build_chunking_config()
        assert name in xberg.list_tokenizer_backends()
        calls = older.calls
        doc = extract_document(_NOTE, "text/plain", config=config)
        assert len(doc.chunks or []) > 0
        assert older.calls > calls

    def test_a_gone_provider_name_is_unregistered_by_the_next_registration(self, services_with):
        import gc

        import xberg

        provider = _WordCounter()
        services_with(provider)
        config = ExtractionConfig(chunking=build_chunking_config())
        name = provider_tokenizers.name_for(provider)
        del provider
        services_with(_WordCounter())
        gc.collect()
        assert name in xberg.list_tokenizer_backends()
        build_chunking_config()
        assert name not in xberg.list_tokenizer_backends()
        assert provider_tokenizers.failures(name) is None
        with pytest.raises(TokenCountError, match="failed to count tokens"):
            extract_document(_NOTE, "text/plain", config=config)

    def test_the_losing_name_of_a_first_use_race_is_unregistered(self, monkeypatch):
        import xberg

        provider = _WordCounter()
        register = tokenizer_module._register
        names: list[str] = []

        def _register_while_another_registers(backend: LilbeeTokenizerBackend) -> None:
            register(backend)
            if not names:
                names.append(backend.name())
                names.append(provider_tokenizers.name_for(provider))

        monkeypatch.setattr(tokenizer_module, "_register", _register_while_another_registers)
        assert provider_tokenizers.name_for(provider) == names[1]
        assert names[0] != names[1]
        provider_tokenizers.name_for(_WordCounter())
        assert names[0] not in xberg.list_tokenizer_backends()
        assert names[1] in xberg.list_tokenizer_backends()
        assert provider_tokenizers.failures(names[0]) is None


def test_a_second_provider_registering_its_tokenizer_leaves_a_built_config_its_chunks(
    services_with,
):
    """A config built for one provider keeps its tokenizer while another provider's first
    count is held open, as a fan-out worker with its own provider does under load."""
    first = _WordCounter()
    services_with(first)
    config = ExtractionConfig(chunking=build_chunking_config())

    probing, release = threading.Event(), threading.Event()

    class _HeldProbe(_Provider):
        def count_tokens(self, text: str) -> int:
            probing.set()
            release.wait(_HOLD_S)
            return max(1, len(text) // 4)

    def _register_another() -> None:
        services_with(_HeldProbe())
        build_chunking_config()

    other = threading.Thread(target=_register_another)
    other.start()
    try:
        assert probing.wait(_HOLD_S)
        doc = extract_document(_NOTE, "text/plain", config=config)
    finally:
        release.set()
        other.join()
    assert len(doc.chunks or []) > 0
    assert first.calls > 1


class _HeldCounter(_Provider):
    """A word counter whose first count after the probe waits until it is released."""

    def __init__(self) -> None:
        super().__init__()
        self.counting, self.release = threading.Event(), threading.Event()

    def count_tokens(self, text: str) -> int:
        if self.calls == 1:
            self.counting.set()
            self.release.wait(_HOLD_S)
        return super().count_tokens(text)


def _extract_while(services_with, during) -> ExtractedDocument | TokenCountError:
    """Extract a note with a held provider's tokenizer while *during* runs; return the result."""
    held = _HeldCounter()
    services_with(held)
    config = ExtractionConfig(chunking=build_chunking_config())
    results: list[ExtractedDocument | TokenCountError] = []

    def _extract() -> None:
        try:
            results.append(extract_document(_NOTE, "text/plain", config=config))
        except TokenCountError as error:
            results.append(error)

    worker = threading.Thread(target=_extract)
    worker.start()
    try:
        assert held.counting.wait(_HOLD_S)
        during()
    finally:
        held.release.set()
        worker.join()
    return results[0]


def test_a_failed_count_of_another_tokenizer_leaves_a_running_extraction_its_chunks(
    services_with,
):
    def _fail_another_extraction() -> None:
        services_with(_StopsAfterTheProbe())
        config = ExtractionConfig(chunking=build_chunking_config())
        with pytest.raises(TokenCountError, match="failed to count tokens"):
            extract_document(_NOTE, "text/plain", config=config)

    doc = _extract_while(services_with, _fail_another_extraction)
    assert not isinstance(doc, Exception)
    assert doc.chunks


def test_a_failed_registration_probe_leaves_a_running_extraction_its_chunks(services_with):
    class _NoModel(_Provider):
        def count_tokens(self, text: str) -> int:
            raise ProviderError("No embedding model is configured")

    def _fail_a_registration() -> None:
        with pytest.raises(ProviderError, match="No embedding model is configured"):
            provider_tokenizers.name_for(_NoModel())

    doc = _extract_while(services_with, _fail_a_registration)
    assert not isinstance(doc, Exception)
    assert doc.chunks


def test_a_config_whose_provider_is_gone_fails_the_extraction(services_with):
    import gc

    services_with(_WordCounter())
    config = ExtractionConfig(chunking=build_chunking_config())
    services_with(_WordCounter())
    gc.collect()
    with pytest.raises(TokenCountError, match="failed to count tokens"):
        extract_document(_NOTE, "text/plain", config=config)


def test_a_count_that_fails_while_chunking_raises_instead_of_cutting_small_chunks(
    services_with,
):
    services_with(_StopsAfterTheProbe())
    with pytest.raises(TokenCountError, match="failed to count tokens"):
        chunk_text(_NOTE.decode(), use_semantic=False)


def test_a_count_that_fails_in_a_batch_fails_every_extracted_input(services_with):
    import asyncio

    services_with(_StopsAfterTheProbe())
    config = ExtractionConfig(chunking=build_chunking_config())
    items = [BatchItem(data=_NOTE, mime="text/plain", filename=None, ocr=None)] * 2
    results = asyncio.run(aextract_batch(items, config))
    assert [type(r) for r in results] == [TokenCountError, TokenCountError]
    assert results[0] is not results[1]
    assert [r.kind for r in results] == [ProviderErrorKind.CONNECTION] * 2


@pytest.mark.parametrize(
    "chunking",
    [
        pytest.param(None, id="no-chunking"),
        pytest.param(ChunkingConfig(max_characters=64, overlap=0), id="default-sizing"),
        pytest.param(
            ChunkingConfig(max_characters=64, overlap=0, sizing=ChunkSizing(type="characters")),
            id="character-sizing",
        ),
    ],
)
def test_a_count_failure_elsewhere_leaves_an_extraction_not_sized_by_tokens(monkeypatch, chunking):
    """Another file's count failure during this extraction does not fail it."""
    import asyncio
    import itertools

    counts = itertools.count()
    monkeypatch.setattr(provider_tokenizers, "failures", lambda _name: next(counts))
    config = ExtractionConfig(chunking=chunking)
    items = [BatchItem(data=_NOTE, mime="text/plain", filename=None, ocr=None)]
    (doc,) = asyncio.run(aextract_batch(items, config))
    assert not isinstance(doc, Exception)
    assert extract_document(_NOTE, "text/plain", config=config).content


def test_a_batch_whose_counts_succeed_returns_its_documents(services_with):
    import asyncio

    services_with(_WordCounter())
    config = ExtractionConfig(chunking=build_chunking_config())
    items = [BatchItem(data=_NOTE, mime="text/plain", filename=None, ocr=None)]
    (doc,) = asyncio.run(aextract_batch(items, config))
    assert not isinstance(doc, Exception)
    assert doc.chunks


def test_estimate_is_conservative_and_at_least_one():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("abc") == 1
    assert _estimate_tokens("a" * 9) == 3
