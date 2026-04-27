"""Tests for :class:`NerConceptsExtractor` (entities-only output).

The extractor emits only ``EntityKind.ENTITY`` records. Concept pages
are proposed by the LLM inside the per-source batched call downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity
from lilbee.wiki.entity_extractor.ner_concepts import (
    NerConceptsExtractor,
    pre_clean_for_ner,
)


@dataclass
class _FakeSpan:
    """Stand-in for a spaCy ``Span`` item."""

    text: str
    label_: str = ""


@dataclass
class _FakeDoc:
    """Stand-in for a spaCy ``Doc``. ``noun_chunks`` is accepted and
    ignored: the extractor does not read it, so older fixtures and
    integration tests that still populate it keep compiling."""

    ents: list[_FakeSpan] = field(default_factory=list)
    noun_chunks: list[_FakeSpan] = field(default_factory=list)


class _FakePipeline:
    """Mimics ``spacy.language.Language.pipe``: returns one doc per text."""

    def __init__(self, doc_map: dict[str, _FakeDoc]) -> None:
        self._doc_map = doc_map

    def pipe(self, texts: Any) -> Any:
        for text in texts:
            yield self._doc_map.get(text, _FakeDoc())


def _chunk(source: str, idx: int, text: str) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text/plain",
        page_start=1,
        page_end=1,
        line_start=1,
        line_end=1,
        chunk=text,
        chunk_index=idx,
        vector=[0.0] * 3,
    )


@pytest.fixture(autouse=True)
def _lower_min_mentions() -> Any:
    prev = cfg.wiki_entity_min_mentions
    cfg.wiki_entity_min_mentions = 1
    yield
    cfg.wiki_entity_min_mentions = prev


def _patch_pipeline(doc_map: dict[str, _FakeDoc]) -> Any:
    return patch(
        "lilbee.wiki.entity_extractor.ner_concepts._load_spacy",
        return_value=_FakePipeline(doc_map),
    )


class TestEmptyAndMissingCases:
    def test_empty_chunks_returns_empty(self) -> None:
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        assert extractor.extract([]) == []

    def test_spacy_unavailable_returns_empty(self) -> None:
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with patch(
            "lilbee.wiki.entity_extractor.ner_concepts._load_spacy",
            return_value=None,
        ):
            assert extractor.extract([_chunk("a.txt", 0, "anything")]) == []


class TestLoadSpacyFallbacks:
    def test_concepts_module_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        from lilbee.wiki.entity_extractor import ner_concepts

        real_import = builtins.__import__

        def boom(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lilbee.retrieval.concepts":
                raise ImportError("concepts module missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert ner_concepts._load_spacy() is None

    def test_spacy_model_import_error(self) -> None:
        from lilbee.wiki.entity_extractor import ner_concepts

        with patch(
            "lilbee.retrieval.concepts.load_spacy_pipeline",
            side_effect=ImportError("model missing"),
        ):
            assert ner_concepts._load_spacy() is None


class TestNerExtraction:
    def test_extracts_allowed_ner_labels_as_entities(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Henry Ford", "PERSON"),
                _FakeSpan("Ford Motor Company", "ORG"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"Henry Ford founded Ford Motor Company": doc}):
            result = extractor.extract(
                [_chunk("hist.txt", 0, "Henry Ford founded Ford Motor Company")]
            )
        kinds = {e.label: e.kind for e in result}
        assert kinds == {"Henry Ford": EntityKind.ENTITY, "Ford Motor Company": EntityKind.ENTITY}

    def test_disallowed_ner_labels_filtered_out(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("2024", "DATE"),
                _FakeSpan("five", "CARDINAL"),
                _FakeSpan("Berlin", "GPE"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "text")])
        labels = {e.label for e in result}
        assert labels == {"Berlin"}

    def test_all_default_allowed_labels_accepted(self) -> None:
        allowed = sorted(cfg.concept_allowed_ent_types)
        doc = _FakeDoc(ents=[_FakeSpan(f"Thing{i}", lbl) for i, lbl in enumerate(allowed)])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "text")])
        assert {e.type_hint for e in result} == set(allowed)


class TestOnlyEntityKind:
    """The extractor emits ENTITY records exclusively."""

    def test_extractor_never_emits_concept_kind(self) -> None:
        """Even with noun_chunks populated, no CONCEPT records appear."""
        doc = _FakeDoc(
            ents=[_FakeSpan("Henry Ford", "PERSON")],
            noun_chunks=[_FakeSpan("tire pressure"), _FakeSpan("engine coolant")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        assert all(e.kind is EntityKind.ENTITY for e in result)
        assert {e.label for e in result} == {"Henry Ford"}


class TestDedupAndAggregation:
    def test_duplicate_ner_surface_folds_into_one(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Henry Ford", "PERSON"),
                _FakeSpan("Henry Ford", "PERSON"),
            ],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t0": doc, "t1": doc}):
            result = extractor.extract([_chunk("a.txt", 0, "t0"), _chunk("a.txt", 1, "t1")])
        assert len(result) == 1
        rec = result[0]
        assert len(rec.chunk_refs) == 2

    def test_output_sorted_by_slug(self) -> None:
        doc = _FakeDoc(
            ents=[_FakeSpan("Zulu", "PERSON"), _FakeSpan("Alpha", "PERSON")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("a.txt", 0, "t")])
        assert [e.slug for e in result] == ["alpha", "zulu"]


class TestMinMentionsThreshold:
    def test_below_threshold_filtered(self) -> None:
        cfg.wiki_entity_min_mentions = 2
        doc_once = _FakeDoc(ents=[_FakeSpan("One-off Person", "PERSON")])
        doc_twice = _FakeDoc(ents=[_FakeSpan("Repeated Person", "PERSON")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"a": doc_once, "b": doc_twice, "c": doc_twice}):
            result = extractor.extract(
                [
                    _chunk("s.txt", 0, "a"),
                    _chunk("s.txt", 1, "b"),
                    _chunk("s.txt", 2, "c"),
                ]
            )
        labels = {e.label for e in result}
        assert labels == {"Repeated Person"}


class TestReturnRecordShape:
    def test_records_are_extractedentity_instances_with_slugs(self) -> None:
        doc = _FakeDoc(ents=[_FakeSpan("Ford Motor Company", "ORG")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        assert len(result) == 1
        rec = result[0]
        assert isinstance(rec, ExtractedEntity)
        assert rec.slug == "ford-motor-company"

    def test_unicode_label_passes_sanity_but_slugs_empty(self) -> None:
        doc = _FakeDoc(ents=[_FakeSpan("αβγ", "PERSON"), _FakeSpan("Ford", "ORG")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"Ford"}


class TestLabelSanityRejection:
    def test_rejects_pipe_delimited_ner_entity(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("| | Designer", "PERSON"),
                _FakeSpan("Chevrolet Caprice", "PRODUCT"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"Chevrolet Caprice"}

    def test_logs_rejection_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(ents=[_FakeSpan("| pipe-entity", "PERSON")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        messages = [r.message for r in caplog.records]
        assert any("rejected entity" in m for m in messages)

    def test_short_ner_surface_is_filtered(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("X", "PERSON"),
                _FakeSpan("Berlin", "GPE"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "text")])
        assert {e.label for e in result} == {"Berlin"}


class TestPreCleanForNer:
    """Pure-function tests for the A2 markdown-noise stripper."""

    def test_strips_markdown_table_row(self) -> None:
        noisy = "intro\n| Designer | Irv Rybicki |\noutro"
        cleaned = pre_clean_for_ner(noisy)
        assert "| Designer" not in cleaned
        assert "intro" in cleaned
        assert "outro" in cleaned

    def test_strips_page_number_only_line(self) -> None:
        noisy = "paragraph text\n  42  \nmore text"
        cleaned = pre_clean_for_ner(noisy)
        assert " 42 " not in cleaned
        assert "paragraph text" in cleaned
        assert "more text" in cleaned

    def test_preserves_inline_numbers(self) -> None:
        noisy = "Chapter 42 introduces the concept."
        assert pre_clean_for_ner(noisy) == noisy

    def test_strips_nav_chrome(self) -> None:
        noisy = "lede\nEdit this page\nJump to search\nbody"
        cleaned = pre_clean_for_ner(noisy)
        assert "Edit this page" not in cleaned
        assert "Jump to search" not in cleaned

    def test_normal_prose_round_trips_unchanged(self) -> None:
        prose = "The Chevrolet Caprice is a full-size car.\nIt was produced from 1965."
        assert pre_clean_for_ner(prose) == prose

    def test_empty_input(self) -> None:
        assert pre_clean_for_ner("") == ""


class TestExtractorAppliesPreClean:
    def test_table_row_never_reaches_pipeline(self) -> None:
        seen_texts: list[str] = []

        class _CapturingPipeline:
            def pipe(self, texts: Any) -> Any:
                for text in texts:
                    seen_texts.append(text)
                    yield _FakeDoc()

        extractor = NerConceptsExtractor(MagicMock(), cfg)
        noisy = "Chevrolet Caprice.\n| Designer | Irv Rybicki |\n"
        with patch(
            "lilbee.wiki.entity_extractor.ner_concepts._load_spacy",
            return_value=_CapturingPipeline(),
        ):
            extractor.extract([_chunk("s.txt", 0, noisy)])
        assert seen_texts, "extractor should have called the pipeline"
        assert all("| Designer" not in t for t in seen_texts)


class TestEntityTypeFilter:
    def test_quantity_and_date_entities_are_dropped(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("42 miles", "QUANTITY"),
                _FakeSpan("1965", "DATE"),
                _FakeSpan("Chevrolet", "ORG"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"Chevrolet"}

    def test_config_override_narrows_allowed_types(self) -> None:
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Chevrolet", "ORG"),
                _FakeSpan("Irv Rybicki", "PERSON"),
                _FakeSpan("Americans", "NORP"),
            ]
        )
        prev = cfg.concept_allowed_ent_types
        cfg.concept_allowed_ent_types = frozenset({"PERSON"})
        try:
            extractor = NerConceptsExtractor(MagicMock(), cfg)
            with _patch_pipeline({"t": doc}):
                result = extractor.extract([_chunk("s.txt", 0, "t")])
        finally:
            cfg.concept_allowed_ent_types = prev
        labels = {e.label for e in result}
        assert labels == {"Irv Rybicki"}


class TestFunnelLogging:
    """Concept counters are absent; entity counters remain."""

    def test_funnel_logged_once_at_debug_entities_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("DEBUG", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Chevrolet", "ORG"),
                _FakeSpan("42", "QUANTITY"),
                _FakeSpan("| bad", "PERSON"),
            ],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        funnel_messages = [r.message for r in caplog.records if "ner funnel" in r.message]
        assert len(funnel_messages) == 1
        message = funnel_messages[0]
        assert "raw_ents=3" in message
        assert "type_filter_dropped=1" in message
        assert "label_sanity_dropped_entities=1" in message
        assert "kept_entity_surfaces=1" in message
        # The extractor does not emit concept counters.
        assert "raw_noun_chunks" not in message
        assert "label_sanity_dropped_concepts" not in message
        assert "kept_concept_surfaces" not in message

    def test_funnel_not_emitted_when_debug_off(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(ents=[_FakeSpan("Chevrolet", "ORG")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        assert not any("ner funnel" in r.message for r in caplog.records)
