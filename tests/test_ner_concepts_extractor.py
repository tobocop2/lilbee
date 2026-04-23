"""Tests for :class:`NerConceptsExtractor`.

Focus: spaCy NER + noun-phrase aggregation, case-insensitive dedup of
concept labels against NER surface forms, NER-label filtering, the
``wiki_entity_min_mentions`` threshold, and graceful fallback when
spaCy is unavailable. The spaCy pipeline is replaced with a lightweight
fake so tests stay fast and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.store import SearchChunk
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity
from lilbee.wiki.entity_extractor.ner_concepts import (
    NerConceptsExtractor,
    pre_clean_for_ner,
)


@dataclass
class _FakeSpan:
    """Stand-in for a spaCy ``Span`` (``Doc.ents`` + ``doc.noun_chunks`` items)."""

    text: str
    label_: str = ""


@dataclass
class _FakeDoc:
    """Stand-in for a spaCy ``Doc``."""

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
    """Build a minimal SearchChunk fixture for the extractor."""
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
    """Fixtures are tiny, so count a single mention as one entity or concept."""
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
    """_load_spacy swallows both import failure modes and returns None."""

    def test_concepts_module_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        from lilbee.wiki.entity_extractor import ner_concepts

        real_import = builtins.__import__

        def boom(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lilbee.concepts":
                raise ImportError("concepts module missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert ner_concepts._load_spacy() is None

    def test_spacy_model_import_error(self) -> None:
        from lilbee.wiki.entity_extractor import ner_concepts

        with patch(
            "lilbee.concepts.load_spacy_pipeline",
            side_effect=ImportError("model missing"),
        ):
            assert ner_concepts._load_spacy() is None

    def test_short_ner_surface_is_filtered(self) -> None:
        """NER entities with < 2 chars after strip never become ExtractedEntity records."""
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
        type_hints = {e.label: e.type_hint for e in result}
        assert type_hints == {"Henry Ford": "PERSON", "Ford Motor Company": "ORG"}

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
        """All nine labels in ``DEFAULT_ALLOWED_NER_LABELS`` round-trip to
        ``ExtractedEntity`` records. Uses the live default set from config
        so the test doesn't drift when types are added or removed there.
        """
        allowed = sorted(cfg.concept_allowed_ent_types)
        doc = _FakeDoc(ents=[_FakeSpan(f"Thing{i}", lbl) for i, lbl in enumerate(allowed)])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "text")])
        assert {e.type_hint for e in result} == set(allowed)


class TestConceptExtraction:
    def test_noun_chunks_become_concept_records(self) -> None:
        doc = _FakeDoc(noun_chunks=[_FakeSpan("tire pressure"), _FakeSpan("engine coolant")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("m.txt", 0, "text")])
        assert {e.kind for e in result} == {EntityKind.CONCEPT}
        assert {e.label for e in result} == {"tire pressure", "engine coolant"}
        assert all(e.type_hint == "noun_phrase" for e in result)

    def test_noun_chunks_too_short_dropped(self) -> None:
        doc = _FakeDoc(noun_chunks=[_FakeSpan("a"), _FakeSpan("tires")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"text": doc}):
            result = extractor.extract([_chunk("m.txt", 0, "text")])
        assert [e.label for e in result] == ["tires"]


class TestDedupAndAggregation:
    def test_concept_matching_entity_folds_into_entity(self) -> None:
        doc_0 = _FakeDoc(
            ents=[_FakeSpan("Ford Motor Company", "ORG")],
            noun_chunks=[_FakeSpan("ford motor company")],
        )
        doc_1 = _FakeDoc(noun_chunks=[_FakeSpan("Ford motor company")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"txt0": doc_0, "txt1": doc_1}):
            result = extractor.extract([_chunk("a.txt", 0, "txt0"), _chunk("b.txt", 1, "txt1")])
        assert len(result) == 1
        merged = result[0]
        assert merged.kind is EntityKind.ENTITY
        assert merged.type_hint == "ORG"
        assert {(r.source, r.chunk_index) for r in merged.chunk_refs} == {
            ("a.txt", 0),
            ("b.txt", 1),
        }

    def test_chunk_refs_aggregate_across_chunks_and_dedupe_same_chunk(self) -> None:
        doc = _FakeDoc(
            ents=[_FakeSpan("Henry Ford", "PERSON"), _FakeSpan("Henry Ford", "PERSON")],
            noun_chunks=[_FakeSpan("henry ford")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t0": doc, "t1": doc}):
            result = extractor.extract([_chunk("a.txt", 0, "t0"), _chunk("a.txt", 1, "t1")])
        assert len(result) == 1
        rec = result[0]
        assert rec.kind is EntityKind.ENTITY
        assert len(rec.chunk_refs) == 2
        assert {(r.source, r.chunk_index) for r in rec.chunk_refs} == {
            ("a.txt", 0),
            ("a.txt", 1),
        }

    def test_output_sorted_entities_before_concepts_then_by_slug(self) -> None:
        doc = _FakeDoc(
            ents=[_FakeSpan("Zulu", "PERSON"), _FakeSpan("Alpha", "PERSON")],
            noun_chunks=[_FakeSpan("banana pudding"), _FakeSpan("apple sauce")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("a.txt", 0, "t")])
        kinds_then_slugs = [(e.kind.value, e.slug) for e in result]
        assert kinds_then_slugs == [
            ("concept", "apple-sauce"),
            ("concept", "banana-pudding"),
            ("entity", "alpha"),
            ("entity", "zulu"),
        ]


class TestMinMentionsThreshold:
    def test_below_threshold_filtered(self) -> None:
        cfg.wiki_entity_min_mentions = 2
        doc_once = _FakeDoc(
            ents=[_FakeSpan("One-off Person", "PERSON")],
            noun_chunks=[_FakeSpan("fleeting thought")],
        )
        doc_twice = _FakeDoc(
            ents=[_FakeSpan("Repeated Person", "PERSON")],
            noun_chunks=[_FakeSpan("recurring topic")],
        )
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
        assert labels == {"Repeated Person", "recurring topic"}


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

    def test_empty_slug_record_is_dropped(self) -> None:
        """Structural-noise labels never become ExtractedEntity records.

        Belt-and-suspenders: ``is_valid_label`` rejects ``>>>>`` at the
        extractor boundary (structural char), and ``make_slug`` would
        also strip it to an empty slug downstream. Either path drops
        the record before the generator writes ``/.md``.
        """
        doc = _FakeDoc(noun_chunks=[_FakeSpan(">>>>"), _FakeSpan("tires")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"tires"}


class TestLabelSanityRejection:
    """QA-driven (bb-8b7s) regressions on the noise patterns observed
    in Wikipedia table markup and PDF chrome."""

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

    def test_rejects_page_number_prefixed_noun_chunk(self) -> None:
        doc = _FakeDoc(
            noun_chunks=[
                _FakeSpan("158 vehicle"),
                _FakeSpan("brake pads"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"brake pads"}

    def test_rejects_short_fragment(self) -> None:
        doc = _FakeDoc(noun_chunks=[_FakeSpan("ab"), _FakeSpan("tires")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"tires"}

    def test_logs_rejection_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(
            ents=[_FakeSpan("| pipe-entity", "PERSON")],
            noun_chunks=[_FakeSpan("| bad |"), _FakeSpan("tires")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        messages = [r.message for r in caplog.records]
        assert any("rejected entity" in m for m in messages)
        assert any("rejected noun-chunk" in m for m in messages)

    def test_unicode_label_passes_sanity_but_slugs_empty(self) -> None:
        """Unicode letters satisfy ``is_valid_label`` (alnum + length) but
        slug-clean to empty since ``make_slug`` restricts to ``[a-z0-9-]``.
        The defensive ``if not slug`` check in ``_make_record`` drops the
        aggregate rather than writing a file named ``.md``.
        """
        doc = _FakeDoc(ents=[_FakeSpan("αβγ", "PERSON"), _FakeSpan("Ford", "ORG")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"Ford"}


class TestPreCleanForNer:
    """Pure-function unit tests for the A2 markdown-noise stripper."""

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
        # "42" inside prose must NOT be stripped; only standalone
        # page-number-only lines get removed.
        noisy = "Chapter 42 introduces the concept."
        assert pre_clean_for_ner(noisy) == noisy

    def test_strips_nav_chrome(self) -> None:
        noisy = "lede\nEdit this page\nJump to search\nbody"
        cleaned = pre_clean_for_ner(noisy)
        assert "Edit this page" not in cleaned
        assert "Jump to search" not in cleaned
        assert "lede" in cleaned
        assert "body" in cleaned

    def test_normal_prose_round_trips_unchanged(self) -> None:
        prose = "The Chevrolet Caprice is a full-size car.\nIt was produced from 1965."
        assert pre_clean_for_ner(prose) == prose

    def test_mixed_noise_and_prose(self) -> None:
        noisy = (
            "The Caprice was designed by\n"
            "| Designer | Irv Rybicki |\n"
            "| --- | --- |\n"
            "Irv Rybicki, a GM designer."
        )
        cleaned = pre_clean_for_ner(noisy)
        assert "|" not in cleaned
        assert "The Caprice was designed by" in cleaned
        assert "Irv Rybicki, a GM designer." in cleaned

    def test_empty_input(self) -> None:
        assert pre_clean_for_ner("") == ""


class TestExtractorAppliesPreClean:
    """Integration: the extractor hands pre-cleaned text to spaCy."""

    def test_table_row_never_reaches_pipeline(self) -> None:
        """Captured pipeline input must not contain the table markup."""
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
    """A3: spaCy NER labels outside ``concept_allowed_ent_types`` are dropped."""

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

    def test_fac_and_norp_are_included_by_default(self) -> None:
        """FAC (facilities) and NORP (nationalities/political/religious groups)
        were missing from the pre-A3 hardcoded set but are useful wiki topics.
        """
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Pentagon", "FAC"),
                _FakeSpan("Americans", "NORP"),
            ]
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            result = extractor.extract([_chunk("s.txt", 0, "t")])
        labels = {e.label for e in result}
        assert labels == {"Pentagon", "Americans"}

    def test_config_override_narrows_allowed_types(self) -> None:
        """Setting a narrower override via config keeps only those types."""
        from dataclasses import replace  # noqa: F401  (pattern documentation)

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
    """A3: per-extraction funnel counters emitted at DEBUG once per pass."""

    def test_funnel_logged_once_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(
            ents=[
                _FakeSpan("Chevrolet", "ORG"),
                _FakeSpan("42", "QUANTITY"),
                _FakeSpan("| bad", "PERSON"),
            ],
            noun_chunks=[_FakeSpan("the car"), _FakeSpan("ab")],
        )
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        funnel_messages = [r.message for r in caplog.records if "ner funnel" in r.message]
        assert len(funnel_messages) == 1, "funnel should log exactly once per extraction"
        message = funnel_messages[0]
        assert "raw_ents=3" in message
        assert "raw_noun_chunks=2" in message
        assert "type_filter_dropped=1" in message  # the QUANTITY
        assert "label_sanity_dropped=2" in message  # "| bad" and "ab"
        assert "kept_entity_surfaces=1" in message  # Chevrolet
        assert "kept_concept_surfaces=1" in message  # the car

    def test_funnel_not_emitted_when_debug_off(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO", logger="lilbee.wiki.entity_extractor.ner_concepts")
        doc = _FakeDoc(ents=[_FakeSpan("Chevrolet", "ORG")])
        extractor = NerConceptsExtractor(MagicMock(), cfg)
        with _patch_pipeline({"t": doc}):
            extractor.extract([_chunk("s.txt", 0, "t")])
        assert not any("ner funnel" in r.message for r in caplog.records)
