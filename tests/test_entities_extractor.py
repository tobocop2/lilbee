"""Tests for entity schema induction and two-phase extraction."""

import json
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.retrieval.entities import (
    EntitySchema,
    EntityType,
    ExtractorKind,
    extract_entities,
    induce_schema,
    load_schema,
    normalize_value,
    save_schema,
)
from lilbee.retrieval.language import noun_variants


def _text_result(text):
    return SimpleNamespace(text=text)


def _chunk(text, source="a.txt", idx=0, page=1):
    return {"chunk": text, "source": source, "chunk_index": idx, "page_start": page}


PART = EntityType(
    name="part_number",
    kind=ExtractorKind.REGEX,
    pattern=r"PX\d{4}",
    synonyms=["part number"],
)
PERSON = EntityType(name="person", kind=ExtractorKind.SPACY, pattern="PERSON")
VESSEL = EntityType(name="vessel", kind=ExtractorKind.LLM, description="named ships or boats")


class TestNormalizeValue:
    def test_casefolds_and_collapses(self):
        assert normalize_value("  Split   Rock ") == "split rock"

    def test_numeric_leading_zeros(self):
        assert normalize_value("00482") == "482"
        assert normalize_value("0") == "0"


class TestSchemaPersistence:
    @pytest.fixture()
    def store(self, tmp_path):
        from lilbee.core.config import cfg
        from lilbee.data.store import Store

        return Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"}))

    def test_round_trip(self, store):
        save_schema(EntitySchema(types=[PART, PERSON]), store, applied=False, source_count=1)
        loaded = load_schema(store)
        assert loaded is not None
        assert [t.name for t in loaded.types] == ["part_number", "person"]
        state = store.entity_schema_state()
        assert state is not None
        assert state["applied"] is False  # saved but not yet applied

    def test_never_induced_reads_none(self, store):
        assert store.entity_schema_state() is None
        assert load_schema(store) is None

    def test_unreadable_row_reads_none(self, store):
        store.save_entity_schema("{not json", applied=False, source_count=1)
        assert load_schema(store) is None

    def test_mark_applied_flips_state(self, store):
        save_schema(EntitySchema(types=[PART]), store, applied=False, source_count=1)
        store.mark_entity_schema_applied()
        state = store.entity_schema_state()
        assert state is not None
        assert state["applied"] is True

    def test_mark_applied_without_schema_is_a_noop(self, store):
        store.mark_entity_schema_applied()
        assert store.entity_schema_state() is None

    def test_emptied_state_table_reads_none(self, store):
        """A state table whose row was deleted reads as never-induced."""
        from lilbee.core.config import ENTITY_SCHEMA_TABLE
        from lilbee.data.store.types import ENTITY_SCHEMA_DELETE_ALL_PREDICATE

        save_schema(EntitySchema(types=[PART]), store, applied=False, source_count=1)
        store.clear_table(ENTITY_SCHEMA_TABLE, ENTITY_SCHEMA_DELETE_ALL_PREDICATE)
        assert store.entity_schema_state() is None

    def test_save_overwrites_the_single_row(self, store):
        save_schema(EntitySchema(types=[PART]), store, applied=False, source_count=1)
        save_schema(EntitySchema(types=[PERSON]), store, applied=False, source_count=1)
        loaded = load_schema(store)
        assert loaded is not None
        assert [t.name for t in loaded.types] == ["person"]

    def test_type_for_noun_matches_synonyms_and_plurals(self):
        schema = EntitySchema(types=[PART])
        assert schema.type_for_noun("part numbers") is not None
        assert schema.type_for_noun("Part Number") is not None
        assert schema.type_for_noun("vessels") is None

    def test_type_for_noun_matches_irregular_plurals(self):
        schema = EntitySchema(types=[PERSON])
        assert schema.type_for_noun("people") is not None

    def test_type_for_noun_matches_ies_plurals(self):
        company = EntityType(name="company", kind=ExtractorKind.LLM, description="orgs")
        assert EntitySchema(types=[company]).type_for_noun("companies") is not None


class TestNounVariants:
    def test_inflects_only_the_last_word(self):
        assert "tail number" in noun_variants("tail numbers")
        assert "tail numbers" in noun_variants("tail number")

    def test_irregular_forms_map_both_ways(self):
        assert "people" in noun_variants("person")
        assert "person" in noun_variants("people")

    def test_es_and_ies_forms(self):
        assert "box" in noun_variants("boxes")
        assert "company" in noun_variants("companies")
        assert "companies" in noun_variants("company")

    def test_normalizes_case_and_spacing(self):
        assert "part number" in noun_variants("  Part   Numbers ")

    def test_blank_has_no_variants(self):
        assert noun_variants("   ") == set()


class TestInduceSchema:
    def test_parses_valid_proposal(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            "<think>identifiers dominate</think>"
            + json.dumps(
                {
                    "types": [
                        {
                            "name": "part number",
                            "kind": "regex",
                            "pattern": r"PX\d{4}",
                            "description": "catalog part ids",
                            "synonyms": ["part"],
                        },
                        {"name": "person", "kind": "spacy", "pattern": "PERSON"},
                    ]
                }
            )
        )
        schema = induce_schema(["part PX4471 shipped"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["part_number", "person"]

    def test_duplicate_extractors_collapse_to_the_first_name(self):
        """Small models sometimes propose several names for one pattern;
        identical extractors would fill the table with identical row sets."""
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {
                    "types": [
                        {"name": "title", "kind": "regex", "pattern": "[A-Z][a-z]+"},
                        {"name": "event", "kind": "regex", "pattern": "[A-Z][a-z]+"},
                        {"name": "person", "kind": "spacy", "pattern": "PERSON"},
                        {"name": "vessel", "kind": "llm", "description": "ships"},
                        {"name": "boat", "kind": "llm", "description": "ships"},
                    ]
                }
            )
        )
        schema = induce_schema(["text"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["title", "person", "vessel"]

    def test_drops_empty_pattern_for_pattern_bearing_kinds(self):
        """An empty pattern compiles fine but makes finditer yield a
        zero-width match at every position of every chunk in the corpus.
        The induction prompt tells the model to leave the pattern empty for
        llm kinds, so a confused model emitting regex+'' is realistic."""
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {
                    "types": [
                        {"name": "empty_regex", "kind": "regex", "pattern": ""},
                        {"name": "empty_spacy", "kind": "spacy", "pattern": ""},
                        {"name": "part_number", "kind": "regex", "pattern": r"PX\d{4}"},
                    ]
                }
            )
        )
        schema = induce_schema(["text"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["part_number"]

    def test_llm_kind_may_keep_an_empty_pattern(self):
        """llm kinds carry a description instead, exactly as the prompt asks."""
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {
                    "types": [
                        {"name": "vessel", "kind": "llm", "pattern": "", "description": "ships"}
                    ]
                }
            )
        )
        schema = induce_schema(["text"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["vessel"]

    def test_drops_bad_regex_keeps_rest(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {
                    "types": [
                        {"name": "broken", "kind": "regex", "pattern": "(["},
                        {"name": "person", "kind": "spacy", "pattern": "PERSON"},
                    ]
                }
            )
        )
        schema = induce_schema(["text"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["person"]

    def test_unparseable_response_is_none(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result("no json here")
        assert induce_schema(["text"], provider) is None

    def test_provider_error_is_none(self):
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("down")
        assert induce_schema(["text"], provider) is None

    def test_empty_sample_is_none(self):
        assert induce_schema([], MagicMock()) is None


class TestFirstJsonObjectEdges:
    def test_malformed_json_is_none(self):
        from lilbee.retrieval.entities.extractor import _first_json_object

        assert _first_json_object('{"a": }') is None

    def test_unbalanced_is_none(self):
        from lilbee.retrieval.entities.extractor import _first_json_object

        assert _first_json_object("{unclosed") is None

    def test_braces_inside_string_values_do_not_derail_the_scan(self):
        """A regex pattern or entity text containing a brace is legal JSON;
        a naive brace counter truncates mid-string and loses the object."""
        from lilbee.retrieval.entities.extractor import _first_json_object

        text = '{"types": [{"name": "x", "pattern": "\\\\}"}]}'
        parsed = _first_json_object(text)
        assert parsed == {"types": [{"name": "x", "pattern": "\\}"}]}

        assert _first_json_object('{"0": [{"type": "note", "text": "a } b"}]}') == {
            "0": [{"type": "note", "text": "a } b"}]
        }

    def test_object_after_stray_brace_in_prose_is_found(self):
        from lilbee.retrieval.entities.extractor import _first_json_object

        assert _first_json_object('opening { thoughts... {"kind": "x"}') == {"kind": "x"}


class TestInduceSchemaEdges:
    def test_non_dict_type_entry_dropped(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {"types": ["nonsense", {"name": "person", "kind": "spacy", "pattern": "PERSON"}]}
            )
        )
        schema = induce_schema(["text"], provider)
        assert schema is not None
        assert [t.name for t in schema.types] == ["person"]

    def test_invalid_type_name_rejected(self):
        with pytest.raises(Exception, match="alphanumeric"):
            EntityType(name="bad!name", kind=ExtractorKind.REGEX, pattern="x")


class _FakeEnt(SimpleNamespace):
    pass


class _FakeNlp:
    def __call__(self, text):
        ents = []
        if "Larsen" in text:
            ents.append(_FakeEnt(label_="PERSON", text="E. Larsen"))
        return SimpleNamespace(ents=ents)


class TestExtractEntities:
    def test_regex_kind(self):
        rows = extract_entities(
            [_chunk("parts PX4471 and PX9001, PX4471 again")],
            EntitySchema(types=[PART]),
        )
        assert {(r["entity"], r["normalized_value"]) for r in rows} == {
            ("PX4471", "px4471"),
            ("PX9001", "px9001"),
        }
        assert all(r["confidence"] == 1.0 for r in rows)

    def test_runaway_pattern_is_abandoned_and_named(self, caplog):
        """An induced pattern that blows its match budget must not stall the
        pass: the type is dropped with a warning naming it, and the other
        types still extract."""
        import lilbee.retrieval.entities.extractor as extractor_mod

        runaway = EntityType(name="runaway", kind=ExtractorKind.REGEX, pattern=r"(a+)+b")
        real_compile = extractor_mod._compile_pattern

        def fake_compile(pattern):
            if pattern == r"(a+)+b":
                stub = MagicMock()
                stub.finditer.side_effect = TimeoutError("regex timed out")
                return stub
            return real_compile(pattern)

        with (
            mock.patch.object(extractor_mod, "_compile_pattern", fake_compile),
            caplog.at_level("WARNING"),
        ):
            rows = extract_entities(
                [_chunk("parts PX4471 aaaaaaaaaa")],
                EntitySchema(types=[runaway, PART]),
            )
        assert {r["normalized_value"] for r in rows} == {"px4471"}
        assert any("runaway" in rec.message for rec in caplog.records)

    def test_runaway_pattern_is_not_retried_across_batches(self):
        """With stats carried across a full pass, a pattern that timed out
        once is skipped for the rest of the pass rather than costing the
        budget again on every chunk."""
        import lilbee.retrieval.entities.extractor as extractor_mod
        from lilbee.retrieval.entities.extractor import ExtractionStats

        runaway = EntityType(name="runaway", kind=ExtractorKind.REGEX, pattern=r"(a+)+b")
        stub = MagicMock()
        stub.finditer.side_effect = TimeoutError("regex timed out")

        stats = ExtractionStats()
        with mock.patch.object(extractor_mod, "_compile_pattern", lambda pattern: stub):
            extract_entities([_chunk("aaaa")], EntitySchema(types=[runaway]), stats=stats)
            extract_entities([_chunk("aaaa")], EntitySchema(types=[runaway]), stats=stats)
        assert stats.timed_out_types == {"runaway"}
        # Two chunks, but the pattern was only ever attempted once.
        assert stub.finditer.call_count == 1

    def test_anchored_pattern_matches_identifiers_inline(self):
        """The OCR-corpus failure: an induced pattern anchored with ^...$
        matches only a token that IS the identifier, so identifiers inline in
        running text (with adjacent punctuation) were never found. Anchored
        patterns must apply per token."""
        anchored = EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"^PX\d{4}$")
        rows = extract_entities(
            [_chunk("shipment via PX4471, then PX9001; see also (PX4471).")],
            EntitySchema(types=[anchored]),
        )
        assert {r["normalized_value"] for r in rows} == {"px4471", "px9001"}

    def test_anchored_pattern_still_rejects_partial_tokens(self):
        anchored = EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"^PX\d{4}$")
        rows = extract_entities(
            [_chunk("codes PX44712 and XPX4471 are different families")],
            EntitySchema(types=[anchored]),
        )
        assert rows == []

    def test_anchored_alternation_is_not_mangled(self):
        """^A$|^B$ must not have its outer anchors stripped (that would turn
        it into A$|^B); it falls through to plain finditer untouched."""
        alternation = EntityType(
            name="part_number", kind=ExtractorKind.REGEX, pattern=r"^PX\d{4}$|^QX\d{4}$"
        )
        rows = extract_entities(
            [_chunk("PX4471"), _chunk("inline PX4471 here", idx=1)],
            EntitySchema(types=[alternation]),
        )
        # Whole-token chunk still matches via finditer; inline stays unmatched,
        # same as before the per-token conversion existed.
        assert [r["chunk_index"] for r in rows] == [0]

    def test_spacy_kind_uses_injected_nlp(self):
        rows = extract_entities(
            [_chunk("keeper E. Larsen wrote nightly")],
            EntitySchema(types=[PERSON]),
            nlp=_FakeNlp(),
        )
        assert rows and rows[0]["type"] == "person"
        assert rows[0]["confidence"] == pytest.approx(0.8)

    def test_spacy_kind_skipped_without_nlp(self):
        rows = extract_entities(
            [_chunk("keeper E. Larsen wrote nightly")],
            EntitySchema(types=[PERSON]),
            nlp=None,
        )
        assert rows == []

    def test_llm_kind_batches_and_parses(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps({"0": [{"type": "vessel", "text": "the Meridian"}], "1": []})
        )
        rows = extract_entities(
            [_chunk("the Meridian docked"), _chunk("no ships here", idx=1)],
            EntitySchema(types=[VESSEL]),
            provider=provider,
        )
        assert len(rows) == 1
        assert rows[0]["entity"] == "the Meridian"
        assert rows[0]["confidence"] == pytest.approx(0.6)

    def test_llm_kind_skipped_without_provider(self):
        rows = extract_entities(
            [_chunk("the Meridian docked")], EntitySchema(types=[VESSEL]), provider=None
        )
        assert rows == []

    def test_llm_failure_degrades_to_empty(self):
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("down")
        rows = extract_entities(
            [_chunk("the Meridian docked")], EntitySchema(types=[VESSEL]), provider=provider
        )
        assert rows == []

    def test_llm_provider_failure_is_reported_through_stats(self):
        """A caller that passes stats can tell a clean empty result from a
        provider failure -- the full pass must not mark the schema applied
        when batches failed."""
        from lilbee.retrieval.entities.extractor import ExtractionStats

        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("down")
        stats = ExtractionStats()
        rows = extract_entities(
            [_chunk("the Meridian docked")],
            EntitySchema(types=[VESSEL]),
            provider=provider,
            stats=stats,
        )
        assert rows == []
        assert stats.llm_batches == 1
        assert stats.llm_batches_failed == 1

    def test_successful_llm_batches_count_zero_failures(self):
        from lilbee.retrieval.entities.extractor import ExtractionStats

        provider = MagicMock()
        provider.chat.return_value = _text_result(json.dumps({"0": []}))
        stats = ExtractionStats()
        extract_entities(
            [_chunk("no ships here")],
            EntitySchema(types=[VESSEL]),
            provider=provider,
            stats=stats,
        )
        assert stats.llm_batches == 1
        assert stats.llm_batches_failed == 0

    def test_llm_response_edge_shapes_are_ignored(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result(
            json.dumps(
                {
                    "zz": [{"type": "vessel", "text": "Meridian"}],
                    "9": [{"type": "vessel", "text": "OutOfRange"}],
                    "0": "not a list",
                    "1": [42, {"type": "unknown", "text": "x"}, {"type": "vessel", "text": ""}],
                }
            )
        )
        rows = extract_entities(
            [_chunk("one"), _chunk("two", idx=1)], EntitySchema(types=[VESSEL]), provider=provider
        )
        assert rows == []

    def test_llm_unparseable_response_yields_nothing(self):
        provider = MagicMock()
        provider.chat.return_value = _text_result("plain prose, no json")
        rows = extract_entities([_chunk("one")], EntitySchema(types=[VESSEL]), provider=provider)
        assert rows == []

    def test_whitespace_only_mention_is_dropped(self):
        blank = EntityType(name="blank", kind=ExtractorKind.REGEX, pattern=r" {2}")
        rows = extract_entities([_chunk("a  b")], EntitySchema(types=[blank]))
        assert rows == []

    def test_rows_carry_provenance_and_dedupe_per_chunk(self):
        rows = extract_entities(
            [_chunk("PX4471 PX4471", source="s.txt", idx=3, page=7)],
            EntitySchema(types=[PART]),
        )
        assert len(rows) == 1
        assert (rows[0]["source"], rows[0]["chunk_index"], rows[0]["page"]) == ("s.txt", 3, 7)
