"""Tests for entity schema induction and two-phase extraction."""

import json
from types import SimpleNamespace
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


class TestSchemaArtifact:
    def test_round_trip(self, tmp_path):
        schema = EntitySchema(types=[PART, PERSON])
        path = save_schema(schema, tmp_path)
        assert path.name == "entity_schema.json"
        loaded = load_schema(tmp_path)
        assert loaded is not None
        assert [t.name for t in loaded.types] == ["part_number", "person"]

    def test_absent_reads_none(self, tmp_path):
        assert load_schema(tmp_path) is None

    def test_unreadable_reads_none(self, tmp_path):
        (tmp_path / "entity_schema.json").write_text("{not json")
        assert load_schema(tmp_path) is None

    def test_type_for_noun_matches_synonyms_and_plurals(self):
        schema = EntitySchema(types=[PART])
        assert schema.type_for_noun("part numbers") is not None
        assert schema.type_for_noun("Part Number") is not None
        assert schema.type_for_noun("vessels") is None


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
