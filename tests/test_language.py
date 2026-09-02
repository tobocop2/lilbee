"""Tests for the query-language pack seam."""

from __future__ import annotations

import re

from lilbee.retrieval.language import ENGLISH, QueryLanguage, noun_variants, query_language
from lilbee.retrieval.query.intent import AggregateKind, document_references, parse_aggregate

# A miniature non-English pack: Spanish-shaped count questions. Proves the
# parser routes by pack data, not by anything hardcoded in the logic.
SPANISH_LIKE = QueryLanguage(
    code="xx",
    doc_ref_pattern=re.compile(r"\b(?:documento|informe)\s+#?([\w][\w.-]{0,23})\b", re.IGNORECASE),
    ref_stopwords=frozenset({"que", "el", "la"}),
    stopwords=frozenset({"que", "el", "la", "los", "las", "de", "en", "es", "son", "tienes"}),
    how_many_pattern=re.compile(r"^\s*cu[aá]ntos\b", re.IGNORECASE),
    total_pattern=re.compile(r"cu[aá]ntos\s+documentos\s+hay[?\s]*$", re.IGNORECASE),
    association_pattern=re.compile(
        r"cu[aá]ntos\s+(.+?)\s+tiene\s+cada\s+(.+?)[?\s]*$", re.IGNORECASE
    ),
    per_pattern=re.compile(r"cu[aá]ntos\s+(.+?)\s+por\s+(.+?)[?\s]*$", re.IGNORECASE),
    distinct_pattern=re.compile(
        r"cu[aá]ntos\s+(?:distintos|diferentes)\s+(.+?)[?\s]*$", re.IGNORECASE
    ),
    term_mention_pattern=re.compile(
        r"cu[aá]ntos\s+documentos\s+mencionan\s+(.+?)[?\s]*$", re.IGNORECASE
    ),
    listing_pattern=re.compile(r"^\s*qu[eé]\s+documentos\s+tienes[?\s]*$", re.IGNORECASE),
    leading_article_pattern=re.compile(r"^(?:el|la|los|las)\s+", re.IGNORECASE),
    known_item_patterns=(re.compile(r"^\s*resume\s+(.+?)[?\s]*$", re.IGNORECASE),),
    noun_variants=lambda noun: {noun.strip().lower()},
)


class TestActivePack:
    def test_active_pack_is_english(self):
        assert query_language() is ENGLISH
        assert query_language().code == "en"

    def test_module_noun_variants_delegates_to_active_pack(self):
        assert noun_variants("tail numbers") == ENGLISH.noun_variants("tail numbers")


class TestPackDrivenParsing:
    def test_english_pack_parses_english_count_question(self):
        parsed = parse_aggregate("how many documents mention the observatory")
        assert parsed is not None
        assert parsed.kind is AggregateKind.TERM_MENTIONS
        assert parsed.term == "observatory"

    def test_foreign_pack_parses_its_own_count_question(self):
        parsed = parse_aggregate("cuántos documentos mencionan el observatorio?", SPANISH_LIKE)
        assert parsed is not None
        assert parsed.kind is AggregateKind.TERM_MENTIONS
        assert parsed.term == "observatorio"

    def test_foreign_pack_ignores_english_shapes(self):
        assert parse_aggregate("how many documents are there", SPANISH_LIKE) is None
        assert parse_aggregate("what files can you see?", SPANISH_LIKE) is None

    def test_foreign_pack_parses_its_own_listing_question(self):
        parsed = parse_aggregate("qué documentos tienes?", SPANISH_LIKE)
        assert parsed is not None
        assert parsed.kind is AggregateKind.SOURCE_LISTING

    def test_foreign_pack_doc_references(self):
        refs = document_references("resume el documento 214", SPANISH_LIKE)
        assert refs == ["214"]

    def test_foreign_pack_title_candidates(self):
        from lilbee.retrieval.query.intent import title_candidates

        assert title_candidates("resume Don Quijote", SPANISH_LIKE) == ["Don Quijote"]
        assert title_candidates("summarize Don Quijote", SPANISH_LIKE) == []

    def test_foreign_pack_association(self):
        parsed = parse_aggregate("cuántos vuelos tiene cada persona?", SPANISH_LIKE)
        assert parsed is not None
        assert parsed.kind is AggregateKind.TYPE_ASSOCIATION
        assert parsed.noun == "vuelos"
        assert parsed.group_noun == "persona"


class TestEnglishMorphology:
    def test_irregular_plural_both_ways(self):
        assert "people" in noun_variants("person")
        assert "person" in noun_variants("people")

    def test_phrase_varies_last_word_only(self):
        variants = noun_variants("tail numbers")
        assert "tail number" in variants
        assert "tail numbers" in variants

    def test_empty_noun(self):
        assert noun_variants("   ") == set()
