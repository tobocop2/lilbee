"""Tests for query intent detection and routing."""

import pytest

from lilbee.retrieval.query.intent import (
    AggregateKind,
    document_references,
    matches_reference,
    parse_aggregate,
)


class TestDocumentReferences:
    def test_filename_reference(self):
        assert document_references("summarize survey_report.pdf for me") == ["survey_report.pdf"]

    def test_document_number_reference(self):
        assert document_references("summarize Document 214") == ["214"]

    def test_quoted_reference(self):
        assert document_references('what does "harbor survey 2010" say') == ["harbor survey 2010"]

    def test_filename_beats_doc_number(self):
        refs = document_references("compare report 12 with summary_final.md")
        assert refs[0] == "summary_final.md"

    def test_topical_question_yields_nothing(self):
        assert document_references("which documents mention the harbor?") == []
        assert document_references("what do the files say about travel?") == []

    def test_generic_noun_after_document_is_not_a_reference(self):
        assert document_references("is there a document that lists the deliveries?") == []


class TestMatchesReference:
    def test_numeric_ref_matches_zero_padded_token(self):
        assert matches_reference("482", "ARC-REC-00000482.pdf")

    def test_numeric_ref_rejects_different_number_with_shared_suffix(self):
        assert not matches_reference("482", "ARC-REC-00010482.pdf")

    def test_word_ref_matches_token_case_insensitively(self):
        assert matches_reference("12b", "exhibit-12B-scan.pdf")
        assert not matches_reference("12b", "exhibit-12-scan.pdf")

    def test_delimiter_edges_yield_no_empty_token_match(self):
        # A leading delimiter makes the split emit an empty first token,
        # which must be skipped, not compared.
        assert matches_reference("482", "_482_final.pdf")
        assert not matches_reference("", "_482_final.pdf")

    def test_filename_ref_matches_whole_name(self):
        assert matches_reference("survey_report.pdf", "survey_report.pdf")
        assert matches_reference("survey_report.pdf", "archive/survey_report.pdf")


class TestParseAggregate:
    def test_non_count_question_is_none(self):
        assert parse_aggregate("what did the keeper record?") is None

    def test_term_mention_count(self):
        agg = parse_aggregate("how many documents mention the observatory?")
        assert agg is not None
        assert agg.kind is AggregateKind.TERM_MENTIONS
        assert agg.term == "observatory"

    def test_roughly_prefix(self):
        agg = parse_aggregate("Roughly how many passages reference kerosene?")
        assert agg is not None
        assert agg.kind is AggregateKind.TERM_MENTIONS
        assert agg.term == "kerosene"

    def test_total_sources(self):
        agg = parse_aggregate("how many documents are there?")
        assert agg is not None
        assert agg.kind is AggregateKind.TOTAL_SOURCES

    def test_typed_count_is_unsupported_not_topical(self):
        """Counts over records the store does not hold must be declined
        precisely, not fed to retrieval that cannot count."""
        agg = parse_aggregate("how many shipments is each part number associated with?")
        assert agg is not None
        assert agg.kind is AggregateKind.UNSUPPORTED


class TestCountTermMentions:
    @pytest.fixture()
    def store(self, tmp_path):
        from lilbee.core.config import cfg
        from lilbee.data.store import Store

        config = cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"})
        store = Store(config)
        dim = config.embedding_dim
        store.add_chunks(
            [
                {
                    "source": f"doc{i}.md",
                    "content_type": "text",
                    "chunk_type": "raw",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": text,
                    "chunk_index": 0,
                    "vector": [0.1] * dim,
                }
                for i, text in enumerate(
                    [
                        "the Lighthouse Trust funded the repair",
                        "no mention here",
                        "lighthouse trust grants were reviewed; the trust replied",
                        "unrelated text",
                    ]
                )
            ]
        )
        return store

    def test_counts_are_exact_and_case_insensitive(self, store):
        chunk_hits, source_hits = store.count_term_mentions("lighthouse")
        assert chunk_hits == 2
        assert source_hits == 2

    def test_zero_for_absent_term(self, store):
        assert store.count_term_mentions("zebra") == (0, 0)

    def test_empty_store_counts_zero(self, tmp_path):
        from lilbee.core.config import cfg
        from lilbee.data.store import Store

        config = cfg.model_copy(update={"lancedb_dir": tmp_path / "empty_lancedb"})
        empty = Store(config)
        assert empty.count_term_mentions("anything") == (0, 0)
        assert empty.count_chunks() == 0

    def test_count_chunks(self, store):
        assert store.count_chunks() == 4
