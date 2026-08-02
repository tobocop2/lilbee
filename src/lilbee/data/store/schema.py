"""PyArrow schemas for the LanceDB tables managed by the store."""

from __future__ import annotations

import pyarrow as pa


def _meta_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("embedding_model", pa.utf8()),
            pa.field("embedding_dim", pa.int32()),
            pa.field("schema_version", pa.int32()),
            pa.field("updated_at", pa.utf8()),
        ]
    )


def _entity_schema_state_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_json", pa.utf8()),
            pa.field("applied", pa.bool_()),
            pa.field("source_count", pa.int32()),
            pa.field("updated_at", pa.utf8()),
        ]
    )


def _sources_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("filename", pa.utf8()),
            pa.field("file_hash", pa.utf8()),
            pa.field("ingested_at", pa.utf8()),
            pa.field("chunk_count", pa.int32()),
            pa.field("source_type", pa.utf8()),
            pa.field("size_bytes", pa.int64()),
            pa.field("mtime_ns", pa.int64()),
            pa.field("stat_captured_ns", pa.int64()),
            pa.field("title", pa.utf8()),
            pa.field("authors", pa.utf8()),
            pa.field("created_at", pa.utf8()),
        ]
    )


def _page_texts_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("page", pa.int32()),
            pa.field("text", pa.utf8()),
            pa.field("content_type", pa.utf8()),
        ]
    )


def _citations_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("wiki_source", pa.utf8()),
            pa.field("wiki_chunk_index", pa.int32()),
            pa.field("citation_key", pa.utf8()),
            pa.field("claim_type", pa.utf8()),
            pa.field("source_filename", pa.utf8()),
            pa.field("source_hash", pa.utf8()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("excerpt", pa.utf8()),
            pa.field("created_at", pa.utf8()),
        ]
    )


def _wiki_mentions_schema() -> pa.Schema:
    """One row per (subject, source): the subject's mention evidence in that
    source. ``slug`` is the make_slug key the wiki stub index uses; the wiki
    aggregates these rows across sources to judge the corpus-wide floor."""
    return pa.schema(
        [
            pa.field("slug", pa.utf8()),
            pa.field("label", pa.utf8()),
            pa.field("kind", pa.utf8()),
            pa.field("type_hint", pa.utf8()),
            pa.field("source", pa.utf8()),
            pa.field("mention_count", pa.int32()),
            pa.field("chunk_indices", pa.list_(pa.int32())),
        ]
    )
