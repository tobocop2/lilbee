"""Arrow schemas for the concept graph LanceDB tables."""

from __future__ import annotations

import pyarrow as pa


def _concept_nodes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("concept", pa.utf8()),
            pa.field("cluster_id", pa.int32()),
            pa.field("degree", pa.int32()),
        ]
    )


def _concept_edges_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("target", pa.utf8()),
            pa.field("weight", pa.float32()),
        ]
    )


def _chunk_concepts_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_source", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("concept", pa.utf8()),
        ]
    )
