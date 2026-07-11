"""The ``lilbee entities`` command: schema induction, backfill, status.

The mode is deliberately two-phase and operator-driven: ``induce`` samples
the indexed chunks and writes a reviewable ``entity_schema.json``; after the
schema is inspected (and edited if needed), ``backfill`` applies it across
every stored chunk without re-ingesting documents, and syncs keep new files
current once ``entity_extraction`` is enabled.
"""

from __future__ import annotations

from pathlib import Path

import typer

from lilbee.app.services import get_services
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output
from lilbee.core.config import CHUNKS_TABLE, cfg

# Chunks per extraction batch during backfill; bounds the working set.
_BACKFILL_BATCH = 2000


def _sample_chunks(limit: int) -> list[str]:
    """Stratified sample: chunks read evenly across the table so a corpus with
    several document families contributes all of them."""
    store = get_services().store
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        return []
    total = table.count_rows()
    if total == 0:
        return []
    step = max(1, total // limit)
    texts: list[str] = []
    arrow = table.to_arrow().select(["chunk"])
    index = 0
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        for text in batch.column("chunk").to_pylist():
            if index % step == 0 and len(texts) < limit:
                texts.append(text)
            index += 1
    return texts


def entities(
    action: str = typer.Argument(help="induce | backfill | status"),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="backfill only: clear previously extracted rows first, so repeated "
        "backfills (after a pattern fix) don't double-count",
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Typed entity extraction: propose a schema, backfill the index, or inspect state.

    ``induce`` writes entity_schema.json next to the index for review;
    ``backfill`` extracts entities from every stored chunk under the reviewed
    schema (no document re-ingest); ``status`` shows the schema and row counts.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if action == "induce":
        _induce()
    elif action == "backfill":
        _backfill(replace=replace)
    elif action == "status":
        _status()
    else:
        console.print(f"Unknown action {action!r}: expected induce, backfill, or status")
        raise SystemExit(2)


def _induce() -> None:
    from lilbee.retrieval.entities import induce_schema, save_schema
    from lilbee.retrieval.entities.extractor import INDUCTION_SAMPLE_SIZE

    texts = _sample_chunks(INDUCTION_SAMPLE_SIZE)
    if not texts:
        console.print("Nothing indexed yet; sync documents before inducing a schema.")
        raise SystemExit(1)
    schema = induce_schema(texts, get_services().provider)
    if schema is None:
        console.print("Schema induction produced nothing usable; check the chat model and retry.")
        raise SystemExit(1)
    path = save_schema(schema, cfg.data_dir)
    if cfg.json_mode:
        json_output({"command": "entities", "action": "induce", "schema": str(path)})
        return
    console.print(f"Proposed {len(schema.types)} types; review and edit {path}")
    for entity_type in schema.types:
        console.print(f"  {entity_type.name} ({entity_type.kind.value})")
    console.print("Then run: lilbee entities backfill")


def _backfill(*, replace: bool = False) -> None:
    from lilbee.core.config import ENTITIES_TABLE
    from lilbee.retrieval.concepts import concepts_available
    from lilbee.retrieval.entities import ExtractorKind, extract_entities, load_schema

    schema = load_schema(cfg.data_dir)
    if schema is None:
        console.print("No reviewed entity_schema.json found; run: lilbee entities induce")
        raise SystemExit(1)
    store = get_services().store
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        console.print("Nothing indexed yet; sync documents first.")
        raise SystemExit(1)
    if replace:
        store.clear_table(ENTITIES_TABLE, "entity IS NOT NULL")
    nlp = None
    if any(t.kind is ExtractorKind.SPACY for t in schema.types) and concepts_available():
        from lilbee.retrieval.concepts.nlp import _ensure_spacy_model

        nlp = _ensure_spacy_model()
    provider = None
    if any(t.kind is ExtractorKind.LLM for t in schema.types):
        provider = get_services().provider
    written = 0
    columns = ["chunk", "source", "chunk_index", "page_start"]
    arrow = table.to_arrow().select(columns)
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        records = batch.to_pylist()
        rows = extract_entities(records, schema, provider=provider, nlp=nlp)
        written += store.add_entities(rows)
    if cfg.json_mode:
        json_output({"command": "entities", "action": "backfill", "rows": written})
        return
    console.print(f"Backfill complete: {written} entity rows extracted.")


def _status() -> None:
    from lilbee.core.config import ENTITIES_TABLE
    from lilbee.retrieval.entities import load_schema, schema_path

    schema = load_schema(cfg.data_dir)
    table = get_services().store.open_table(ENTITIES_TABLE)
    rows = table.count_rows() if table is not None else 0
    if cfg.json_mode:
        json_output(
            {
                "command": "entities",
                "action": "status",
                "schema": str(schema_path(cfg.data_dir)) if schema else None,
                "types": [t.name for t in schema.types] if schema else [],
                "rows": rows,
                "extraction_enabled": cfg.entity_extraction,
            }
        )
        return
    if schema is None:
        console.print("No entity schema; run: lilbee entities induce")
    else:
        names = ", ".join(t.name for t in schema.types)
        console.print(f"Schema: {schema_path(cfg.data_dir)} ({names})")
    console.print(f"Extracted rows: {rows}")
    console.print(f"Sync-time extraction: {'on' if cfg.entity_extraction else 'off'}")
