"""The ``Store`` class: high-level LanceDB read/write API used across lilbee."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, cast

import pyarrow as pa
import pyarrow.compute as pc

from lilbee.core import settings
from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CHUNKS_TABLE,
    CITATIONS_TABLE,
    ENTITIES_TABLE,
    ENTITY_SCHEMA_TABLE,
    MEMORIES_TABLE,
    META_TABLE,
    PAGE_TEXTS_TABLE,
    SOURCES_TABLE,
    WIKI_MENTIONS_TABLE,
    Config,
)
from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.core.security import write_private_text
from lilbee.core.vectors import Vector
from lilbee.retrieval.embedding_profiles import resolve_embedding_profile
from lilbee.runtime.lock import (
    LOCK_TIMEOUT,
    LockingUnsupportedError,
    LockTimeoutError,
    write_lock,
)
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    LibraryUpgradeEvent,
    UpgradePhase,
    noop_callback,
)

from .fusion import adaptive_weight_scale, fuse_arms, normalized_bm25, vector_similarity
from .held_import import legacy_files, read_legacy_records, remove_legacy_files
from .lance_helpers import (
    _CHUNK_COLUMN,
    _chunk_type_predicate,
    _has_fts_index,
    _has_scalar_index,
    _has_vector_index,
    _index_registry,
    _safe_delete_unlocked,
    _scalar_index_dangling,
    _vector_index_dangling,
    ensure_table,
    escape_sql_string,
    open_committed,
    refs_compatible,
    table_names,
)
from .locations import SourceLocations, root_location
from .ranking import mmr_rerank
from .schema import (
    _citations_schema,
    _entity_schema_state_schema,
    _meta_schema,
    _page_texts_schema,
    _sources_schema,
    _wiki_mentions_schema,
)
from .types import (
    ENTITY_SCHEMA_DELETE_ALL_PREDICATE,
    META_DELETE_ALL_PREDICATE,
    META_SCHEMA_VERSION,
    READ_CONSISTENCY_INTERVAL,
    SOURCE_STAT_UNKNOWN,
    ChunkType,
    ChunkWrite,
    CitationRecord,
    EmbeddingModelMismatchError,
    EntitySchemaState,
    HeldOut,
    HeldRecord,
    MemoryKind,
    MemoryRow,
    PageTextRecord,
    RemoveResult,
    SearchChunk,
    SourceMeta,
    SourceMove,
    SourceRecord,
    SourceRelocation,
    SourcesCheckpoint,
    SourceStat,
    SourceStatBackfill,
    SourceType,
    StoreMeta,
)

if TYPE_CHECKING:
    from lancedb.db import LanceDBConnection
    from lancedb.expr import Expr
    from lancedb.index import FTS, IndexConfig
    from lancedb.query import LanceFtsQueryBuilder
    from lancedb.table import LanceTable

log = logging.getLogger(__name__)

# Batched ingest flushes contend with long store operations (a search-triggered
# FTS optimize can hold the lock past the interactive 30s), and failing the
# flush replans and re-embeds the whole batch. Give the batch path more
# patience than interactive writes before it gives up.
BATCH_LOCK_TIMEOUT = 120.0
# Lock budget for index builds reached from the read path: a query must not
# stall behind a long ingest, so it skips the build and retries next search.
_READ_LOCK_TIMEOUT = 2.0
# What makes a read-path index build skip rather than fail the query.
_UPKEEP_SKIPPED = (LockTimeoutError, LockingUnsupportedError)


def _drop_unsupported_far_rows(
    results: list[SearchChunk], max_distance: float
) -> list[SearchChunk]:
    """Apply ``max_distance`` to rows whose only signal is the vector arm.

    A row the BM25 arm also matched keeps lexical support regardless of its
    vector distance; dropping it on distance alone would re-bury exactly the
    identifier hits rank fusion exists to preserve.
    """
    if max_distance <= 0:
        return results
    return [
        r
        for r in results
        if r.bm25_score is not None or r.distance is None or r.distance <= max_distance
    ]


_MAX_THRESHOLD = 1.0
_MAX_FILTER_ITERATIONS = 20  # safety cap to prevent runaway loops


def _is_fts_position_overflow(exc: Exception) -> bool:
    """True when *exc* is LanceDB's positional-FTS list-encoding overflow.

    A positional index (built by an intermediate dev commit) raises e.g.
    "Max offset N exceeds length of values M" on optimize(); a positionless
    rebuild is the remediation. Matched on message because LanceDB raises it
    as a generic error type.
    """
    msg = str(exc).lower()
    return "offset" in msg and "exceeds" in msg


def _lexical_rows(
    table: LanceTable,
    query_text: str,
    limit: int,
    chunk_type: ChunkType | None,
    column: str = _CHUNK_COLUMN,
) -> list[dict[str, Any]]:
    """BM25 rows for *query_text* over a single FTS *column*.

    ``MatchQuery`` pins the column and matches plain terms, so an unpinned search
    cannot widen to the title index and a quoted span cannot reach LanceDB as a
    phrase (which the positionless index rejects). This is the one place FTS
    queries are built; every arm goes through it.
    """
    from lancedb.query import MatchQuery

    # lancedb's stubs omit FullTextQuery from the fts overload and read MatchQuery's
    # kw-only defaults as required; the call is the documented FTS query form.
    query: LanceFtsQueryBuilder = table.search(  # type: ignore[call-overload]
        MatchQuery(query_text, column),  # type: ignore[call-arg]
        query_type="fts",
    ).limit(limit)
    if chunk_type:
        query = query.where(_chunk_type_predicate(chunk_type))
    return cast("list[dict[str, Any]]", query.to_list())


# Vector ANN index. IVF_PQ compresses vectors so search scales to millions;
# refine_factor re-ranks the PQ candidates against full vectors to recover recall.
# The index type is carried by the lancedb IvfPq config at build time.
_VECTOR_METRIC: Final = "cosine"
# The scalar index kinds lilbee builds on the columns its queries filter by.
ScalarIndexType = Literal["BTREE", "BITMAP"]
_ANN_NPROBES_FLOOR = 20
# Fraction of IVF partitions probed per query. 0.05 was the "fast" end of the
# recall/latency curve and measurably cost recall at scale: on the 8.8M-passage
# MS MARCO index it gave recall@100 of 67%, where probing more partitions
# reached 87% (docs/benchmarks/retrieval-msmarco.md). 0.15 is the "balanced /
# high-recall" range for IVF and recovers most of that gap; refine_factor below
# still re-ranks the survivors against full vectors.
_ANN_NPROBES_PARTITION_FRACTION = 0.15
_ANN_REFINE_FACTOR = 10

# Stat columns of ``_sources``; mirrors the field names in ``schema._sources_schema``
# and ``types.SourceRecord``. Legacy tables that predate these columns are migrated
# in place with the SOURCE_STAT_UNKNOWN sentinel.
_SOURCE_STAT_COLUMNS = ("size_bytes", "mtime_ns", "stat_captured_ns")

# Extraction-metadata columns of ``_sources``; nullable strings, so legacy
# tables migrate in place with NULL (meaning "extractor reported nothing").
_SOURCE_META_COLUMNS = ("title", "authors", "created_at")

# Document-title column of the chunks table; nullable so pre-title rows and
# writers that carry no title (wiki pages) read as NULL.
_TITLE_COLUMN = "title"

# The column that names a source in every table; ``_sources`` alone maps it to a path.
_SOURCE_ID = "source_id"
# The name of each thread an upgrade adds a table's id column on.
_UPGRADE_THREAD_PREFIX = "library-upgrade"
# The ``_sources`` column that holds a source's path.
_PATH = "path"
# The ``_sources`` column that names why a sync holds a row's path out (``HeldOut``).
_HELD_OUT = "held_out"
# The content hash a sync holds the path out at, and the reason shown for it.
_HELD_HASH = "held_hash"
_HELD_REASON = "held_reason"
_HELD_COLUMNS = (_HELD_OUT, _HELD_HASH, _HELD_REASON)

# Tables that hold a source's rows, deleted with the source. The concept
# nodes/edges tables carry no source column (corpus-level aggregates).
_PER_SOURCE_TABLES = (
    CHUNKS_TABLE,
    PAGE_TEXTS_TABLE,
    CHUNK_CONCEPTS_TABLE,
    ENTITIES_TABLE,
    WIKI_MENTIONS_TABLE,
)

# Each table that named a source by its path before ``source_id``, and that column.
_PATH_KEY_COLUMNS = (
    (CHUNKS_TABLE, "source"),
    (PAGE_TEXTS_TABLE, "source"),
    (CHUNK_CONCEPTS_TABLE, "chunk_source"),
    (ENTITIES_TABLE, "source"),
    (CITATIONS_TABLE, "source_filename"),
    (WIKI_MENTIONS_TABLE, "source"),
)
_PATH_KEY_BY_TABLE = dict(_PATH_KEY_COLUMNS)
# The tables an upgrade visits, in its order: ``_sources`` first.
_UPGRADE_ORDER = (SOURCES_TABLE, *_PATH_KEY_BY_TABLE)
# The path column of ``_sources`` before it was renamed.
_OLD_PATH = "filename"
# The ``_meta`` rows an upgrade has yet to stamp; a store stamped 1 keeps 1.
_UNSTAMPED = f"schema_version >= 2 AND schema_version < {META_SCHEMA_VERSION}"

# The source rows a listing shows: each that holds a document, failed at a later edit or not.
_LISTED = (
    f"source_type != '{SourceType.WIKI}' AND ({_HELD_OUT} IS NULL"
    f" OR {_HELD_OUT} = '{HeldOut.FAILED}' AND file_hash IS NOT NULL)"
)
# The source rows stored at their location; every other source is stored by its key.
_DOCUMENT = f"source_type = '{SourceType.DOCUMENT}'"
# The source rows of wiki pages, which share no row with a source of another type.
_WIKI = f"source_type = '{SourceType.WIKI}'"
_SOURCE_TYPE = "source_type"
_CHUNK_TYPE = "chunk_type"
# The chunks of wiki pages; an upgrade gives them the id of a page, never of a document.
_WIKI_CHUNK = f"{_CHUNK_TYPE} = '{ChunkType.WIKI}'"
# What a wiki page's key is joined with before its md5 gives the page its id at an upgrade.
_WIKI_ID_SUFFIX = "wiki"
_WIKI_ID_SQL = "md5(concat({key}, chr(0), '" + _WIKI_ID_SUFFIX + "'))"
# The source rows a move found by a sync carries: every row but a wiki page's.
_MOVABLE = f"NOT ({_WIKI})"

# Sentinel: relocation must leave the stored title untouched (extraction-derived).
_KEEP_TITLE = "\x00keep"

# The key columns a batched flush deletes, looks up or merges by in each table it writes.
_FLUSH_KEY_COLUMNS = (
    (CHUNKS_TABLE, _SOURCE_ID),
    (PAGE_TEXTS_TABLE, _SOURCE_ID),
    (SOURCES_TABLE, _PATH),
    (SOURCES_TABLE, _SOURCE_ID),
)

# A filtered delete scans every row its key index does not cover. The index is
# rebuilt once this many flushes of rows sit past it.
_KEY_INDEX_STALE_FLUSHES = 32

# Source rows per locked write or per ``IN`` list: the first sync after a
# stat-column upgrade backfills every source, and an unchunked replace would
# join millions of paths into one predicate.
_SOURCE_STAT_BATCH_ROWS = 2000

# One hold of the write lock moves at most this many sources, so a writer that
# waits is not kept out for the length of a whole plan batch.
_RELOCATE_HOLD_MOVES = 2000

_NULL_TEXT_SQL = "CAST(NULL AS STRING)"

# What a dropped source row a citation names keeps: its path, and no record a sync reads.
_CITED_ONLY_SQL = {
    _HELD_OUT: f"'{HeldOut.REMOVED}'",
    "chunk_count": "0",
    _HELD_HASH: _NULL_TEXT_SQL,
    _HELD_REASON: _NULL_TEXT_SQL,
}

# Rows a location fill reads and writes back in one batch: bounds its memory at any row count.
_FILL_BATCH_ROWS = 8192
# The address LanceDB gives a row in one version of its table.
_ROW_ID = "_rowid"
# The scratch table a fill pairs each row already at a new place with the row moving onto it in.
_FILL_PAIRS_TABLE = "_fill_pairs"
_LANDING_ID = "landing_id"
_MOVED_ID = "moved_id"
_FILL_PAIRS_SCHEMA = pa.schema(
    [pa.field(_PATH, pa.utf8()), pa.field(_LANDING_ID, pa.utf8()), pa.field(_MOVED_ID, pa.utf8())]
)
# Ids one citation update re-points: its lookup list is built again for every row it reads.
_CITATION_LOOKUP_IDS = 128

# Beside the tables: the checkpoint up to which the wiki's page index covers the source rows.
_WIKI_REFRESHED_NAME = ".wiki_refreshed"
_WIKI_REFRESHED_UNREADABLE = "Cannot read %s (%s): the wiki index reads every source again"

# Rows per Arrow batch when a filtered read walks the whole source table.
_SOURCE_SCAN_BATCH_ROWS = 20_000

# Rows per Arrow batch when the aggregate scan walks the whole chunks table;
# bounds the decoded-text working set while the scan stays columnar.
_TERM_SCAN_BATCH_ROWS = 20_000

# The title arm collapses each matched document to one row, so it over-fetches
# to gather enough distinct documents before deduping. Bounded so a title that
# hits a huge document can't scan the whole corpus.
_TITLE_FETCH_FACTOR = 20
_TITLE_MIN_FETCH = 200
_TITLE_FETCH_CEILING = 4096


def _ann_nprobes(row_count: int) -> int:
    """Partitions to probe: a fixed fraction of the IVF partition count (~sqrt(N)), floored."""
    partitions = math.isqrt(max(row_count, 0))
    return max(_ANN_NPROBES_FLOOR, math.ceil(partitions * _ANN_NPROBES_PARTITION_FRACTION))


def _check_vector_dims(records: list[dict], embedding_dim: int) -> None:
    """Raise ``ValueError`` when any record's vector is not *embedding_dim* wide."""
    for rec in records:
        vec = rec.get("vector", [])
        if len(vec) != embedding_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {embedding_dim}, "
                f"got {len(vec)} (source={rec.get('source', '?')})"
            )


def _citations_for_wiki_predicate(wiki_source: str) -> str:
    """SQL predicate selecting every citation row belonging to *wiki_source*."""
    return f"wiki_source = '{escape_sql_string(wiki_source)}'"


def _under_sql(column: str, old: str) -> str:
    """The filter for *old* and each key below it.

    ``starts_with`` takes the key literally, so no character in a key is a wildcard.
    """
    old_literal = escape_sql_string(old)
    return f"{column} = '{old_literal}' OR starts_with({column}, '{old_literal}/')"


def _swap_head_sql(column: str, old_head: str, new_head: str) -> str:
    """The SQL value of *column* with *new_head* in place of the characters of *old_head*."""
    return f"concat('{escape_sql_string(new_head)}', substr({column}, {len(old_head) + 1}))"


def _in_list(names: Iterable[str]) -> str:
    """The body of a SQL ``IN`` list that holds *names*."""
    return ", ".join(f"'{escape_sql_string(name)}'" for name in names)


def _claimed_names(
    moves: Sequence[SourceMove], unclaimed: set[str], removed: set[str], tails: Sequence[str | None]
) -> list[str | None]:
    """The old name each of *moves* takes out of *unclaimed*: its first candidate still there.

    A move takes a candidate of *removed* first, and only one whose location ends
    in the move's tail of *tails*, the part of its new place below its root: a
    removal stays with a file whose name below its root is unchanged. Moves that
    share one candidates tuple share one cursor into it, so files of one content
    cost one pass over their old names and not one for each file.
    """
    cursors: dict[int, int] = {}
    removed_in: dict[int, list[str]] = {}
    claimed: list[str | None] = []
    for move, tail in zip(moves, tails, strict=True):
        pool = move.candidates
        held = removed_in.setdefault(id(pool), [name for name in pool if name in removed])
        kept = next((name for name in held if tail and name.endswith(tail)), None)
        if kept is not None:
            held.remove(kept)
            claimed.append(kept)
            continue
        at = cursors.get(id(pool), 0)
        while at < len(pool) and pool[at] not in unclaimed:
            at += 1
        cursors[id(pool)] = at
        if at == len(pool):
            claimed.append(None)
            continue
        unclaimed.discard(pool[at])
        claimed.append(pool[at])
    return claimed


def _located_moves(
    moves: Sequence[SourceMove], locations: SourceLocations
) -> tuple[list[SourceMove], dict[str, str]]:
    """*moves* with every name as its location, and the name of each location.

    Moves that share one candidates tuple still share one, which the claim reads.
    """
    names: dict[str, str] = {}

    def _at(name: str) -> str:
        names[location := locations.location(name)] = name
        return location

    pools = {id(move.candidates): tuple(map(_at, move.candidates)) for move in moves}
    located = [SourceMove(pools[id(move.candidates)], _at(move.new), move.stat) for move in moves]
    return located, names


def _title_after_move(
    row: Mapping[str, Any], old: str, new: str, derive: Callable[[str], str]
) -> str | None:
    """The title the source *row* takes when *old* becomes *new*, or ``_KEEP_TITLE``.

    A title that came from the old name follows the new one; any other title stays,
    and so does a title the new name derives again, so that no row is written for it.
    """
    stored = row[_TITLE_COLUMN] or ""
    if stored != (derive(old) or "") or stored == (derive(new) or ""):
        return _KEEP_TITLE
    return derive(new) or None


def _moved_source_row(row: dict[str, Any], move: SourceMove, title: str | None) -> dict[str, Any]:
    """The source *row* under *move*'s new name, with *title* and the stat of the file."""
    moved = {**row, _PATH: move.new}
    if title is not _KEEP_TITLE:
        moved[_TITLE_COLUMN] = title
    if move.stat is not None:
        moved["size_bytes"] = move.stat.size_bytes
        moved["mtime_ns"] = move.stat.mtime_ns
        moved["stat_captured_ns"] = move.stat.captured_ns
    return moved


def _member_rows(rows: Iterable[dict[str, Any]], moves: Mapping[str, str]) -> list[dict[str, Any]]:
    """The *rows* below a moved archive, each under the archive's new name."""
    members = _members_by_archive(moves, (row[_PATH] for row in rows))
    paths = {
        path: moves[archive] + path[len(archive) :]
        for archive, found in members.items()
        for path in found
    }
    return [{**row, _PATH: paths[row[_PATH]]} for row in rows if row[_PATH] in paths]


def _new_source_id() -> str:
    """An id for a source no row holds: random, so a new file at a vacated path takes none."""
    return uuid.uuid4().hex


def _keyed(rows: Iterable[Mapping[str, Any]], key: str, ids: Mapping[str, str]) -> list[dict]:
    """*rows* with the id of the path under *key* in its place."""
    return [
        {**{name: value for name, value in row.items() if name != key}, _SOURCE_ID: ids[row[key]]}
        for row in rows
    ]


def _pathed(rows: Iterable[Mapping[str, Any]], key: str, paths: Mapping[str, str]) -> list[dict]:
    """*rows* with the path of their source id under *key*; a row of no source row drops."""
    return [
        {
            **{name: value for name, value in row.items() if name != _SOURCE_ID},
            key: paths[row[_SOURCE_ID]],
        }
        for row in rows
        if row[_SOURCE_ID] in paths
    ]


def _id_chunks(rows: list[dict[str, Any]]) -> list[SearchChunk]:
    """Search results of chunk *rows* that name their source by id until the search ends."""
    return [
        SearchChunk(
            **{name: value for name, value in row.items() if name != _SOURCE_ID},
            source=row[_SOURCE_ID],
        )
        for row in rows
    ]


def _source_record(row: Mapping[str, Any], locations: SourceLocations) -> SourceRecord:
    """The ``SourceRecord`` a ``_sources`` row shows: its shown name as ``filename``, no id."""
    shown = {name: value for name, value in row.items() if name not in (_SOURCE_ID, *_HELD_COLUMNS)}
    shown[_OLD_PATH] = locations.name(shown.pop(_PATH))
    return cast("SourceRecord", shown)


class _SourceWrite(NamedTuple):
    """One source row a write puts in ``_sources``, by the name and type of its source.

    With *fields* the row holds them; without, a found row keeps its own and a new
    row only names the source. A *record* holds the source out of a sync.
    """

    name: str
    source_type: SourceType
    fields: Mapping[str, Any] | None = None
    record: HeldRecord | None = None
    new_id: str | None = None


def _documents(names: Iterable[str]) -> Iterator[tuple[str, SourceType]]:
    """Each of *names* as the name of a document."""
    return ((name, SourceType.DOCUMENT) for name in names)


def _source_type_of(chunk_type: str) -> SourceType:
    """The type of the source a chunk of *chunk_type* is of: a wiki page, else a document."""
    return SourceType.WIKI if chunk_type == ChunkType.WIKI else SourceType.DOCUMENT


# The order a lookup by a name with no type tries the types in.
_READ_ORDER = (SourceType.WIKI, SourceType.IMPORTED, SourceType.DOCUMENT)


def _is_wiki(source_type: str) -> bool:
    """Whether *source_type* is a wiki page's, the one type no other source shares a row with."""
    return source_type == SourceType.WIKI


def _write_paths(write: _SourceWrite, locations: SourceLocations) -> tuple[str, ...]:
    """Where the row *write* finds can be: its key, then, for a non-wiki source, its other form."""
    if _is_wiki(write.source_type):
        return (write.name,)
    return tuple(
        dict.fromkeys(
            (
                locations.key(write.name, write.source_type),
                write.name,
                locations.location(write.name),
            )
        )
    )


def _found_rows(
    sources: LanceTable, writes: Sequence[_SourceWrite], locations: SourceLocations
) -> dict[str, dict[str, Any]]:
    """The row of its type class each of *writes* finds, by name; the key is looked at first."""
    paths = sorted({path for write in writes for path in _write_paths(write, locations)})
    held: dict[tuple[str, bool], dict[str, Any]] = {}
    for start in range(0, len(paths), _SOURCE_STAT_BATCH_ROWS):
        where = f"{_PATH} IN ({_in_list(paths[start : start + _SOURCE_STAT_BATCH_ROWS])})"
        for row in sources.search().where(where).limit(None).to_list():
            held[row[_PATH], _is_wiki(row["source_type"])] = row
    found: dict[str, dict[str, Any]] = {}
    for write in writes:
        wiki = _is_wiki(write.source_type)
        rows = [held[path, wiki] for path in _write_paths(write, locations) if (path, wiki) in held]
        if rows:
            found[write.name] = rows[0]
    return found


def _stat_write(backfill: SourceStatBackfill) -> _SourceWrite:
    """The write that gives a tracked source its verified stat; a record names no held-out one."""
    record, stat = backfill
    fields = {name: value for name, value in record.items() if name != _OLD_PATH}
    fields |= {
        "size_bytes": stat.size_bytes,
        "mtime_ns": stat.mtime_ns,
        "stat_captured_ns": stat.captured_ns,
    }
    return _SourceWrite(record["filename"], SourceType(record["source_type"]), fields)


def _at_or_below(path: str, roots: set[str]) -> bool:
    """Whether *path* is one of *roots* or lies below one, on whole path segments."""
    if path in roots:
        return True
    cut = path.find("/")
    while cut != -1:
        if path[:cut] in roots:
            return True
        cut = path.find("/", cut + 1)
    return False


def _members_by_archive(names: Iterable[str], filenames: Iterable[str]) -> dict[str, list[str]]:
    """The *filenames* under ``name/`` for each of *names*, in one pass over *filenames*."""
    wanted = set(names)
    members: dict[str, list[str]] = {}
    for filename in filenames:
        cut = filename.find("/")
        while cut != -1:
            if filename[:cut] in wanted:
                members.setdefault(filename[:cut], []).append(filename)
            cut = filename.find("/", cut + 1)
    return members


def _held_sources(sources: LanceTable, names: Iterable[str]) -> tuple[set[str], set[str]]:
    """The *names* that hold a row a move carries in the source table: kept, then removed."""
    wanted = sorted(names)
    kept: set[str] = set()
    removed: set[str] = set()
    for start in range(0, len(wanted), _SOURCE_STAT_BATCH_ROWS):
        quoted = _in_list(wanted[start : start + _SOURCE_STAT_BATCH_ROWS])
        query = sources.search().where(f"{_PATH} IN ({quoted}) AND {_MOVABLE}")
        for row in query.select([_PATH, _HELD_OUT]).limit(None).to_list():
            (removed if row[_HELD_OUT] == HeldOut.REMOVED else kept).add(row[_PATH])
    return kept, removed


def _source_rows(sources: LanceTable, names: list[str]) -> dict[str, dict[str, Any]]:
    """The row a move carries that the source table holds for each of *names*, by path."""
    if not names:
        return {}
    query = sources.search().where(f"{_PATH} IN ({_in_list(names)}) AND {_MOVABLE}")
    return {row[_PATH]: row for row in query.limit(None).to_list()}


def _listed_matching(search: str | None, locations: SourceLocations) -> str:
    """The listed source rows whose shown name or title holds *search*."""
    if not search:
        return _LISTED
    return f"{_LISTED} AND {locations.search_filter(search, _PATH, _TITLE_COLUMN)}"


def _md5(text: str) -> str:
    """The id the migration gives the source at *text*, as LanceDB's ``md5`` computes it."""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _wiki_id(key: str) -> str:
    """The id the migration gives the wiki page at *key*, as ``_WIKI_ID_SQL`` computes it."""
    return _md5(f"{key}\0{_WIKI_ID_SUFFIX}")


def _holds_document(row: Mapping[str, Any]) -> bool:
    """Whether a document source *row* holds an indexed document, as ``_LISTED`` reads it."""
    kind = row[_HELD_OUT]
    return kind is None or (kind == HeldOut.FAILED and row["file_hash"] is not None)


def _held_row(row: dict[str, Any], record: HeldRecord, *, holds_document: bool) -> dict[str, Any]:
    """*row* with *record*; a failure keeps the document the row holds, a removal holds none."""
    held = {
        **row,
        _HELD_OUT: record.kind.value,
        _HELD_HASH: record.file_hash,
        _HELD_REASON: record.reason,
    }
    if record.kind is HeldOut.FAILED and holds_document:
        return held
    return {**held, "file_hash": None, "chunk_count": 0}


def _keyed_by_path(db: LanceDBConnection) -> bool:
    """Whether a table of *db* still names a source by path, or ``_sources`` lacks a column."""
    sources = open_committed(db, SOURCES_TABLE)
    if sources is not None:
        added = (_SOURCE_ID, _PATH, *_SOURCE_STAT_COLUMNS, *_SOURCE_META_COLUMNS, _HELD_OUT)
        if not set(sources.schema.names).issuperset(added):
            return True
    for name, column in _PATH_KEY_COLUMNS:
        table = open_committed(db, name)
        if table is not None and column in table.schema.names:
            return True
    return False


def _upgrade_pending(db: LanceDBConnection) -> bool:
    """Whether *db* is keyed by path, or ``_meta`` still waits for the upgrade's stamp."""
    if _keyed_by_path(db):
        return True
    meta = open_committed(db, META_TABLE)
    return meta is not None and meta.count_rows(_UNSTAMPED) > 0


def _upgrade_steps(db: LanceDBConnection, names: list[str]) -> list[tuple[str, LanceTable]]:
    """The tables of *db* an upgrade visits: ``_sources``, then each still keyed by path."""
    tables = [(name, db.open_table(name)) for name in _UPGRADE_ORDER if name in names]
    return [
        (name, table)
        for name, table in tables
        if name == SOURCES_TABLE or _PATH_KEY_BY_TABLE[name] in table.schema.names
    ]


def _held_unmigrated(db: LanceDBConnection, data_root: Path) -> bool:
    """Whether ``_sources`` lacks a held-out record column, or a record file is at *data_root*."""
    if legacy_files(data_root):
        return True
    sources = open_committed(db, SOURCES_TABLE)
    return sources is not None and not {_HELD_HASH, _HELD_REASON} <= set(sources.schema.names)


def _add_held_columns(db: LanceDBConnection) -> None:
    """Give an existing ``_sources`` the held-out record columns it lacks. Lock held."""
    sources = open_committed(db, SOURCES_TABLE)
    if sources is None:
        return
    missing = [name for name in (_HELD_HASH, _HELD_REASON) if name not in sources.schema.names]
    if missing:
        sources.add_columns(dict.fromkeys(missing, _NULL_TEXT_SQL))


def _index_names(table: LanceTable) -> dict[str, list[str]]:
    """The columns of each index of *table*, by index name; none when the listing fails."""
    return {index.name: index.columns for index in _index_registry(table) or []}


def _key_column(table: LanceTable, column: str, type_column: str | None = None) -> pa.Table:
    """*column* of every row of *table*, with *type_column* beside it, read as Arrow."""
    columns = [column] if type_column is None else [column, type_column]
    return table.search().select(columns).limit(None).to_arrow()


def _wiki_mask(read: pa.Table, type_column: str | None, wiki: str) -> pa.ChunkedArray:
    """Whether *type_column* of each row of *read* is *wiki*; with no column, no row is."""
    if type_column is None:
        return pa.chunked_array([pa.repeat(pa.scalar(False), read.num_rows)])
    return pc.fill_null(pc.equal(read.column(type_column), wiki), False)


def _keys_missing_from(
    sources: LanceTable, table: LanceTable, column: str
) -> list[tuple[str, SourceType]]:
    """Each distinct key in *column* of *table* that no ``_sources`` row of its class holds.

    A wiki chunk names a wiki page; every other key names a document. Arrow
    compares the key columns, so no row becomes a Python object.
    """
    type_column = _CHUNK_TYPE if table.name == CHUNKS_TABLE else None
    read = _key_column(table, column, type_column)
    held = _key_column(sources, _PATH, _SOURCE_TYPE)
    pages = _wiki_mask(read, type_column, ChunkType.WIKI)
    wiki_rows = _wiki_mask(held, _SOURCE_TYPE, SourceType.WIKI)
    keys, paths = read.column(column), held.column(_PATH).cast(read.column(column).type)
    missing: list[tuple[str, SourceType]] = []
    for source_type, mine, theirs in (
        (SourceType.WIKI, pages, wiki_rows),
        (SourceType.DOCUMENT, pc.invert(pages), pc.invert(wiki_rows)),
    ):
        wanted = pc.filter(keys, mine)
        absent = pc.filter(wanted, pc.invert(pc.is_in(wanted, value_set=pc.filter(paths, theirs))))
        missing += [(key, source_type) for key in sorted(pc.unique(absent).to_pylist())]
    return missing


def _add_source_id(db: LanceDBConnection, name: str, column: str) -> None:
    """Give table *name* its ``source_id`` from the path in *column*, indexed as the path is."""
    from lancedb.index import BTree

    table = db.open_table(name)
    if _SOURCE_ID not in table.schema.names:
        table.add_columns({_SOURCE_ID: f"md5({column})"})
    if name == CHUNKS_TABLE:
        table.update(where=_WIKI_CHUNK, values_sql={_SOURCE_ID: _WIKI_ID_SQL.format(key=column)})
    indexes = _index_names(table)
    if [column] in indexes.values() and [_SOURCE_ID] not in indexes.values():
        table.create_index(_SOURCE_ID, config=BTree())


def _migrate_sources_table(table: LanceTable) -> None:
    """Give ``_sources`` an id per row, its path column, and every column it lacks. Lock held."""
    from lancedb.index import BTree

    if _SOURCE_ID not in table.schema.names:
        table.add_columns({_SOURCE_ID: f"md5({_OLD_PATH})"})
    if _OLD_PATH in table.schema.names:
        # lancedb's stub types each alteration as an iterable; it takes one dict.
        table.alter_columns({"path": _OLD_PATH, "rename": _PATH})  # type: ignore[dict-item]
    if f"{_OLD_PATH}_idx" in _index_names(table):
        # A key index keeps its name through the rename; the flush refreshes it by the new one.
        table.create_index(_PATH, config=BTree(), replace=True)
        table.create_index(_SOURCE_ID, config=BTree(), replace=True)
        table.drop_index(f"{_OLD_PATH}_idx")
    defaults = {name: f"CAST({SOURCE_STAT_UNKNOWN} AS BIGINT)" for name in _SOURCE_STAT_COLUMNS}
    defaults |= {name: _NULL_TEXT_SQL for name in (*_SOURCE_META_COLUMNS, _HELD_OUT)}
    missing = {name: sql for name, sql in defaults.items() if name not in table.schema.names}
    if missing:
        table.add_columns(missing)


def _label_keyed(locations: SourceLocations) -> str:
    """The document rows still named by a registered label in place of their location."""
    labels = " OR ".join(_under_sql(_PATH, label) for label in locations.roots)
    return f"{_DOCUMENT} AND ({labels})"


def _holds_label_keys(db: LanceDBConnection, locations: SourceLocations) -> bool:
    """Whether a source row of *db* is still named by a registered label."""
    sources = open_committed(db, SOURCES_TABLE)
    if sources is None or not locations.roots or _PATH not in sources.schema.names:
        return False
    return bool(sources.count_rows(_label_keyed(locations)))


def _pinned(db: LanceDBConnection, table: LanceTable) -> LanceTable:
    """*table* held at its version now, read while later commits go to *table*."""
    pinned: LanceTable = db.open_table(table.name)
    pinned.checkout(table.version)
    return pinned


def _rekey_prefix(
    db: LanceDBConnection, sources: LanceTable, where: str, old: str, new: str
) -> None:
    """Put each source row *where* selects at the place below *new* it has below *old*. Lock held.

    A row whose new place a row it does not move holds folds into that row. A
    selection of no row writes nothing. Rows stream in batches of
    ``_FILL_BATCH_ROWS``, so memory does not follow the row count.
    """
    if not sources.count_rows(where):
        return
    _fold_landings(db, sources, where, old, new)
    _rewrite_prefix(db, sources, where, old, new)


def _rewrite_prefix(
    db: LanceDBConnection, sources: LanceTable, where: str, old: str, new: str
) -> None:
    """Give each row *where* selects *new* in place of its head *old*: one update a batch.

    Each update names its rows by their address in the version read, which a row
    keeps until it is rewritten, so no update reads past its own rows.
    """
    query = _pinned(db, sources).search().where(where).select([_SOURCE_ID]).with_row_id(True)
    values = {_PATH: _swap_head_sql(_PATH, old, new)}
    for batch in query.limit(None).to_batches(_FILL_BATCH_ROWS):
        rows = ", ".join(map(str, batch.column(_ROW_ID).to_pylist()))
        sources.update(where=f"{_ROW_ID} IN ({rows})", values_sql=values)


def _fold_landings(
    db: LanceDBConnection, sources: LanceTable, where: str, old: str, new: str
) -> None:
    """Fold each row *where* selects into the non-wiki row not moving at its place below *new*.

    The citations of the folded row cite the row it lands on; its own rows go.
    """
    landed = f"NOT ({_WIKI}) AND {_under_sql(_PATH, new)} AND NOT ({where})"
    if not sources.count_rows(landed):
        return
    pairs = _fill_pairs(db, sources, where, landed, (old, new))
    found = pairs.search().where(f"{_MOVED_ID} IS NOT NULL").select([_MOVED_ID, _LANDING_ID])
    for batch in found.limit(None).to_batches(_FILL_BATCH_ROWS):
        moved, landing = batch.column(_MOVED_ID), batch.column(_LANDING_ID)
        _fold_batch(db, sources, moved.to_pylist(), landing.to_pylist())
    db.drop_table(_FILL_PAIRS_TABLE)


def _fill_pairs(
    db: LanceDBConnection, sources: LanceTable, where: str, landed: str, heads: tuple[str, str]
) -> LanceTable:
    """A scratch table of each *landed* row under its *old* path, with the row moving onto it."""
    from lancedb.index import BTree

    old, new = heads
    pairs: LanceTable = db.create_table(
        _FILL_PAIRS_TABLE, schema=_FILL_PAIRS_SCHEMA, mode="overwrite"
    )
    labelled: dict[str, str | Expr] = {
        _PATH: _swap_head_sql(_PATH, new, old),
        _LANDING_ID: _SOURCE_ID,
    }
    for batch in (
        sources.search().where(landed).select(labelled).limit(None).to_batches(_FILL_BATCH_ROWS)
    ):
        empty = pa.nulls(batch.num_rows, pa.utf8())
        pairs.add(
            pa.table(
                [batch.column(_PATH), batch.column(_LANDING_ID), empty], schema=_FILL_PAIRS_SCHEMA
            )
        )
    pairs.create_index(_PATH, config=BTree())
    moving = (
        _pinned(db, sources).search().where(where).select({_PATH: _PATH, _MOVED_ID: _SOURCE_ID})
    )
    for batch in moving.limit(None).to_batches(_FILL_BATCH_ROWS):
        pairs.merge_insert(_PATH).when_matched_update_all().execute(pa.Table.from_batches([batch]))
    return pairs


def _fold_batch(
    db: LanceDBConnection, sources: LanceTable, moved: list[str], landing: list[str]
) -> None:
    """Fold each of the ids *moved* into the id *landing* holds beside it. Lock held."""
    folded = f"{_SOURCE_ID} IN ({_in_list(moved)})"
    citations = open_committed(db, CITATIONS_TABLE)
    if citations is not None:
        _repoint_citations(citations, moved, landing)
    for name in _PER_SOURCE_TABLES:
        table = open_committed(db, name)
        if table is not None:
            table.delete(folded)
    _safe_delete_unlocked(sources, folded)


def _repoint_citations(citations: LanceTable, moved: list[str], landing: list[str]) -> None:
    """Make each citation of an id of *moved* cite the id of *landing* beside it. Lock held."""
    for start in range(0, len(moved), _CITATION_LOOKUP_IDS):
        keys = moved[start : start + _CITATION_LOOKUP_IDS]
        values = landing[start : start + _CITATION_LOOKUP_IDS]
        where = f"{_SOURCE_ID} IN ({_in_list(keys)})"
        citations.update(where=where, values_sql={_SOURCE_ID: _looked_up_sql(keys, values)})


def _looked_up_sql(keys: list[str], values: list[str]) -> str:
    """The SQL value of the item of *values* at the place of ``source_id`` in *keys*."""
    at = f"CAST(array_position(make_array({_in_list(keys)}), {_SOURCE_ID}) AS BIGINT)"
    return f"array_element(make_array({_in_list(values)}), {at})"


def _drop_fill_pairs(db: LanceDBConnection) -> None:
    """Drop the scratch table of a fill that was killed before its end."""
    if _FILL_PAIRS_TABLE in table_names(db):
        db.drop_table(_FILL_PAIRS_TABLE)


def _fill_locations(db: LanceDBConnection, locations: SourceLocations) -> None:
    """Give each source row a registered label names its location in place of that key. Lock held.

    A deeper root goes first, so a row an outer root keeps for a file of a root
    inside it folds into the inner root's row. Each batch is one commit, and a
    filled row matches no label, so a killed run is finished by the next.
    """
    sources = open_committed(db, SOURCES_TABLE)
    if sources is None or not _holds_label_keys(db, locations):
        return
    _drop_fill_pairs(db)
    for root, label in locations.deepest_first:
        _rekey_prefix(db, sources, f"{_DOCUMENT} AND ({_under_sql(_PATH, label)})", label, root)
    log.info("Stored where each source of the library at %s is", db.uri)


def _refresh_key_index_unlocked(table: LanceTable, column: str, flush_rows: int) -> None:
    """Rebuild the BTREE index on *column* once enough appended rows sit past it.

    Caller holds ``write_lock()``. A delete filtered on *column* probes the index
    and scans only the rows past it, so its cost follows the flush size and not
    the table size.
    """
    from lancedb.index import BTree

    stats = table.index_stats(f"{column}_idx")
    unindexed = table.count_rows() if stats is None else stats.num_unindexed_rows
    if unindexed >= _KEY_INDEX_STALE_FLUSHES * flush_rows:
        table.create_index(column, config=BTree(), replace=True)


def _holds_rows(table: LanceTable) -> bool:
    """Whether *table* holds a row; a table that cannot be counted is taken to hold one."""
    try:
        return bool(table.count_rows())
    except Exception:
        return True


def _get_distance(chunk: SearchChunk) -> float:
    """Extract distance as a sortable float (inf for None)."""
    return chunk.distance if chunk.distance is not None else float("inf")


def _count_within_threshold(sorted_results: list[SearchChunk], threshold: float) -> int:
    """Count results whose distance is within the given threshold."""
    for i, r in enumerate(sorted_results):
        if _get_distance(r) > threshold:
            return i
    return len(sorted_results)


class Store:
    """LanceDB vector store: wraps all DB operations with config-driven defaults."""

    def __init__(
        self, config: Config, on_upgrade: DetailedProgressCallback = noop_callback
    ) -> None:
        self._config = config
        self._on_upgrade = on_upgrade
        # The latest event of an upgrade that runs; None when none does.
        self._upgrade: LibraryUpgradeEvent | None = None
        # One open at a time: a caller that finds an open running waits for it.
        self._open_lock = threading.Lock()
        self._fts_ready: bool = False
        self._title_fts_ready: bool = False
        self._doc_prefix_warned: bool = False
        # Degradations the health endpoint reports; set where they are detected.
        self._fts_degraded: bool = False
        self._doc_prefix_mismatch: bool = False
        # Scalar indexes (source/chunk_type) are built at ingest; a serve-only
        # store builds them lazily from the search path.
        self._scalar_ready: bool = False
        # Tables whose failed key index refresh was already warned about.
        self._key_index_warned: set[str] = set()
        self._db: LanceDBConnection | None = None
        # Cache of {filename: ingested_at} rebuilt only when sources
        # mutate; callers (temporal filter) hit it per-query.
        self._source_ingested_cache: dict[str, str] | None = None
        # The source registry names are shown by, the config.toml state it was read at,
        # and how many blocks hold it fixed.
        self._registry = SourceLocations({})
        self._registry_stamp: tuple[int, int, int] | None = None
        self._registry_pins = 0

    def _fts_config(self) -> FTS:
        """Shared FTS index config: positionless (with_position=True overflows
        LanceDB's list encoding on optimize()) and stemmed for the configured
        corpus language."""
        from lancedb.index import FTS

        return FTS(with_position=False, language=self._config.fts_language)

    def _index_build_lock(self, blocking: bool) -> AbstractContextManager[None]:
        """Write lock for index builds: full budget from ingest, short from the read path."""
        return self._write_lock() if blocking else self._write_lock(_READ_LOCK_TIMEOUT)

    def _write_lock(self, timeout: float = LOCK_TIMEOUT) -> AbstractContextManager[None]:
        """Acquire the write lock keyed on *this* store's data directory.

        A per-instance ``Lilbee`` writes to its own ``lancedb_dir``; locking the
        global ``cfg`` dir instead would leave those writes uncoordinated across
        processes.
        """
        self.get_db()  # opens the library, and so migrates it, outside the lock
        return write_lock(self._config.lancedb_dir, timeout)

    def _locations(self) -> SourceLocations:
        """The rule sources are named by: the registry, read again when config.toml changes."""
        if not self._registry_pins:
            stamp = settings.overlay_stamp(self._config.data_root)
            if stamp != self._registry_stamp:
                self._registry_stamp = stamp
                settings.overlay_persisted_roots(self._config)
            self._take_registry()
        return self._registry

    def _take_registry(self) -> None:
        """Name sources by the registry the config holds."""
        if self._config.linked_roots != self._registry.roots:
            self._registry = SourceLocations(self._config.linked_roots)
            self._invalidate_source_cache()

    @contextmanager
    def registry_pinned(self) -> Iterator[None]:
        """Name sources by the registry the config holds now, whatever changes it in the block."""
        self._take_registry()
        self._registry_pins += 1
        try:
            yield
        finally:
            self._registry_pins -= 1

    def _invalidate_source_cache(self) -> None:
        """Drop the cached {filename: ingested_at} map."""
        self._source_ingested_cache = None

    def source_ingested_at_map(self) -> dict[str, str]:
        """Return {filename: ingested_at} for every source, cached until mutation.

        Best-effort: a reader racing a concurrent invalidation can store a
        pre-mutation snapshot. The only consumer (temporal query filter)
        treats a missing/stale entry as "do not filter," so staleness
        degrades ranking precision, never correctness.
        """
        if self._source_ingested_cache is not None:
            return self._source_ingested_cache
        mapping = {s["filename"]: s.get("ingested_at", "") for s in self.get_sources()}
        self._source_ingested_cache = mapping
        return mapping

    def _chunks_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(_SOURCE_ID, pa.utf8()),
                pa.field("content_type", pa.utf8()),
                pa.field("chunk_type", pa.utf8()),
                pa.field("page_start", pa.int32()),
                pa.field("page_end", pa.int32()),
                pa.field("line_start", pa.int32()),
                pa.field("line_end", pa.int32()),
                pa.field("chunk", pa.utf8()),
                pa.field("chunk_index", pa.int32()),
                pa.field(_TITLE_COLUMN, pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
            ]
        )

    def _chunks_table(self) -> LanceTable:
        """Open/create the chunks table, adding the title column to pre-title tables."""
        table = ensure_table(self.get_db(), CHUNKS_TABLE, self._chunks_schema())
        if _TITLE_COLUMN not in table.schema.names:
            table.add_columns({_TITLE_COLUMN: _NULL_TEXT_SQL})
            self._backfill_stem_titles_unlocked(table)
        return table

    def _backfill_stem_titles_unlocked(self, table: LanceTable) -> None:
        """Backfill filename-stem titles for pre-upgrade rows. Caller holds ``write_lock()``.

        Without this the title arm only matches documents ingested after the
        upgrade. Extracted (H1/EXIF) titles still need ``lilbee rebuild``;
        failure leaves NULLs, the pre-backfill behavior.
        """
        from lilbee.data.title import derive_title  # circular at module scope

        try:
            rows = table.search().select([_SOURCE_ID]).limit(None).to_list()
            sources = self.source_paths({r[_SOURCE_ID] for r in rows})
            filled = 0
            for source_id, source in sorted(sources.items(), key=lambda item: item[1]):
                title = derive_title(source)
                if not title:
                    continue
                table.update(where=f"{_SOURCE_ID} = '{source_id}'", values={_TITLE_COLUMN: title})
                filled += 1
            log.info(
                "Backfilled filename titles for %d of %d existing sources; run "
                "`lilbee rebuild` to derive titles from document content",
                filled,
                len(sources),
            )
        except Exception:
            log.warning(
                "Title backfill failed; pre-upgrade rows keep NULL titles until `lilbee rebuild`",
                exc_info=True,
            )

    def get_meta(self) -> StoreMeta | None:
        """Return the persisted store metadata row, or ``None`` if unset."""
        table = self.open_table(META_TABLE)
        if table is None:
            return None
        rows = table.search().limit(None).to_list()
        if not rows:
            return None
        # _meta is meant to hold one row, but a swallowed delete on rewrite could
        # leave a stale one behind; take the newest so identity reads stay
        # deterministic rather than returning an arbitrary row.
        row = max(rows, key=lambda r: r["updated_at"])
        return StoreMeta(
            embedding_model=row["embedding_model"],
            embedding_dim=int(row["embedding_dim"]),
            schema_version=int(row["schema_version"]),
            updated_at=row["updated_at"],
        )

    def _write_meta_unlocked(self, *, embedding_model: str, embedding_dim: int) -> None:
        """Overwrite the single ``_meta`` row with the supplied identity.

        Caller must hold ``write_lock()``. Args are passed explicitly rather than
        re-read from ``self._config`` so the caller can snapshot cfg at a coherent
        instant and not race with a concurrent ``set_embedding_model``.
        """
        db = self.get_db()
        table = ensure_table(db, META_TABLE, _meta_schema())
        _safe_delete_unlocked(table, META_DELETE_ALL_PREDICATE)
        table.add(
            [
                {
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "schema_version": META_SCHEMA_VERSION,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            ]
        )
        # A rebuild is the remedy the prefix warning names, so the next query
        # re-decides instead of logging a mismatch this write just resolved.
        self._doc_prefix_warned = False

    def _has_chunks(self) -> bool:
        """Return True when the chunks table exists and has at least one row."""
        chunks = self.open_table(CHUNKS_TABLE)
        return chunks is not None and chunks.count_rows() > 0

    def has_chunks(self) -> bool:
        """Public predicate: True iff the store currently holds at least one chunk."""
        return self._has_chunks()

    def initialize_meta_if_legacy(self) -> bool:
        """Pin a legacy store's identity to the current cfg if not already set.

        Returns ``True`` when a meta row was just written. No-op when meta already
        exists or no chunks are present. This is the path that converts a
        pre-upgrade store (chunks present, no ``_meta``) into a gated store. It
        snapshots cfg under the write lock to keep the recorded identity coherent
        with what the gate is comparing against.
        """
        if self.get_meta() is not None:
            return False
        if not self._has_chunks():
            return False
        embedding_model = self._config.embedding_model
        embedding_dim = self._config.embedding_dim
        with self._write_lock():
            # Re-check under the lock so two callers do not both warn-and-write.
            if self.get_meta() is not None:
                return False
            log.warning(
                "Legacy store has chunks but no _meta row. Initializing _meta from "
                "the current configuration (embedding_model=%s, embedding_dim=%d). "
                "If you changed embedding_model before upgrading, run `lilbee rebuild` "
                "to ensure the store is consistent.",
                embedding_model,
                embedding_dim,
            )
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)
            return True

    def index_mismatch(self) -> EmbeddingModelMismatchError | None:
        """The drift between the persisted embedding identity and cfg, or None.

        Pure check, no side effects, and the one verdict search, ingest, the
        embedding setter, sync and health all read. Migration of legacy stores
        (chunks present, no ``_meta``) is the caller's responsibility via
        ``initialize_meta_if_legacy``; rewriting a legacy bare-repo ``_meta`` row
        to the canonical full ref is the caller's responsibility via
        ``canonicalize_meta_if_legacy``. Safe to call from inside an existing
        ``write_lock()`` (no recursive lock attempt). cfg fields are snapshotted
        at entry so the comparison is coherent even if another thread mutates
        them mid-call.
        """
        current_model = self._config.embedding_model
        current_dim = self._config.embedding_dim
        meta = self.get_meta()
        if meta is None:
            return None
        if refs_compatible(
            meta["embedding_model"], current_model, meta["embedding_dim"], current_dim
        ):
            return None
        return EmbeddingModelMismatchError(
            persisted_model=meta["embedding_model"],
            persisted_dim=meta["embedding_dim"],
            current_model=current_model,
            current_dim=current_dim,
        )

    def _ensure_embedding_compat(self) -> None:
        """Raise when the persisted embedding identity drifts from cfg."""
        mismatch = self.index_mismatch()
        if mismatch is not None:
            raise mismatch

    def _doc_prefix_is_stale(self) -> bool:
        """Whether the stored documents predate the embedding family's document prefix."""
        meta = self.get_meta()
        if meta is None:
            return False
        profile = resolve_embedding_profile(self._config.embedding_model)
        return bool(profile.doc_prefix) and meta["schema_version"] < profile.doc_prefix_since

    def _warn_stale_doc_prefix(self) -> None:
        """Warn once when the embedding family's document prefix postdates this store.

        Queries would carry the family prefix while stored documents do not;
        the mismatch degrades quality silently until a rebuild re-embeds. The
        check reads the metadata row, which the query path cannot pay per search.
        """
        if self._doc_prefix_warned:
            return
        self._doc_prefix_warned = True
        if not self._doc_prefix_is_stale():
            return
        self._doc_prefix_mismatch = True
        log.warning(
            "This index predates '%s' document prefixes: queries are prefixed "
            "but stored documents are not. Run `lilbee rebuild` to re-embed.",
            self._config.embedding_model,
        )

    def _upgrade_warnings(self) -> list[HealthWarning]:
        """The warning of an upgrade that runs, or none."""
        upgrade = self._upgrade
        if upgrade is None:
            return []
        remedy = "Server requests answer 503 after 10 s. Try again when it ends."
        return [
            HealthWarning(code=WarningCode.LIBRARY_UPGRADING, message=upgrade.text, remedy=remedy)
        ]

    def _index_mismatch_warning(self) -> HealthWarning | None:
        """The stale-index degradation, from the same verdict search refuses on."""
        mismatch = self.index_mismatch()
        if mismatch is None:
            return None
        return HealthWarning(
            code=WarningCode.INDEX_EMBEDDING_MISMATCH,
            message=(
                f"The index was built with embedding model '{mismatch.persisted_model}' "
                f"({mismatch.persisted_dim} dims), but '{mismatch.current_model}' "
                f"({mismatch.current_dim} dims) is configured, so search refuses it."
            ),
            remedy=(
                f"Rebuild the index under '{mismatch.current_model}', or switch the "
                f"embedding model back to '{mismatch.persisted_model}'."
            ),
        )

    def health_warnings(self) -> list[HealthWarning]:
        """Retrieval degradations a client should know about before it searches.

        A library that is not open reports only an upgrade that runs: opening it here
        would wait for that upgrade.
        """
        if self._db is None:
            return self._upgrade_warnings()
        warnings: list[HealthWarning] = []
        if (stale := self._index_mismatch_warning()) is not None:
            warnings.append(stale)
        if self._fts_degraded:
            warnings.append(
                HealthWarning(
                    code=WarningCode.FTS_UNAVAILABLE,
                    message=(
                        "Keyword search is unavailable, so answers are drawn from "
                        "vector similarity alone and exact terms may be missed."
                    ),
                    remedy="Run `lilbee rebuild` to rebuild the search index.",
                )
            )
        table = self.open_table(CHUNKS_TABLE)
        indices = _index_registry(table) if table is not None else None
        if table is not None and indices is not None:
            dangling_scalar = _scalar_index_dangling(table, indices, self._config.lancedb_dir)
            if dangling_scalar:
                warnings.append(
                    HealthWarning(
                        code=WarningCode.SCALAR_INDEX_UNAVAILABLE,
                        message=(
                            "The scalar index on "
                            + ", ".join(dangling_scalar)
                            + " is missing its files, so filtered search is degraded."
                        ),
                        remedy="Run `lilbee rebuild` to rebuild the search index.",
                    )
                )
        if self._doc_prefix_mismatch and not self._doc_prefix_is_stale():
            # The remedy is `lilbee rebuild`, which usually runs in another
            # process, so the query path's cached verdict outlives the fix.
            self._doc_prefix_mismatch = False
        if self._doc_prefix_mismatch:
            warnings.append(
                HealthWarning(
                    code=WarningCode.EMBEDDING_PREFIX_MISMATCH,
                    message=(
                        f"This index predates '{self._config.embedding_model}' document "
                        "prefixes: queries are prefixed but stored documents are not, "
                        "so retrieval is less accurate."
                    ),
                    remedy="Run `lilbee rebuild` to re-embed the documents.",
                )
            )
        return warnings

    def assert_embedding_compatible(self) -> None:
        """Run the full embedding-identity gate (legacy init, canonicalize, check).

        Mirrors the gate ``search`` applies. Callers that write under a fresh
        embedder (import) use this to fail before any destructive work when the
        store was built by a different model.
        """
        self.initialize_meta_if_legacy()
        self.canonicalize_meta_if_legacy()
        self._ensure_embedding_compat()

    def _needs_canonical_meta_rewrite(
        self, meta: StoreMeta | None, current_model: str, current_dim: int
    ) -> bool:
        """True iff *meta* is the legacy form and refs-compatible with current cfg."""
        if meta is None or meta["embedding_model"] == current_model:
            return False
        return refs_compatible(
            meta["embedding_model"], current_model, meta["embedding_dim"], current_dim
        )

    def canonicalize_meta_if_legacy(self) -> bool:
        """Rewrite a legacy bare-repo ``_meta`` row to the canonical full ref.

        Pre-canonical lilbee persisted only ``<org>/<repo>`` in
        ``_meta.embedding_model``. The current code persists the full
        ``<org>/<repo>/<filename>.gguf``. When the two refer to the same
        model under :func:`refs_compatible` but differ as raw strings, the
        meta row is rewritten so the legacy name never surfaces. Returns
        ``True`` on write; ``False`` when missing, already canonical, or
        incompatible (the gate handles incompatibility).
        """
        current_model = self._config.embedding_model
        current_dim = self._config.embedding_dim
        if not self._needs_canonical_meta_rewrite(self.get_meta(), current_model, current_dim):
            return False
        with self._write_lock():
            meta = self.get_meta()  # re-read under the lock for racing callers
            if not self._needs_canonical_meta_rewrite(meta, current_model, current_dim):
                return False
            assert meta is not None  # filtered above  # noqa: S101
            log.info(
                "Migrating legacy embedding ref in store meta: %r -> %r",
                meta["embedding_model"],
                current_model,
            )
            self._write_meta_unlocked(embedding_model=current_model, embedding_dim=current_dim)
            return True

    def get_db(self) -> LanceDBConnection:
        """The open library; the first call opens it, and upgrades it when it is older."""
        db = self._db
        if db is None:
            with self._open_lock:
                db = self._open_held()
        return db

    @property
    def is_open(self) -> bool:
        """Whether the library is open, so a call needs no open or upgrade first."""
        return self._db is not None

    def open_in_background(self) -> None:
        """Open the library on a daemon thread; ``wait_for_open`` waits from this call on.

        An open that already runs is left to finish: its waiters wait for it.
        """
        if not self._open_lock.acquire(blocking=False):
            return
        threading.Thread(target=self._open_then_release, name="library-open", daemon=True).start()

    def wait_for_open(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for another thread to open the library.

        Return False when the open is still running at the deadline.
        """
        if not self._open_lock.acquire(timeout=timeout):
            return False
        self._open_lock.release()
        return True

    def _open_then_release(self) -> None:
        """Open the library, then free the open lock ``open_in_background`` took."""
        try:
            self._open_held()
        finally:
            self._open_lock.release()

    def _open_held(self) -> LanceDBConnection:
        """The library, opened unless it is open. Caller holds the open lock."""
        if self._db is not None:
            return self._db
        from lancedb.db import LanceDBConnection

        self._config.lancedb_dir.mkdir(parents=True, exist_ok=True)
        db = LanceDBConnection(
            str(self._config.lancedb_dir), read_consistency_interval=READ_CONSISTENCY_INTERVAL
        )
        if self._upgrade_needed(db):
            self._upgrade_library(db)
        self._db = db
        return db

    def _upgrade_needed(self, db: LanceDBConnection) -> bool:
        """Whether *db* is keyed by path, keeps the older record files, or keys a row by label."""
        return (
            _upgrade_pending(db)
            or _held_unmigrated(db, self._config.data_root)
            or _holds_label_keys(db, self._locations())
        )

    def _upgrade_library(self, db: LanceDBConnection) -> None:
        """Upgrade *db* under the write lock, waiting for it without a deadline, and report it."""
        started = time.monotonic()
        try:
            with ExitStack() as held:
                try:
                    held.enter_context(write_lock(self._config.lancedb_dir, timeout=0.0))
                except LockTimeoutError:
                    self._report_upgrade(LibraryUpgradeEvent(phase=UpgradePhase.WAITING))
                    held.enter_context(write_lock(self._config.lancedb_dir, timeout=None))
                # Another process can have upgraded it during the wait.
                if self._upgrade_needed(db):
                    self._upgrade_unlocked(db)
            elapsed = time.monotonic() - started
            self._report_upgrade(LibraryUpgradeEvent(phase=UpgradePhase.DONE, elapsed_s=elapsed))
        finally:
            self._upgrade = None

    def _upgrade_unlocked(self, db: LanceDBConnection) -> None:
        """Run each step of the upgrade that *db* still needs. Caller holds the write lock."""
        held = _held_unmigrated(db, self._config.data_root)
        if held:
            _add_held_columns(db)
        if _upgrade_pending(db):
            self._migrate_unlocked(db)
        if held:
            self._import_held_unlocked(db)
        _fill_locations(db, self._locations())

    def _report_upgrade(self, event: LibraryUpgradeEvent) -> None:
        """Record *event* as the upgrade's state, then pass it to the reporter."""
        self._upgrade = event
        self._on_upgrade(EventType.LIBRARY_UPGRADE, event)

    def _migrate_unlocked(self, db: LanceDBConnection) -> None:
        """Key every table by ``source_id``. Caller holds the write lock.

        Each step reads the table's columns, and the stamp reads ``_meta``, so a
        run that was killed is finished by the next one. An existing source takes
        the md5 of its path as its id. A store stamped 2 is stamped 3 at the end; a
        store stamped 1 predates the document prefix, and keeps 1 so that the
        stale-prefix warning still names it.
        """
        names = table_names(db)
        steps = _upgrade_steps(db, names)
        keyed = [(name, _PATH_KEY_BY_TABLE[name]) for name, _ in steps if name != SOURCES_TABLE]
        # LanceDB adds a column one fragment at a time: each table's runs on its own thread
        # from the start, and its path column goes once its every path has a source row.
        with ThreadPoolExecutor(len(_PATH_KEY_COLUMNS), _UPGRADE_THREAD_PREFIX) as pool:
            added = [pool.submit(_add_source_id, db, name, column) for name, column in keyed]
            for step, (name, table) in enumerate(steps, start=1):
                shown = name.strip("_").replace("_", " ")
                self._report_upgrade(
                    LibraryUpgradeEvent(
                        phase=UpgradePhase.TABLE,
                        table=shown,
                        step=step,
                        total_steps=len(steps),
                        rows=table.count_rows(),
                    )
                )
                if name == SOURCES_TABLE:
                    _migrate_sources_table(table)
                else:
                    self._hold_missing_paths(db, table, _PATH_KEY_BY_TABLE[name])
            for (name, column), done in zip(keyed, added, strict=True):
                done.result()
                db.open_table(name).drop_columns([column])
        if META_TABLE in names:
            db.open_table(META_TABLE).update(
                where="schema_version >= 2", values={"schema_version": META_SCHEMA_VERSION}
            )
        log.info("Keyed the library at %s by source id", self._config.lancedb_dir)

    def _import_held_unlocked(self, db: LanceDBConnection) -> None:
        """Move the held-out records of the older files into ``_sources``, then delete them.

        Gives each recorded name its record in one commit; a row that holds an
        indexed document takes no removal. Lock held. The files go last; an open
        killed before then imports the same records again onto the same rows.
        """
        sources = ensure_table(db, SOURCES_TABLE, _sources_schema())
        root = self._config.data_root
        writes = [
            _SourceWrite(name, SourceType.DOCUMENT, record=record)
            for name, record in read_legacy_records(root).items()
        ]
        self._put_sources_unlocked(sources, writes, keep_documents=True)
        remove_legacy_files(root)
        log.info("Moved the held-out file records of %s into the index", root)

    def _hold_missing_paths(self, db: LanceDBConnection, table: LanceTable, column: str) -> None:
        """Give each key in *column* of *table* no source row of its class holds a row. Lock held.

        A wiki page, or a removed source a citation names, gets one under the id the
        column's key and type give, so no key is lost when the column goes.
        """
        sources = ensure_table(db, SOURCES_TABLE, _sources_schema())
        writes = [
            _SourceWrite(key, kind, new_id=_wiki_id(key) if _is_wiki(kind) else _md5(key))
            for key, kind in _keys_missing_from(sources, table, column)
        ]
        self._put_sources_unlocked(sources, writes)

    def open_table(self, name: str) -> LanceTable | None:
        """Open a table if it exists, otherwise return None."""
        return open_committed(self.get_db(), name)

    def ensure_fts_index(self, *, blocking: bool = True) -> None:
        """Create the chunks FTS index, or run ``optimize()`` once it exists.

        ``optimize()`` folds newly added rows into the FTS index and also
        runs LanceDB's default compaction + version pruning (default prune
        window: 7 days). Work scales with recent deltas rather than total
        chunk count, so large corpora no longer pay the full
        ``create_index(config=FTS(), replace=True)`` rebuild cost on every sync.

        ``blocking=False`` (the search path) marks an existing index ready
        without the lock and skips maintenance when another process holds it
        or the filesystem cannot lock, so a long concurrent ingest cannot stall
        or fail a query. An unreadable
        index registry never counts as an existing index.
        """
        probe = self.open_table(CHUNKS_TABLE)
        if probe is None:
            return
        indices = _index_registry(probe)
        if indices is not None and _has_fts_index(indices):
            self._fts_ready = True
        try:
            with self._index_build_lock(blocking):
                self._ensure_fts_index_unlocked()
        except _UPKEEP_SKIPPED as exc:
            if blocking:
                raise
            log.debug("Skipped FTS index maintenance: %s", exc)

    def _ensure_fts_index_unlocked(self) -> None:
        """Body of ``ensure_fts_index``. Caller holds ``write_lock()``."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return
        try:
            indices = _index_registry(table)
            if indices is None:
                self._repair_fts_registry(table)
                return
            if _has_fts_index(indices):
                self._fts_ready = True
                try:
                    # One optimize folds new rows into every index on the table.
                    table.optimize()
                    log.debug("FTS index optimized on '%s'", CHUNKS_TABLE)
                    if _index_registry(table) is None:
                        # Suspected producer of an index without files; unconfirmed.
                        self._repair_fts_registry(table)
                        return
                except Exception as exc:
                    if _is_fts_position_overflow(exc):
                        # Positional indexes overflow on optimize(); rebuild
                        # positionless once.
                        self._rebuild_fts(table, "a positional index overflowed on optimize()")
                    else:
                        log.warning(
                            "FTS optimize() failed; the existing index still serves hybrid search",
                            exc_info=True,
                        )
            else:
                # Positionless: with_position=True overflows LanceDB's list
                # encoding on optimize(), and nothing issues phrase queries.
                table.create_index(_CHUNK_COLUMN, config=self._fts_config(), replace=False)
                self._fts_ready = True
                log.debug("FTS index created on '%s'", CHUNKS_TABLE)
            # Only the opt-in title arm needs the title index.
            if self._config.title_search:
                self._ensure_title_fts_unlocked(table)
        except Exception:
            log.debug("FTS index ensure failed (empty table?)", exc_info=True)

    def _ensure_title_fts_unlocked(self, table: LanceTable) -> None:
        """Create the title FTS index when the column exists. Caller holds ``write_lock()``.

        Failure never blocks the chunk index: the title arm feature-detects the
        index per query, so a store without it simply searches without titles.
        """
        indices = _index_registry(table)
        if indices is None:
            self._repair_fts_registry(table)
            return
        if _TITLE_COLUMN not in table.schema.names or _has_fts_index(indices, _TITLE_COLUMN):
            self._title_fts_ready = _has_fts_index(indices, _TITLE_COLUMN)
            return
        try:
            # Positionless for the same reason as the chunk index.
            table.create_index(_TITLE_COLUMN, config=self._fts_config(), replace=False)
            self._title_fts_ready = True
            log.debug("Title FTS index created on '%s'", CHUNKS_TABLE)
        except Exception:
            # Only reached with title_search enabled, so a silent failure means
            # the user's opted-in title arm quietly does nothing. Warn, don't hide.
            log.warning(
                "Title FTS index creation failed; the title-search arm will "
                "contribute nothing until it can be built",
                exc_info=True,
            )

    def ensure_title_fts_index(self, *, blocking: bool = True) -> None:
        """Build the title FTS index for a title_search toggle after startup.

        Without this, a process that latched ``_fts_ready`` before the toggle
        never builds the index and the title arm stays silently dead until the
        next ingest or restart.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return
        indices = _index_registry(table)
        if indices is not None and _has_fts_index(indices, _TITLE_COLUMN):
            self._title_fts_ready = True
            return
        try:
            with self._index_build_lock(blocking):
                self._ensure_title_fts_unlocked(table)
        except _UPKEEP_SKIPPED as exc:
            if blocking:
                raise
            log.debug("Skipped title FTS build: %s", exc)

    def _rebuild_fts(self, table: LanceTable, reason: str) -> None:
        """Replace the FTS indexes with fresh positionless ones. Caller holds the lock.

        The one-shot remediation for an index built ``with_position=True`` that
        overflows on every ``optimize()``. The title index is rebuilt too when
        the title arm is enabled.
        """
        try:
            table.create_index(_CHUNK_COLUMN, config=self._fts_config(), replace=True)
            if self._config.title_search and _TITLE_COLUMN in table.schema.names:
                table.create_index(_TITLE_COLUMN, config=self._fts_config(), replace=True)
            log.warning("Rebuilt the FTS index because %s", reason)
        except Exception:
            log.warning(
                "Positionless FTS rebuild failed; the existing index still serves",
                exc_info=True,
            )

    def _repair_fts_registry(self, table: LanceTable) -> list[IndexConfig] | None:
        """Rebuild the chunks FTS indexes after the index listing raised. Caller holds the lock.

        lancedb opens each FTS index's files while it lists the indexes, so an
        FTS index without its files makes the listing raise; scalar and vector
        indexes do not. Each FTS index (chunk text, plus title when the title
        arm is on) is rebuilt with ``replace=True`` in its own try. The listing
        is then read once more; the directory check on a readable listing
        decides whether a scalar or vector index needs its own rebuild.

        Returns the listing after the rebuild, ``None`` (logged once) when it
        is still unreadable.
        """
        columns = [_CHUNK_COLUMN]
        if self._config.title_search and _TITLE_COLUMN in table.schema.names:
            columns.append(_TITLE_COLUMN)
        for column in columns:
            try:
                table.create_index(column, config=self._fts_config(), replace=True)
            except Exception:
                log.warning(
                    "Could not rebuild the FTS index on '%s.%s'", table.name, column, exc_info=True
                )
                continue
            log.warning(
                "Rebuilt the FTS index on '%s.%s' because the index listing is unreadable",
                table.name,
                column,
            )
            if column == _CHUNK_COLUMN:
                self._fts_ready = True
            else:
                self._title_fts_ready = True
        indices = _index_registry(table)
        if indices is None:
            log.warning(
                "The index listing on '%s' is still unreadable after the FTS rebuild; "
                "run `lilbee rebuild` to rebuild the search index",
                table.name,
            )
        return indices

    # Tables and (column, index_type) pairs the query path filters by.
    # chunk_concepts serves the concept boost (ConceptGraph._chunk_concepts_from);
    # without its index every boosted query full-scans the table.
    _SCALAR_TARGETS: tuple[tuple[str, tuple[tuple[str, ScalarIndexType], ...]], ...] = (
        (CHUNKS_TABLE, ((_SOURCE_ID, "BTREE"), ("chunk_type", "BITMAP"))),
        (CHUNK_CONCEPTS_TABLE, ((_SOURCE_ID, "BTREE"),)),
    )

    def ensure_scalar_indexes(self, *, blocking: bool = True) -> None:
        """Build scalar indexes on the columns lilbee filters by.

        ``source`` and ``chunk_type`` predicates run as prefilters (LanceDB's
        default), but without an index each is a full-table scan. Readiness
        latches only when every target table exists and is covered, so a table
        created later (chunk_concepts under serve ordering) still gets its
        index on a following call. A table whose scalar index lost its files
        counts as pending, so the build step replaces it. A table whose index
        listing is unreadable keeps readiness unlatched until the FTS path
        repairs the listing. The lock is taken only
        when there is something to build; ``blocking=False`` (the search path)
        skips the build when another process holds it or the filesystem cannot
        lock, instead of stalling or failing the query.
        """
        pending = []
        complete = True
        for name, columns in self._SCALAR_TARGETS:
            table = self.open_table(name)
            if table is None:
                complete = False
                continue
            indices = _index_registry(table)
            if indices is None:
                # The FTS path repairs an unreadable listing; check again then.
                complete = False
                continue
            names = table.schema.names
            dangling = _scalar_index_dangling(table, indices, self._config.lancedb_dir)
            needs_build = any(
                c in names and (not _has_scalar_index(indices, c) or c in dangling)
                for c, _ in columns
            )
            if needs_build:
                pending.append((name, columns))
        if not pending:
            self._scalar_ready = complete
            return
        try:
            with self._index_build_lock(blocking):
                for name, columns in pending:
                    self._ensure_scalar_index_on(name, columns)
            self._scalar_ready = complete
        except _UPKEEP_SKIPPED as exc:
            if blocking:
                raise
            log.debug("Skipped scalar index build: %s", exc)

    def _ensure_scalar_index_on(
        self, table_name: str, columns: tuple[tuple[str, ScalarIndexType], ...]
    ) -> None:
        """Build the given (column, index_type) scalar indexes on *table_name*.

        Caller holds ``write_lock()``. A registered index whose files are gone
        is replaced; nothing is built while the index listing is unreadable.
        Each column gets its own try so one failure does not skip the rest; a
        failure on a populated table warns (the prefilter speedup is silently
        lost) while an empty table's is debug.
        """
        table = self.open_table(table_name)
        if table is None:
            return
        indices = _index_registry(table)
        if indices is None:
            return
        names = table.schema.names
        fail_level = logging.WARNING if table.count_rows() > 0 else logging.DEBUG
        dangling = _scalar_index_dangling(table, indices, self._config.lancedb_dir)
        for column, index_type in columns:
            if column not in names or (
                _has_scalar_index(indices, column) and column not in dangling
            ):
                continue
            try:
                table.create_scalar_index(column, index_type=index_type, replace=column in dangling)
                log.debug("Scalar (%s) index created on '%s.%s'", index_type, table_name, column)
            except Exception:
                log.log(
                    fail_level,
                    "Scalar index create failed on '%s.%s'",
                    table_name,
                    column,
                    exc_info=True,
                )

    def _create_vector_index(self, table: LanceTable) -> None:
        """Build the IVF_PQ index on the vector column, replacing any existing one."""
        from lancedb.index import IvfPq

        table.create_index("vector", config=IvfPq(distance_type=_VECTOR_METRIC), replace=True)

    def ensure_vector_index(self, *, force: bool = False) -> bool:
        """Build or refresh the ANN vector index when the corpus is large enough.

        Below ``cfg.ann_index_threshold`` (or when it is 0) the store keeps exact
        flat search, which is faster and exact for small vaults and is all a
        laptop needs. Once an index exists, ``optimize()`` folds new rows in.
        Pass ``force=True`` to build regardless of the threshold (publish flow).
        An unreadable index listing gets its FTS repair first, and a registered
        index whose files are gone is rebuilt instead of optimized. Returns
        True when an index was created, refreshed, or rebuilt.
        """
        threshold = self._config.ann_index_threshold
        with self._write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is None:
                return False
            indices = _index_registry(table)
            if indices is None:
                indices = self._repair_fts_registry(table)
                if indices is None:
                    return False
            if _vector_index_dangling(table, indices, self._config.lancedb_dir):
                return self._rebuild_dangling_vector_index(table)
            if _has_vector_index(indices):
                table.optimize()
                log.debug("Vector index optimized on '%s'", CHUNKS_TABLE)
                if _index_registry(table) is None:
                    # Repair in the same step when the prune dropped FTS files.
                    self._repair_fts_registry(table)
                return True
            if not force and (threshold <= 0 or table.count_rows() < threshold):
                return False
            return self._build_vector_index(table)

    def _rebuild_dangling_vector_index(self, table: LanceTable) -> bool:
        """Replace a registered vector index whose files are gone. Caller holds the lock."""
        try:
            self._create_vector_index(table)
        except Exception:
            log.warning(
                "Could not rebuild the vector index on '%s' whose files are missing",
                CHUNKS_TABLE,
                exc_info=True,
            )
            return False
        log.warning("Rebuilt the vector index on '%s' because its files are missing", CHUNKS_TABLE)
        return True

    def _build_vector_index(self, table: LanceTable) -> bool:
        """Build the first vector index, warning on failure. Caller holds the lock."""
        try:
            self._create_vector_index(table)
            log.info("Vector ANN index created on '%s'", CHUNKS_TABLE)
            return True
        except Exception:
            log.warning(
                "Vector ANN index build failed on '%s' at %d rows; search falls back "
                "to exact flat scan, which is slow at this scale. Free up memory/disk "
                "and re-run to rebuild the index.",
                CHUNKS_TABLE,
                table.count_rows(),
                exc_info=True,
            )
            return False

    def _add_chunks_unlocked(self, records: list[dict]) -> int:
        """Add chunk records and return the count. Caller must hold ``write_lock()``."""
        embedding_model = self._config.embedding_model
        embedding_dim = self._config.embedding_dim
        self._ensure_embedding_compat()
        self._fts_ready = False
        self._scalar_ready = False
        if not records:
            return 0
        _check_vector_dims(records, embedding_dim)
        ids = self._ensure_ids_unlocked(
            (record["source"], _source_type_of(record["chunk_type"])) for record in records
        )
        table = self._chunks_table()
        table.add(_keyed(records, "source", ids))
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)
        return len(records)

    def add_chunks(self, records: list[dict]) -> int:
        """Add chunk records to the store. Returns count added.

        Raises ``EmbeddingModelMismatchError`` if the persisted ``_meta`` row was
        written under a different embedding model than the current ``cfg``. On the
        first write to a fresh store, ``_meta`` is initialized from the current cfg.

        The gate runs inside the write lock and uses a single cfg snapshot so a
        concurrent ``set_embedding_model`` cannot slip a write in past a stale
        compatibility check.
        """
        with self._write_lock():
            return self._add_chunks_unlocked(records)

    def replace_chunks(self, source: str, chunk_type: ChunkType, records: list[dict]) -> int:
        """Replace *source*'s chunk rows of *chunk_type* with *records* under one write lock.

        Same compatibility and dimension gates as :meth:`add_chunks`, both run
        before the delete so a rejected write leaves the existing rows in place.
        The lock serializes writers; delete and add are separate commits, so a
        concurrent reader can briefly see the rows absent, and callers retry on
        a crash between the two. A delete failure propagates, following the same
        rule as :meth:`_delete_rows_of_unlocked`: swallowed, the caller would
        read a success-shaped result over rows still describing the old body.
        Returns the count added.
        """
        with self._write_lock():
            self._ensure_embedding_compat()
            _check_vector_dims(records, self._config.embedding_dim)
            source_id = self._ensure_ids_unlocked([(source, _source_type_of(chunk_type))])[source]
            table = self._chunks_table()
            table.delete(f"{_SOURCE_ID} = '{source_id}' AND {_chunk_type_predicate(chunk_type)}")
            return self._add_chunks_unlocked(records)

    def bm25_probe(
        self, query_text: str, top_k: int = 5, chunk_type: ChunkType | None = None
    ) -> list[SearchChunk]:
        """Quick BM25-only search for confidence checking. Returns up to top_k results.

        When *chunk_type* is set, only chunks of that type ("raw" or "wiki") are returned.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        if not self._fts_ready:
            self.ensure_fts_index(blocking=False)
        if not self._fts_ready:
            return []
        try:
            results = self._with_paths(
                _id_chunks(_lexical_rows(table, query_text, top_k, chunk_type))
            )
            norms = normalized_bm25([r.bm25_score or 0.0 for r in results])
            return [
                r.model_copy(update={"score": norm}) for r, norm in zip(results, norms, strict=True)
            ]
        except Exception:
            log.debug("BM25 probe failed", exc_info=True)
            return []

    def search(
        self,
        query_vector: Vector,
        top_k: int | None = None,
        max_distance: float | None = None,
        query_text: str | None = None,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        """Search for similar chunks. Hybrid when FTS available, else vector-only.

        Results with distance > max_distance are filtered out (vector-only path).
        Pass max_distance=0 to disable filtering.
        When *chunk_type* is set, only chunks of that type ("raw" or "wiki") are returned.

        Raises ``EmbeddingModelMismatchError`` if the persisted ``_meta`` row was
        written under a different embedding model than the current ``cfg``.
        """
        if top_k is None:
            top_k = self._config.top_k
        if max_distance is None:
            max_distance = self._config.max_distance
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        self.initialize_meta_if_legacy()
        self.canonicalize_meta_if_legacy()
        self._ensure_embedding_compat()
        self._warn_stale_doc_prefix()

        if not self._scalar_ready:
            # A serve-only store never ran ingest, where scalar indexes are
            # built; without them the source/chunk_type prefilters full-scan.
            self.ensure_scalar_indexes(blocking=False)
            # A repair replaces indexes; the handle must see the new versions.
            table.checkout_latest()

        if query_text:
            hits = self._keyword_arm(
                table, query_text, query_vector, top_k, max_distance, chunk_type
            )
            if hits is not None:
                return self._with_paths(hits)

        try:
            rows = self._vector_arm(
                table, query_vector, top_k * self._config.candidate_multiplier, chunk_type
            )
        except Exception:
            # A dead scalar index fails the prefilter; the next query re-verifies.
            self._scalar_ready = False
            raise
        log.debug(
            "Vector search: query=%r, candidates=%d, max_distance=%.2f",
            query_text or "vector-only",
            len(rows),
            max_distance,
        )
        if rows:
            log.debug("Top 5 distances: %s", [r.distance for r in rows[:5]])
        results = self._filter_and_rerank(rows, query_vector, top_k, max_distance)
        return [
            r.model_copy(
                update={"score": vector_similarity(r.distance) if r.distance is not None else 0.0}
            )
            for r in self._with_paths(results)
        ]

    def _keyword_arm(
        self,
        table: LanceTable,
        query_text: str,
        query_vector: Vector,
        top_k: int,
        max_distance: float,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk] | None:
        """Hybrid results, or None when keyword search cannot serve this query and
        the caller must fall back to vector-only.

        Owns the keyword-search health the endpoint reports. An index that will
        not build and an index that fails mid-query drop every query to vector
        recall alike, so both stamp the flag, and a working index clears it.
        """
        if not self._fts_ready:
            self.ensure_fts_index(blocking=False)
            table.checkout_latest()
        if self._config.title_search and not self._title_fts_ready:
            self.ensure_title_fts_index(blocking=False)
        if not self._fts_ready:
            # An empty corpus has nothing to index and nothing to degrade.
            self._fts_degraded = self._has_chunks()
            return None
        try:
            hits = self._hybrid_search(
                table, query_text, query_vector, top_k, max_distance, chunk_type
            )
        except Exception:
            # Falling back changes recall characteristics for the query;
            # a corpus-wide FTS breakage must not present as silence.
            self._fts_degraded = True
            # A dead scalar index fails the same way; the next query re-verifies.
            self._scalar_ready = False
            log.warning("Hybrid search failed, falling back to vector-only", exc_info=True)
            return None
        self._fts_degraded = False
        return hits

    def _vector_arm(
        self,
        table: LanceTable,
        query_vector: Vector,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """Vector-arm candidates with the ANN recall recovery applied.

        When ``chunk_type`` is set, the predicate is pushed into the query so
        the limit applies *after* the type filter; post-filtering would
        silently starve wiki-only queries whose matches live past the window.
        """
        query = table.search(query_vector).distance_type(_VECTOR_METRIC).limit(limit)
        indices = _index_registry(table)
        if indices is None or _has_vector_index(indices):
            # IVF_PQ is lossy; probe more partitions and refine against full
            # vectors so recall stays close to the exact flat scan. An unreadable
            # registry may hide the index, and the tuning is harmless on a flat scan.
            query = query.nprobes(_ann_nprobes(table.count_rows()))
            query = query.refine_factor(_ANN_REFINE_FACTOR)
        if chunk_type:
            query = query.where(_chunk_type_predicate(chunk_type))
        return _id_chunks(query.to_list())

    def _with_paths(self, results: list[SearchChunk]) -> list[SearchChunk]:
        """*results* with the path of each source in place of its id, in one lookup."""
        paths = self.source_paths({r.source for r in results})
        return [
            r.model_copy(update={"source": paths[r.source]}) for r in results if r.source in paths
        ]

    def _fts_arm(
        self,
        table: LanceTable,
        query_text: str,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """BM25-arm candidates over the chunk text."""
        return _id_chunks(_lexical_rows(table, query_text, limit, chunk_type))

    def _title_arm(
        self,
        table: LanceTable,
        query_text: str,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """One BM25 row per document whose title matches, in title-relevance order.

        Every chunk of a document carries the same title, so all of its chunks
        tie on BM25 and a plain ``limit`` would return an arbitrary tie-ordered
        subset of a single document. Instead this over-fetches, collapses each
        source to one deterministic representative (its first chunk), and returns
        the top *limit* documents ordered by title score -- so "a query naming a
        document by title surfaces its chunks" holds as one stable row per doc.

        Empty when the store predates the title column or its FTS index (old
        indexes keep working), when the index registry is unreadable, and on
        any query-time failure: the optional
        title arm must never take down the healthy chunk arm, so its failure
        degrades to no-titles, mirroring ``bm25_probe``.
        """
        indices = _index_registry(table)
        if indices is None or not _has_fts_index(indices, _TITLE_COLUMN):
            return []
        # Every chunk of one document ties on title BM25, so a fixed window can
        # fill up with a single long document's chunks and starve every other
        # title-matching document. Widen the fetch until enough distinct
        # documents surface, the matches run out, or the ceiling is hit.
        fetch = max(limit * _TITLE_FETCH_FACTOR, _TITLE_MIN_FETCH)
        while True:
            try:
                hits = _lexical_rows(table, query_text, fetch, chunk_type, column=_TITLE_COLUMN)
            except Exception:
                log.debug("Title arm search failed; contributing no title rows", exc_info=True)
                return []
            rows = _id_chunks(hits)
            best: dict[str, SearchChunk] = {}
            for row in rows:
                seen = best.get(row.source)
                if seen is None or row.chunk_index < seen.chunk_index:
                    best[row.source] = row
            if len(best) >= limit or len(hits) < fetch or fetch >= _TITLE_FETCH_CEILING:
                break
            fetch = min(fetch * 4, _TITLE_FETCH_CEILING)
        ordered = sorted(best.values(), key=lambda r: (-(r.bm25_score or 0.0), r.source))
        return ordered[:limit]

    def _hybrid_search(
        self,
        table: LanceTable,
        query_text: str,
        query_vector: Vector,
        top_k: int,
        max_distance: float,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        """Multi-arm retrieval fused by weighted reciprocal rank; the fused ordering is final.

        A vector arm and a chunk-BM25 arm always run; a title-BM25 arm joins
        when ``cfg.title_search`` is on. Each row's fused score is the
        weight-normalized sum of its arm contributions: the vector arm has
        weight 1.0, the lexical arm ``cfg.lexical_fusion_weight`` (scaled per
        query when ``cfg.adaptive_fusion`` is on), the title arm
        ``cfg.title_search_weight``. So a row a single peaked arm is certain
        about scores that arm's share of the total weight, not a fixed 0.5.

        Each arm fetches exactly ``top_k`` rows. Deeper pools measurably hurt
        rank fusion by flooding the fused top-k with both-arm mediocrity and
        burying single-arm certainty (lexical identifier hits above all). No
        MMR runs here: lexical passages are often mutually similar, which MMR
        penalizes, trading relevant hits for off-topic neighbors.

        Title rows carry ``bm25_score``, so a title match counts as lexical
        support for the distance exemption like any other lexical hit.
        """
        title_rows: list[SearchChunk] = []
        if self._config.title_search:
            title_rows = self._title_arm(table, query_text, top_k, chunk_type)
        vector_rows = self._vector_arm(table, query_vector, top_k, chunk_type)
        base_lexical_weight = self._config.lexical_fusion_weight
        base_title_weight = self._config.title_search_weight
        lexical_weight = base_lexical_weight
        title_weight = base_title_weight
        if self._config.adaptive_fusion:
            # Quiet the lexical arms per query by vector confidence. The title
            # arm is lexical too, so the same factor scales it.
            scale = adaptive_weight_scale(vector_rows, self._config.adaptive_fusion_margin)
            lexical_weight = base_lexical_weight * scale
            title_weight = base_title_weight * scale
        fused = fuse_arms(
            vector_rows,
            self._fts_arm(table, query_text, top_k, chunk_type),
            title_rows,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
        )
        fused = _drop_unsupported_far_rows(fused, max_distance)
        return fused[:top_k]

    def _filter_and_rerank(
        self,
        results: list[SearchChunk],
        query_vector: Vector,
        top_k: int,
        max_distance: float,
    ) -> list[SearchChunk]:
        """Apply the configured distance filter, then MMR-rerank down to top_k."""
        if max_distance > 0:
            before = len(results)
            if self._config.adaptive_threshold:
                results = self._adaptive_filter(results, top_k, max_distance)
                filter_name = "adaptive"
            else:
                results = self._fixed_filter(results, max_distance)
                filter_name = "fixed"
            log.debug(
                "After %s filter: %d/%d results, threshold=%.2f",
                filter_name,
                len(results),
                before,
                max_distance,
            )
        if len(results) > top_k:
            results = mmr_rerank(query_vector, results, top_k, self._config.mmr_lambda)
        return results

    def _adaptive_filter(
        self, results: list[SearchChunk], top_k: int, initial_threshold: float
    ) -> list[SearchChunk]:
        """Widen cosine distance threshold when too few results.
        Inspired by grantflow's (grantflow-ai/grantflow) adaptive retrieval
        pattern which widens thresholds on recursive retry. Step size and
        cap are configurable via ``self._config.adaptive_threshold_step``.

        Pre-sorts results by distance for a single-pass cutoff search.
        Step size is ``self._config.adaptive_threshold_step`` (default 0.2).
        """
        cap = max(initial_threshold, _MAX_THRESHOLD)
        step = self._config.adaptive_threshold_step

        sorted_results = sorted(results, key=_get_distance)

        threshold = initial_threshold
        for _ in range(_MAX_FILTER_ITERATIONS):
            if threshold > cap:
                break
            cutoff = _count_within_threshold(sorted_results, threshold)
            if cutoff >= top_k:
                return sorted_results[:cutoff]
            threshold += step
        # Final pass at cap
        cutoff = _count_within_threshold(sorted_results, cap)
        return sorted_results[:cutoff]

    def _fixed_filter(self, results: list[SearchChunk], threshold: float) -> list[SearchChunk]:
        """Simple fixed threshold filter - keep only results within distance threshold."""
        return [r for r in results if _get_distance(r) <= threshold]

    def add_entities(self, records: list[dict]) -> int:
        """Append typed entity rows; creates the table on first write.

        Additive to the store: existing tables and schemas are untouched, and
        stores without this table behave as if nothing was ever extracted.
        """
        if not records:
            return 0
        from lilbee.retrieval.entities.schema import _entities_schema

        with self._write_lock():
            ids = self._ensure_ids_unlocked(_documents(record["source"] for record in records))
            table = ensure_table(self.get_db(), ENTITIES_TABLE, _entities_schema())
            table.add(_keyed(records, "source", ids))
            return len(records)

    def entity_schema_state(self) -> EntitySchemaState | None:
        """The persisted entity schema row, or ``None`` when never induced.

        The schema is machine state induced from the corpus and lives inside
        the index, so it travels with the data.
        """
        table = self.open_table(ENTITY_SCHEMA_TABLE)
        if table is None:
            return None
        rows = table.search().limit(None).to_list()
        if not rows:
            return None
        # One row by contract; take the newest if a rewrite ever left a stale one.
        row = max(rows, key=lambda r: r["updated_at"])
        return EntitySchemaState(
            schema_json=str(row["schema_json"]),
            applied=bool(row["applied"]),
            source_count=int(row["source_count"]),
            updated_at=str(row["updated_at"]),
        )

    def save_entity_schema(self, schema_json: str, *, applied: bool, source_count: int) -> None:
        """Overwrite the single persisted entity schema row."""
        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, ENTITY_SCHEMA_TABLE, _entity_schema_state_schema())
            _safe_delete_unlocked(table, ENTITY_SCHEMA_DELETE_ALL_PREDICATE)
            table.add(
                [
                    {
                        "schema_json": schema_json,
                        "applied": applied,
                        "source_count": source_count,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                ]
            )

    def mark_entity_schema_applied(self) -> None:
        """Record that a full extraction pass completed under the stored schema."""
        state = self.entity_schema_state()
        if state is None:
            return
        self.save_entity_schema(
            state["schema_json"], applied=True, source_count=state["source_count"]
        )

    def entity_value_counts(self, entity_type: str) -> tuple[int, int]:
        """(mentions, distinct normalized values) for one entity type.

        Full scan by design: a count is a corpus property. Streaming batches
        keep memory flat at any corpus size.
        """
        table = self.open_table(ENTITIES_TABLE)
        if table is None:
            return 0, 0
        mentions = 0
        values: set[str] = set()
        arrow = table.to_arrow().select(["type", "normalized_value"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            types = batch.column("type").to_pylist()
            vals = batch.column("normalized_value").to_pylist()
            for t_, v in zip(types, vals, strict=True):
                if t_ == entity_type:
                    mentions += 1
                    values.add(v)
        return mentions, len(values)

    def entity_association_counts(self, counted: str, grouped_by: str) -> dict[str, int]:
        """Distinct *counted*-type values co-occurring with each *grouped_by* value.

        Co-occurrence is per chunk: two entities extracted from the same
        ``(source, chunk_index)`` are associated. This is the GROUP BY that
        answers "how many X is each Y associated with".
        """
        table = self.open_table(ENTITIES_TABLE)
        if table is None:
            return {}
        per_chunk: dict[tuple[str, int], tuple[set[str], set[str]]] = {}
        arrow = table.to_arrow().select(["type", "normalized_value", _SOURCE_ID, "chunk_index"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            rows = zip(
                batch.column("type").to_pylist(),
                batch.column("normalized_value").to_pylist(),
                batch.column(_SOURCE_ID).to_pylist(),
                batch.column("chunk_index").to_pylist(),
                strict=True,
            )
            for t_, v, src, idx in rows:
                if t_ not in (counted, grouped_by):
                    continue
                counted_vals, group_vals = per_chunk.setdefault((src, idx), (set(), set()))
                (counted_vals if t_ == counted else group_vals).add(v)
        associations: dict[str, set[str]] = {}
        for counted_vals, group_vals in per_chunk.values():
            for group_value in group_vals:
                associations.setdefault(group_value, set()).update(counted_vals)
        return {k: len(v) for k, v in sorted(associations.items())}

    def count_term_mentions(self, term: str) -> tuple[int, int]:
        """(matching chunks, distinct matching sources) for a case-insensitive
        substring scan of the WHOLE chunks table.

        This is deliberately a full scan, not a top-k search: a count is a
        corpus property, and any retrieval shortcut undercounts it. Streaming
        Arrow batches keeps the working set to one batch of text at a time,
        so cost is linear in corpus size and memory stays flat.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return 0, 0
        needle = term.lower()
        chunk_hits = 0
        sources: set[str] = set()
        arrow = table.to_arrow().select([_SOURCE_ID, "chunk"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            texts = batch.column("chunk").to_pylist()
            names = batch.column(_SOURCE_ID).to_pylist()
            for name, text in zip(names, texts, strict=True):
                if text and needle in text.lower():
                    chunk_hits += 1
                    sources.add(name)
        return chunk_hits, len(sources)

    def chunk_batches(
        self, columns: list[str], batch_rows: int
    ) -> Iterator[list[Mapping[str, Any]]]:
        """Every chunk row in batches of *batch_rows*, for *columns*; ``source`` is the path."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return
        paths = self._all_source_paths()
        read = [_SOURCE_ID if column == "source" else column for column in columns]
        arrow = table.search().select(read).limit(None).to_arrow()
        for batch in arrow.to_batches(max_chunksize=batch_rows):
            rows: list[Mapping[str, Any]] = list(_pathed(batch.to_pylist(), "source", paths))
            yield rows

    def _all_source_paths(self) -> dict[str, str]:
        """The path of every source row, by id."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return {}
        rows = table.search().select([_SOURCE_ID, _PATH]).limit(None).to_list()
        name = self._locations().name
        return {row[_SOURCE_ID]: name(row[_PATH]) for row in rows}

    def count_chunks(self) -> int:
        """Total chunks in the store."""
        table = self.open_table(CHUNKS_TABLE)
        return table.count_rows() if table is not None else 0

    def get_chunks_by_source(self, source: str) -> list[SearchChunk]:
        """Return every chunk whose ``source`` equals *source*.

        The database does the filtering, so only the matching rows are read.
        A query failure raises rather than falling back to a whole-table scan:
        a document's chunks are a bounded read, and the scan that would rescue
        it costs the entire index, vectors included, in memory.
        """
        return self._chunks_of(source, "1 = 1")

    def _chunks_of(self, source: str, predicate: str) -> list[SearchChunk]:
        """*source*'s chunk rows that *predicate* also matches."""
        table = self.open_table(CHUNKS_TABLE)
        source_id = self.source_ids([source], None).get(source)
        if table is None or source_id is None:
            return []
        query = table.search().where(f"{_SOURCE_ID} = '{source_id}' AND {predicate}")
        rows = _pathed(query.limit(None).to_list(), "source", {source_id: source})
        return [SearchChunk(**row) for row in rows]

    def get_chunks_by_indices(self, source: str, indices: Sequence[int]) -> list[SearchChunk]:
        """Return *source*'s chunks whose ``chunk_index`` is in *indices*.

        Rows come back in ``chunk_index`` order; indices past either end of
        the document are simply absent from the result. Filtering happens in
        the database for the same reason as :meth:`get_chunks_by_source`:
        neighbor expansion runs once per hit source per query, so a
        whole-table rescue would spike memory on the hottest path there is.
        """
        if not indices:
            return []
        wanted = ", ".join(str(int(i)) for i in indices)
        rows = self._chunks_of(source, f"chunk_index IN ({wanted})")
        return sorted(rows, key=lambda c: c.chunk_index)

    def _delete_rows_of_unlocked(self, source_ids: Iterable[str]) -> None:
        """Delete the rows the sources *source_ids* hold in every per-source table.

        Caller must hold ``write_lock()``. One ``IN`` delete per table covers
        every source, so a batched flush pays a constant number of predicate
        deletes instead of one set per document. A delete failure propagates:
        swallowed, it would leave every flushed file silently stale; raised,
        the flush fails and the files replan on the next sync.
        """
        quoted = _in_list(source_ids)
        if not quoted:
            return
        for name in _PER_SOURCE_TABLES:
            table = self.open_table(name)
            if table is not None:
                table.delete(f"{_SOURCE_ID} IN ({quoted})")

    def delete_by_source(self, source: str, source_type: SourceType) -> None:
        """Delete the chunks and page texts of *source* of *source_type*."""
        with self._write_lock():
            self._delete_rows_of_unlocked(self.source_ids([source], source_type).values())
        self._invalidate_source_cache()

    def add_page_texts(self, records: list[dict]) -> int:
        """Add per-page text rows (no vectors). Returns count added."""
        if not records:
            return 0
        with self._write_lock():
            ids = self._ensure_ids_unlocked(_documents(record["source"] for record in records))
            table = ensure_table(self.get_db(), PAGE_TEXTS_TABLE, _page_texts_schema())
            table.add(_keyed(records, "source", ids))
        return len(records)

    def get_page_texts(self, source: str | None = None) -> list[PageTextRecord]:
        """Return per-page text rows, all or for a single *source*."""
        arrow = self.page_texts_arrow(source)
        return cast("list[PageTextRecord]", arrow.to_pylist())

    def page_texts_arrow(self, source: str | None = None) -> pa.Table:
        """Return per-page text rows as an Arrow table in a single scan.

        The columnar sibling of :meth:`get_page_texts`: the export path keeps the
        whole set in Arrow (no per-row Python objects) from read through file
        write. Empty with the canonical schema when the table or *source* is empty.
        """

        table = self.open_table(PAGE_TEXTS_TABLE)
        source_id = None
        if source is not None:
            source_id = self.source_ids([source], SourceType.DOCUMENT).get(source, "")
        if table is None:
            arrow = _page_texts_schema().empty_table()
        else:
            query = table.search().select(_page_texts_schema().names)
            if source_id is not None:
                query = query.where(f"{_SOURCE_ID} = '{escape_sql_string(source_id)}'")
            arrow = query.limit(None).to_arrow()
        ids = arrow.column(_SOURCE_ID).to_pylist()
        paths = self.source_paths(ids)
        shown = pa.array([paths.get(key) for key in ids], pa.utf8())
        return arrow.set_column(0, "source", shown).filter(pc.is_valid(shown))

    def sources_arrow(self) -> pa.Table:
        """Return each tracked source's extraction metadata as an Arrow table.

        The columnar sibling of :meth:`get_sources`, keyed by ``source`` so it
        joins straight onto the page-text table. ``get_sources`` builds a dict per
        source, which on a corpus of millions of single-page documents costs more
        than the text being exported.
        """
        import pyarrow as pa

        table = self.open_table(SOURCES_TABLE)
        columns = ["source", *SourceMeta._fields]
        if table is None:
            return pa.schema([pa.field(name, pa.utf8()) for name in columns]).empty_table()
        query = table.search().where(_LISTED).select([_PATH, *SourceMeta._fields])
        arrow = query.limit(None).to_arrow().rename_columns(columns)
        name = self._locations().name
        shown = pa.array([name(path) for path in arrow.column("source").to_pylist()], pa.utf8())
        arrow = arrow.set_column(0, "source", shown)
        if pc.count_distinct(arrow.column("source")).as_py() == arrow.num_rows:
            return arrow
        # One row per source, so a caller joining on it cannot fan out. A doubled
        # row would otherwise multiply every page it owns. Last wins; single-threaded
        # because that is the only execution mode with an ordered aggregate.
        grouped = arrow.group_by("source", use_threads=False).aggregate(
            [(name, "last") for name in SourceMeta._fields]
        )
        renamed = grouped.rename_columns(
            [name.removesuffix("_last") for name in grouped.schema.names]
        )
        return renamed.select(columns)

    def wiki_chunk_sources(self) -> set[str]:
        """Return the distinct sources of the chunk rows written by the wiki layer."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return set()
        rows = (
            table.search()
            .where(f"chunk_type = '{ChunkType.WIKI}'")
            .select([_SOURCE_ID])
            .limit(None)
            .to_list()
        )
        return set(self.source_paths(row[_SOURCE_ID] for row in rows).values())

    def wiki_pages_citing(self, paths: Iterable[str]) -> set[str]:
        """The wiki pages whose citations cite a source at or below one of *paths*."""
        table = self.open_table(CITATIONS_TABLE)
        roots = set(paths)
        if table is None or not roots:
            return set()
        rows = table.search().select([_SOURCE_ID, "wiki_source"]).limit(None).to_list()
        cited = self.source_paths(row[_SOURCE_ID] for row in rows)
        return {
            row["wiki_source"]
            for row in rows
            if row[_SOURCE_ID] in cited and _at_or_below(cited[row[_SOURCE_ID]], roots)
        }

    def wiki_citations_by_page(self) -> dict[str, list[CitationRecord]]:
        """Every citation, naming its source by path, under the wiki page that holds it."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return {}
        by_page: dict[str, list[CitationRecord]] = {}
        for record in self._shown_citations(table.search().limit(None).to_list()):
            by_page.setdefault(record["wiki_source"], []).append(record)
        return by_page

    def wiki_citation_sources(self) -> set[str]:
        """Return the distinct wiki_source values present in the citations table."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return set()
        rows = table.search().select(["wiki_source"]).limit(None).to_list()
        return {row["wiki_source"] for row in rows}

    def get_sources(
        self,
        *,
        search: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SourceRecord]:
        """Return source records, filtered by *search* and sliced by offset/limit."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        locations = self._locations()
        query = table.search().where(_listed_matching(search, locations))
        if offset:
            query = query.offset(offset)
        return [_source_record(row, locations) for row in query.limit(limit).to_list()]

    def get_sources_where(self, keep: Callable[[str], bool]) -> list[SourceRecord]:
        """Source records whose filename *keep* accepts, read in batches.

        A row *keep* refuses never becomes a Python object, so a caller that wants
        one slice of a large table holds only that slice.
        """
        import pyarrow as pa

        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        kept: list[SourceRecord] = []
        locations = self._locations()
        query = table.search().where(_LISTED).limit(None)
        for batch in query.to_batches(_SOURCE_SCAN_BATCH_ROWS):
            names = [locations.name(path) for path in batch.column(_PATH).to_pylist()]
            mask = pa.array([keep(name) for name in names])
            kept.extend(_source_record(row, locations) for row in batch.filter(mask).to_pylist())
        return kept

    def missing_sources(self, names: Iterable[str]) -> set[str]:
        """The *names* that hold no source row, from one pass over the name column."""
        missing = set(names)
        table = self.open_table(SOURCES_TABLE)
        if table is None or not missing:
            return missing
        name = self._locations().name
        scan = table.search().where(_LISTED).select([_PATH]).limit(None)
        for batch in scan.to_batches(_SOURCE_SCAN_BATCH_ROWS):
            missing.difference_update(name(path) for path in batch.column(_PATH).to_pylist())
        return missing

    def sources_version(self) -> int | None:
        """The version of the source table now, or None when there is no table."""
        table = self.open_table(SOURCES_TABLE)
        return None if table is None else int(table.version)

    def sources_by_hash(self, hashes: Sequence[str], *, version: int) -> dict[str, list[str]]:
        """Document sources at *version* of the source table, by content hash, for *hashes*.

        A failed or removed file counts at the hash it is held out at.
        """
        found: dict[str, list[str]] = {}
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return found
        table.checkout(version)
        name = self._locations().name
        documents = f"source_type != '{SourceType.IMPORTED}' AND {_LISTED}"
        held = f"{_HELD_OUT} IN ('{HeldOut.FAILED}', '{HeldOut.REMOVED}')"
        for start in range(0, len(hashes), _SOURCE_STAT_BATCH_ROWS):
            quoted = _in_list(hashes[start : start + _SOURCE_STAT_BATCH_ROWS])
            for column, where in (("file_hash", documents), (_HELD_HASH, held)):
                query = table.search().where(f"{column} IN ({quoted}) AND {where}")
                for row in query.select([_PATH, column]).limit(None).to_list():
                    found.setdefault(row[column], []).append(name(row[_PATH]))
        return found

    def sources_checkpoint(self) -> SourcesCheckpoint:
        """The time and the version of the source table, read under the write lock.

        A source row is stamped under that lock, so no row stamped before the
        time is committed later.
        """
        with self._write_lock():
            return SourcesCheckpoint(datetime.now(UTC).isoformat(), self.sources_version())

    def wiki_refreshed(self) -> SourcesCheckpoint | None:
        """The checkpoint up to which the wiki's page index covers the source rows.

        None with no record or an unreadable one: every source row is owed.
        An unreadable record is reported and deleted, so the next read is silent.
        """
        record = self._config.lancedb_dir / _WIKI_REFRESHED_NAME
        try:
            fields = json.loads(record.read_text(encoding="utf-8"))
            version = fields["version"]
            return SourcesCheckpoint(str(fields["time"]), None if version is None else int(version))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning(_WIKI_REFRESHED_UNREADABLE, record, exc)
            # A record that cannot be deleted stays, and the next read reports it again.
            with suppress(OSError):
                self.forget_wiki_refreshed()
            return None

    def record_wiki_refreshed(self, checkpoint: SourcesCheckpoint) -> None:
        """Record that the wiki's page index covers the source rows up to *checkpoint*.

        A write that fails leaves the record as it was.
        """
        record = self._config.lancedb_dir / _WIKI_REFRESHED_NAME
        write_private_text(record, json.dumps(checkpoint._asdict()))

    def forget_wiki_refreshed(self) -> None:
        """Delete the record: the wiki's page index covers no source row."""
        (self._config.lancedb_dir / _WIKI_REFRESHED_NAME).unlink(missing_ok=True)

    def sources_ingested_since(self, since: str) -> list[str]:
        """The filename of each source row written at or after *since*, an ISO time in UTC."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        written = f"ingested_at >= '{escape_sql_string(since)}' AND {_LISTED}"
        query = table.search().where(written).select([_PATH])
        name = self._locations().name
        return [name(row[_PATH]) for row in query.limit(None).to_list()]

    def count_sources(self, *, search: str | None = None) -> int:
        """Count tracked sources matching *search* without materializing rows."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return 0
        count: int = table.count_rows(filter=_listed_matching(search, self._locations()))
        return count

    def _source_row(
        self,
        file_hash: str,
        chunk_count: int,
        source_type: SourceType,
        stat: SourceStat | None = None,
        meta: SourceMeta | None = None,
    ) -> dict:
        """The fields of one ``_sources`` row but its id and path; no stat reads as unknown.

        Absent extraction metadata persists as NULL, matching rows written
        before the metadata columns existed.
        """
        meta = meta or SourceMeta()
        return {
            "file_hash": file_hash,
            # Stamped under the write lock that commits the row: sources_checkpoint relies on it.
            "ingested_at": datetime.now(UTC).isoformat(),
            "chunk_count": chunk_count,
            "source_type": source_type,
            "size_bytes": stat.size_bytes if stat else SOURCE_STAT_UNKNOWN,
            "mtime_ns": stat.mtime_ns if stat else SOURCE_STAT_UNKNOWN,
            "stat_captured_ns": stat.captured_ns if stat else SOURCE_STAT_UNKNOWN,
            "title": meta.title or None,
            "authors": meta.authors or None,
            "created_at": meta.created_at or None,
            **dict.fromkeys(_HELD_COLUMNS),
        }

    def _placeholder_row(self, source_type: SourceType) -> dict:
        """The fields of a row that only names a source: a wiki page, or a source held out."""
        row = self._source_row("", 0, source_type)
        return row if _is_wiki(source_type) else {**row, _HELD_OUT: HeldOut.REMOVED}

    def _put_sources_unlocked(
        self, sources: LanceTable, writes: Iterable[_SourceWrite], *, keep_documents: bool = False
    ) -> dict[str, str]:
        """Put each of *writes* in *sources* in one merge on id; returns each name's id. Lock held.

        A write keeps the id of the row of its type class it finds at its key or the
        other form of its name, else takes its new id or a random one, and its row is
        stored at its key. With *keep_documents* a removal leaves a row that holds a
        document as it is.
        """
        writes = list(
            {(write.name, _is_wiki(write.source_type)): write for write in writes}.values()
        )
        locations = self._locations()
        found = _found_rows(sources, writes, locations) if writes else {}
        ids: dict[str, str] = {}
        rows: dict[str, dict[str, Any]] = {}
        for write in writes:
            held = found.get(write.name)
            ids[write.name] = held[_SOURCE_ID] if held else write.new_id or _new_source_id()
            row = self._written_row(write, held, keep_documents=keep_documents)
            if row is not None:
                path = locations.key(write.name, write.source_type)
                rows[ids[write.name]] = {**row, _SOURCE_ID: ids[write.name], _PATH: path}
        if rows:
            merge = sources.merge_insert(_SOURCE_ID).when_matched_update_all()
            merge = merge.when_not_matched_insert_all()
            merge.execute(pa.Table.from_pylist(list(rows.values()), schema=sources.schema))
        return ids

    def _written_row(
        self, write: _SourceWrite, found: dict[str, Any] | None, *, keep_documents: bool
    ) -> dict[str, Any] | None:
        """The fields *write* puts over the row it *found*; None when it changes nothing."""
        if write.record is None and write.fields is None and found is not None:
            return None
        base = write.fields if write.fields is not None else found
        row = dict(base) if base is not None else self._placeholder_row(write.source_type)
        if write.record is None:
            return row
        holds = found is not None and _holds_document(found)
        if keep_documents and holds and write.record.kind is HeldOut.REMOVED:
            return None
        return _held_row(row, write.record, holds_document=holds)

    def _sources_table(self) -> LanceTable:
        """Open or create ``_sources``.

        A table that names sources by id without a ``_sources`` beside it is what a
        rebuild that stopped after its first drop leaves: no path names its rows,
        so it goes before the new ``_sources`` comes.
        """
        db = self.get_db()
        names = table_names(db)
        if SOURCES_TABLE not in names:
            for name in (*_PER_SOURCE_TABLES, CITATIONS_TABLE):
                if name in names:
                    db.drop_table(name)
        return ensure_table(db, SOURCES_TABLE, _sources_schema())

    def _lookup(self, column: str, values: Iterable[str], other: str) -> dict[str, str]:
        """``{value: other}`` for the source rows whose *column* holds one of *values*."""
        return {row[column]: row[other] for row in self._rows_where(column, values, [other])}

    def _rows_where(
        self, column: str, values: Iterable[str], others: list[str]
    ) -> Iterator[dict[str, Any]]:
        """*column* and *others* of each source row whose *column* holds one of *values*."""
        table = self.open_table(SOURCES_TABLE)
        wanted = sorted(set(values))
        if table is None:
            return
        for start in range(0, len(wanted), _SOURCE_STAT_BATCH_ROWS):
            batch = _in_list(wanted[start : start + _SOURCE_STAT_BATCH_ROWS])
            query = table.search().where(f"{column} IN ({batch})").select([column, *others])
            yield from query.limit(None).to_list()

    def source_ids(self, paths: Iterable[str], source_type: SourceType | None) -> dict[str, str]:
        """The id of each of the names *paths* that a source row of *source_type* holds.

        A name finds its source at the key of each type of the class of *source_type*
        (a wiki page, or not), the class a write keeps one row per name in. With
        *source_type* None, a name finds a wiki page first.
        """
        names = list(paths)
        types = [
            kind
            for kind in _READ_ORDER
            if source_type is None or _is_wiki(kind) == _is_wiki(source_type)
        ]
        key = self._locations().key
        wanted = [((key(name, kind), kind), name) for kind in types for name in names]
        rows = self._rows_where(_PATH, {at[0] for at, _ in wanted}, ["source_type", _SOURCE_ID])
        held = {(row[_PATH], row["source_type"]): row[_SOURCE_ID] for row in rows}
        found: dict[str, str] = {}
        for at, name in wanted:
            if at in held:
                found.setdefault(name, held[at])
        return found

    def source_paths(self, source_ids: Iterable[str]) -> dict[str, str]:
        """The shown name of each of *source_ids* that a source row holds."""
        name = self._locations().name
        found = self._lookup(_SOURCE_ID, source_ids, _PATH)
        return {source_id: name(path) for source_id, path in found.items()}

    def keyed_source_ids(self, paths: Iterable[str]) -> dict[str, str]:
        """The id the keying of an older library gave each of *paths*, when a row holds it."""
        keyed = {path: _md5(path) for path in paths}
        held = self.source_paths(keyed.values())
        return {path: source_id for path, source_id in keyed.items() if source_id in held}

    def _ensure_ids_unlocked(self, sources: Iterable[tuple[str, SourceType]]) -> dict[str, str]:
        """The id of each named source of its type, adding a placeholder row where none is.

        Caller holds the lock.
        """
        writes = [_SourceWrite(name, source_type) for name, source_type in sources]
        return self._put_sources_unlocked(self._sources_table(), writes) if writes else {}

    def upsert_source(
        self,
        filename: str,
        file_hash: str,
        chunk_count: int,
        source_type: SourceType = SourceType.DOCUMENT,
        stat: SourceStat | None = None,
        meta: SourceMeta | None = None,
    ) -> None:
        """Add or update a source tracking record."""
        with self._write_lock():
            fields = self._source_row(file_hash, chunk_count, source_type, stat, meta)
            write = _SourceWrite(filename, source_type, fields)
            self._put_sources_unlocked(self._sources_table(), [write])
        self._invalidate_source_cache()

    def update_source_stats(self, backfills: list[SourceStatBackfill]) -> None:
        """Record size/mtime for already-tracked sources in batched locked writes."""
        if not backfills:
            return
        for start in range(0, len(backfills), _SOURCE_STAT_BATCH_ROWS):
            batch = backfills[start : start + _SOURCE_STAT_BATCH_ROWS]
            with self._write_lock():
                writes = [_stat_write(backfill) for backfill in batch]
                self._put_sources_unlocked(self._sources_table(), writes)
        self._invalidate_source_cache()

    def optimize_sources(self) -> None:
        """Compact the sources table; per-flush upserts otherwise accrete tiny versions."""
        with self._write_lock():
            table = self.open_table(SOURCES_TABLE)
            if table is None:
                return
            try:
                table.optimize()
            except Exception:
                log.debug("Sources table optimize failed", exc_info=True)

    def write_chunks_batch(self, items: list[ChunkWrite]) -> int:
        """Write several documents' chunks in one locked transaction. Returns chunks added.

        One ``write_lock`` acquisition covers the batch's cleanup deletes, page
        texts, chunk add, and source upserts, so a reader never observes a
        half-applied batch. Page texts land after the cleanup and before the
        source rows, so a page-text failure leaves the rows stale and the files
        replan next sync; a document with no chunks still persists its page
        texts and source row. The embedding-identity gate and per-vector
        dimension check mirror ``add_chunks``; a dimension mismatch raises and
        the whole batch is rejected.
        """
        if not items:
            return 0
        with self._write_lock(timeout=BATCH_LOCK_TIMEOUT):
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            self._fts_ready = False
            self._scalar_ready = False
            all_records = [rec for it in items for rec in it.records]
            _check_vector_dims(all_records, embedding_dim)
            db = self.get_db()
            self._refresh_key_indexes_unlocked(items, len(all_records))
            # A new source's row exists before its chunks, so a flush that stops
            # between them leaves rows the next flush of the file finds and deletes.
            ids = self._ensure_ids_unlocked((it.source, it.source_type) for it in items)
            self._delete_rows_of_unlocked([ids[it.source] for it in items if it.needs_cleanup])
            page_rows = [row for it in items for row in (it.page_texts or [])]
            if page_rows:
                table = ensure_table(db, PAGE_TEXTS_TABLE, _page_texts_schema())
                table.add(_keyed(page_rows, "source", ids))
            keyed = _keyed(all_records, "source", ids)
            self._add_chunk_records_unlocked(keyed, embedding_model, embedding_dim)
            self._put_sources_unlocked(self._sources_table(), self._batch_source_writes(items))
        self._invalidate_source_cache()
        return len(all_records)

    def _refresh_key_indexes_unlocked(self, items: list[ChunkWrite], chunk_rows: int) -> None:
        """Keep the key index of each table this flush appends to current. Caller holds the lock."""
        flush_rows = {
            CHUNKS_TABLE: chunk_rows,
            PAGE_TEXTS_TABLE: sum(len(it.page_texts or []) for it in items),
            SOURCES_TABLE: len(items),
        }
        for name, column in _FLUSH_KEY_COLUMNS:
            table = self.open_table(name)
            if table is None or not flush_rows[name]:
                continue
            try:
                _refresh_key_index_unlocked(table, column, flush_rows[name])
            except Exception:
                self._report_key_index_failure(table, name, column)

    def _report_key_index_failure(self, table: LanceTable, name: str, column: str) -> None:
        """Warn once for a populated table whose key index cannot be refreshed, then at debug.

        The delete stays correct and scans the table, so every flush costs more
        as the table grows.
        """
        first = name not in self._key_index_warned and _holds_rows(table)
        if first:
            self._key_index_warned.add(name)
        log.log(
            logging.WARNING if first else logging.DEBUG,
            "Key index refresh failed on '%s.%s'; each write scans the table until it succeeds",
            name,
            column,
            exc_info=True,
        )

    def _add_chunk_records_unlocked(
        self,
        all_records: list[dict],
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Add the batch's chunk rows, writing meta on first use. Caller holds ``write_lock()``."""
        if not all_records:
            return
        self._chunks_table().add(all_records)
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)

    def _batch_source_writes(self, items: list[ChunkWrite]) -> list[_SourceWrite]:
        """One ``_sources`` row per batched document, of the type its item states."""
        return [
            _SourceWrite(
                it.source,
                it.source_type,
                self._source_row(it.file_hash, len(it.records), it.source_type, it.stat, it.meta),
            )
            for it in items
        ]

    def _drop_source_rows_unlocked(self, source_ids: list[str]) -> None:
        """Delete the source rows of *source_ids*, holding out each that a citation names.

        A held-out row keeps the path of a removed source for the citations of it,
        and holds nothing out of a sync.
        """
        if not source_ids:
            return
        table = self._sources_table()
        cited = self._cited_ids(source_ids)
        if cited:
            where = f"{_SOURCE_ID} IN ({_in_list(cited)})"
            table.update(where=where, values_sql=_CITED_ONLY_SQL)
        dropped = [source_id for source_id in source_ids if source_id not in cited]
        if dropped:
            _safe_delete_unlocked(table, f"{_SOURCE_ID} IN ({_in_list(dropped)})")

    def _cited_ids(self, source_ids: list[str]) -> set[str]:
        """The *source_ids* a citation row names."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return set()
        query = table.search().where(f"{_SOURCE_ID} IN ({_in_list(source_ids)})")
        return {row[_SOURCE_ID] for row in query.select([_SOURCE_ID]).limit(None).to_list()}

    def delete_source(self, filename: str) -> None:
        """Remove a source file tracking record."""
        with self._write_lock():
            ids = self.source_ids([filename], SourceType.DOCUMENT)
            self._drop_source_rows_unlocked(list(ids.values()))
        self._invalidate_source_cache()

    def _remove_many_unlocked(
        self, names: list[str], held: Mapping[str, HeldRecord] | None = None
    ) -> None:
        """Delete the documents' rows and source records together; hold out each path in *held*.

        All deletes run under the caller's single ``write_lock()`` so no
        reader can observe chunks whose source record is already gone; one
        ``IN`` delete per table covers the whole set.
        """
        held = held or {}
        ids = self.source_ids(names, SourceType.DOCUMENT)
        self._delete_rows_of_unlocked(list(ids.values()))
        self._drop_source_rows_unlocked([ids[name] for name in ids if name not in held])
        self._hold_out_unlocked(held)

    def held_records(self) -> dict[str, HeldRecord]:
        """The record of every path a sync holds out, by path."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return {}
        query = table.search().where(f"{_HELD_HASH} IS NOT NULL")
        rows = query.select([_PATH, *_HELD_COLUMNS]).limit(None).to_list()
        name = self._locations().name
        return {
            name(row[_PATH]): HeldRecord(
                HeldOut(row[_HELD_OUT]), row[_HELD_HASH], row[_HELD_REASON]
            )
            for row in rows
        }

    def hold_failed(self, failed: Mapping[str, HeldRecord]) -> None:
        """Hold each path of *failed* out of later syncs at its hash; a row keeps its document."""
        if not failed:
            return
        with self._write_lock():
            self._hold_out_unlocked(failed)
        self._invalidate_source_cache()

    def _hold_out_unlocked(self, held: Mapping[str, HeldRecord]) -> None:
        """Give each name of *held* its record in one commit. Caller holds ``write_lock()``."""
        if not held:
            return
        writes = [
            _SourceWrite(name, SourceType.DOCUMENT, record=record) for name, record in held.items()
        ]
        self._put_sources_unlocked(self._sources_table(), writes)

    def release_held(self, paths: Iterable[str]) -> None:
        """Hold none of *paths* out of a sync any more."""
        key = self._locations().key
        wanted = sorted({key(path, SourceType.DOCUMENT) for path in paths})
        if not wanted:
            return
        with self._write_lock():
            for start in range(0, len(wanted), _SOURCE_STAT_BATCH_ROWS):
                batch = _in_list(wanted[start : start + _SOURCE_STAT_BATCH_ROWS])
                self._release_unlocked(f"{_PATH} IN ({batch})")
        self._invalidate_source_cache()

    def release_failed(self) -> list[str]:
        """Hold no failed file out of a sync any more; returns their paths, sorted."""
        with self._write_lock():
            released = self._release_unlocked(f"{_HELD_OUT} = '{HeldOut.FAILED}'")
        self._invalidate_source_cache()
        return released

    def _release_unlocked(self, where: str) -> list[str]:
        """Release the held rows *where* selects; a row that holds no document goes.

        Caller holds ``write_lock()``. Returns the released paths, sorted.
        """
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        query = table.search().where(f"({where}) AND {_HELD_HASH} IS NOT NULL")
        rows = query.select([_SOURCE_ID, _PATH, _HELD_OUT, "file_hash"]).limit(None).to_list()
        kept = [row[_SOURCE_ID] for row in rows if _holds_document(row)]
        if kept:
            table.update(
                where=f"{_SOURCE_ID} IN ({_in_list(kept)})",
                values_sql=dict.fromkeys(_HELD_COLUMNS, _NULL_TEXT_SQL),
            )
        self._drop_source_rows_unlocked(
            [row[_SOURCE_ID] for row in rows if not _holds_document(row)]
        )
        name = self._locations().name
        return sorted(name(row[_PATH]) for row in rows)

    def relocate_sources(self, moves: Sequence[SourceMove]) -> SourceRelocation:
        """Move the first of *moves* to their new names in one hold of the write lock.

        Returns the old name each new name took and how many of *moves* the hold
        settled; the caller passes the rest again. A move takes the first of its
        candidates that holds a source row and that no earlier move took; a move
        with none left is settled and absent from the names taken. The choice and
        the move share the lock, so no two writers take one old name.

        A moved source keeps its id, so its rows in every other table follow it:
        only its path, a title that came from the old name, and its disk stat
        change, in one commit of the source table. The sources inside a moved
        archive move with it. A hold settles at most ``_RELOCATE_HOLD_MOVES`` moves.
        """
        if not moves:
            return SourceRelocation({}, 0)
        with self._write_lock():
            step = self._relocate_hold_unlocked(moves[:_RELOCATE_HOLD_MOVES])
        self._invalidate_source_cache()
        return step

    def _relocate_hold_unlocked(self, moves: Sequence[SourceMove]) -> SourceRelocation:
        """Claim and move *moves*. Caller holds ``write_lock()``."""
        from lilbee.data.title import derive_title  # circular at module scope

        sources = self.open_table(SOURCES_TABLE)
        if sources is None:
            return SourceRelocation({}, len(moves))
        locations = self._locations()
        moves, names = _located_moves(moves, locations)
        pools = {id(move.candidates): move.candidates for move in moves}.values()
        unclaimed, removed = _held_sources(sources, {name for pool in pools for name in pool})
        tails = [locations.below_root(move.new) for move in moves]
        claimed = _claimed_names(moves, unclaimed, removed, tails)
        rows = _source_rows(sources, [old for old in claimed if old is not None])
        taken = {move.new: old for move, old in zip(moves, claimed, strict=True) if old}
        titles = {
            old: _title_after_move(rows[old], old, move.new, derive_title)
            for move, old in zip(moves, claimed, strict=True)
            if old
        }
        moved = [
            _moved_source_row(rows[old], move, titles[old])
            for move, old in zip(moves, claimed, strict=True)
            if old
        ]
        if moved:
            movable = sources.search().where(_MOVABLE).limit(None).to_list()
            moved += _member_rows(movable, {old: new for new, old in taken.items()})
            self._delete_rows_of_unlocked(self._displaced_ids(moved))
            merge = sources.merge_insert(_SOURCE_ID).when_matched_update_all()
            displaced = f"{_PATH} IN ({_in_list(row[_PATH] for row in moved)})"
            merge = merge.when_not_matched_by_source_delete(displaced)
            merge.execute(pa.Table.from_pylist(moved, schema=sources.schema))
            self._retitle_chunks_unlocked(
                {rows[old][_SOURCE_ID]: title for old, title in titles.items()}
            )
        return SourceRelocation({names[new]: names[old] for new, old in taken.items()}, len(moves))

    def _displaced_ids(self, moved: list[dict[str, Any]]) -> list[str]:
        """The ids of the rows at the new paths of *moved* that are not moving themselves."""
        moving = {row[_SOURCE_ID] for row in moved}
        held = self._lookup(_PATH, (row[_PATH] for row in moved), _SOURCE_ID).values()
        return [source_id for source_id in held if source_id not in moving]

    def _retitle_chunks_unlocked(self, titles: Mapping[str, str | None]) -> None:
        """Give the chunks of each source id its new title, one update a title. Lock held.

        A title of ``_KEEP_TITLE`` leaves the chunks as they are.
        """
        chunks = self.open_table(CHUNKS_TABLE)
        if chunks is None or _TITLE_COLUMN not in chunks.schema.names:
            return
        by_title: dict[str | None, list[str]] = {}
        for source_id, title in titles.items():
            if title is not _KEEP_TITLE:
                by_title.setdefault(title, []).append(source_id)
        for title, source_ids in by_title.items():
            where = f"{_SOURCE_ID} IN ({_in_list(source_ids)})"
            chunks.update(where=where, values={_TITLE_COLUMN: title})

    def relocate_root(self, old: str, new: str) -> None:
        """Give each source at or below the root folder *old* the same place below *new*."""
        old_at, new_at = root_location(old), root_location(new)
        with self._write_lock():
            sources = self.open_table(SOURCES_TABLE)
            if sources is not None and old_at != new_at:
                _rekey_prefix(self.get_db(), sources, _under_sql(_PATH, old_at), old_at, new_at)
        self._invalidate_source_cache()

    def fill_locations(self) -> None:
        """Give each source row a registered label still names its location."""
        with self._write_lock():
            _fill_locations(self.get_db(), self._locations())
        self._invalidate_source_cache()

    def name_by_label(self, labels: Iterable[str]) -> None:
        """Store each document of the registered roots *labels* by its shown name.

        A root about to leave the registry calls it, so a row it keeps for a
        citation shows its name and not a path on disk, as a label with no root
        does. A document of a root below one of *labels* that stays keeps its place.
        """
        going = set(labels)
        locations = self._locations()
        with self._write_lock():
            sources = self.open_table(SOURCES_TABLE)
            for root, label in locations.deepest_first:
                if sources is not None and label in going:
                    where = f"{_DOCUMENT} AND {locations.named_by_sql(_PATH, root)}"
                    _rekey_prefix(self.get_db(), sources, where, root, label)
        self._invalidate_source_cache()

    def member_sources(self, name: str) -> list[str]:
        """Sources ingested out of the archive *name*: every path under ``name/``."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        locations = self._locations()
        at = escape_sql_string(locations.location(name))
        under = f"starts_with({_PATH}, '{at}/') AND {_LISTED}"
        rows = table.search().where(under).select([_PATH]).limit(None).to_list()
        return [locations.name(row[_PATH]) for row in rows]

    def remove_documents(
        self, names: list[str], held: Mapping[str, HeldRecord] | None = None
    ) -> RemoveResult:
        """Remove documents from the knowledge base by source name.

        Looks up known sources and deletes their chunks and source records. Never
        touches files on disk: source bytes are the user's, and a linked-in corpus
        must never be deleted. Each path of *held* takes its record in the same
        write, so a later sync holds it out. Durable, file-aware removal lives in
        :func:`lilbee.app.ingest.remove_documents_durably`.

        Returns a RemoveResult with removed and not_found lists.
        """
        filenames = [s["filename"] for s in self.get_sources()]
        known = set(filenames)
        members = _members_by_archive(names, filenames)
        targets = [*names, *(m for name in names for m in members.get(name, []))]
        removed = [name for name in targets if name in known]
        not_found = [name for name in names if name not in known]

        if removed or held:
            # One lock acquisition and one IN-delete per table for the whole
            # set, mirroring the batched flush path, instead of a LanceDB
            # version commit per document.
            with self._write_lock():
                self._remove_many_unlocked(removed, held)
            self._invalidate_source_cache()

        return RemoveResult(removed=removed, not_found=not_found)

    def clear_table(self, name: str, predicate: str) -> bool:
        """Delete rows matching *predicate* from *name*. Acquires write lock.

        Returns whether the delete succeeded, so a caller recording the
        outcome (prune, the legacy migration) does not report success over a
        swallowed failure.
        """
        with self._write_lock():
            table = self.open_table(name)
            if table is None:
                return True
            return _safe_delete_unlocked(table, predicate)

    def clear_and_add(self, name: str, schema: pa.Schema, rows: list[dict], predicate: str) -> None:
        """Replace the rows matching *predicate* with *rows* in one locked write.

        Delete and add run under a single write lock, so a reader never observes
        the table emptied mid-rebuild. A delete failure propagates, following the
        same rule as :meth:`_delete_rows_of_unlocked`: adding over rows whose
        predecessors are still there duplicates them, and reporting success would
        leave the caller acting on state it thinks it replaced.
        """
        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, name, schema)
            table.delete(predicate)
            if rows:
                table.add(rows)

    def add_citations(self, records: list[CitationRecord]) -> int:
        """Add citation records to the store. Returns count added."""
        if not records:
            return 0
        with self._write_lock():
            table = ensure_table(self.get_db(), CITATIONS_TABLE, _citations_schema())
            table.add(self._keyed_citations_unlocked(records))
        return len(records)

    def _keyed_citations_unlocked(self, records: Iterable[CitationRecord]) -> list[dict]:
        """Citation rows that name each cited source by id. Caller holds the lock."""
        records = list(records)
        ids = self._ensure_ids_unlocked(_documents(record["source_filename"] for record in records))
        return _keyed(records, "source_filename", ids)

    def _shown_citations(self, rows: list[dict[str, Any]]) -> list[CitationRecord]:
        """Citation *rows* as records, each naming its source by path."""
        paths = self.source_paths(row[_SOURCE_ID] for row in rows)
        return cast("list[CitationRecord]", _pathed(rows, "source_filename", paths))

    def get_citations_for_wiki(self, wiki_source: str) -> list[CitationRecord]:
        """Get all citations for a wiki page."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(wiki_source)
        return self._shown_citations(table.search().where(f"wiki_source = '{escaped}'").to_list())

    def get_citations_for_source(self, source_filename: str) -> list[CitationRecord]:
        """Get all citations that reference a source document (reverse lookup)."""
        table = self.open_table(CITATIONS_TABLE)
        source_id = self.source_ids([source_filename], SourceType.DOCUMENT).get(source_filename)
        if table is None or source_id is None:
            return []
        rows = table.search().where(f"{_SOURCE_ID} = '{source_id}'").to_list()
        return cast(
            "list[CitationRecord]", _pathed(rows, "source_filename", {source_id: source_filename})
        )

    def delete_citations_for_wiki(self, wiki_source: str) -> bool:
        """Delete all citations for a wiki page. Returns whether the delete succeeded."""
        return self.clear_table(CITATIONS_TABLE, _citations_for_wiki_predicate(wiki_source))

    def delete_all_wiki_rows(self) -> bool:
        """Delete every wiki chunk row, every citation, and every mention.
        Returns whether all deletes succeeded, so a caller cannot report a wipe
        over a swallowed failure. Only the wiki layer writes citations and
        mentions, so wiping it empties those tables outright. All deletes run
        before the results combine.
        """
        chunks_cleared = self.clear_table(CHUNKS_TABLE, f"chunk_type = '{ChunkType.WIKI}'")
        citations_cleared = self.clear_table(CITATIONS_TABLE, "1 = 1")
        mentions_cleared = self.clear_wiki_mentions()
        return chunks_cleared and citations_cleared and mentions_cleared

    def replace_citations_for_wiki(self, wiki_source: str, records: list[CitationRecord]) -> None:
        """Swap a wiki page's citations for *records* under one write lock.

        The lock keeps another writer from landing between the delete and the
        add; the two remain separate commits, so a crash in between leaves the
        rows absent until the page is regenerated or accepted again.
        """
        with self._write_lock():
            table = ensure_table(self.get_db(), CITATIONS_TABLE, _citations_schema())
            rows = self._keyed_citations_unlocked(records)
            table.delete(_citations_for_wiki_predicate(wiki_source))
            if rows:
                table.add(rows)

    def replace_wiki_mentions_for_source(self, source: str, rows: list[dict]) -> None:
        """Swap one source's wiki mention rows for *rows* under one write lock.

        A source contributes its whole mention set at once, so an incremental
        wiki refresh replaces exactly that source's rows and leaves every other
        source's evidence in place for the corpus-wide aggregate.
        """
        with self._write_lock():
            ids = self._ensure_ids_unlocked(_documents(row["source"] for row in rows))
            table = ensure_table(self.get_db(), WIKI_MENTIONS_TABLE, _wiki_mentions_schema())
            for source_id in self.source_ids([source], SourceType.DOCUMENT).values():
                table.delete(f"{_SOURCE_ID} = '{source_id}'")
            if rows:
                table.add(_keyed(rows, "source", ids))

    def wiki_mention_rows(self, slugs: Iterable[str] | None = None) -> list[dict]:
        """Every wiki mention row, or only those for *slugs*, with its source's id and path.

        The wiki aggregates these across sources to rebuild the stub index. A
        full refresh reads them all; an incremental one reads only the slugs it
        touched, to recompute their corpus-wide totals without a full scan.
        """
        table = self.open_table(WIKI_MENTIONS_TABLE)
        if table is None:
            return []
        query = table.search()
        if slugs is not None:
            wanted = list(slugs)
            if not wanted:
                return []
            joined = ", ".join(f"'{escape_sql_string(s)}'" for s in wanted)
            query = query.where(f"slug IN ({joined})")
        rows = query.limit(None).to_list()
        paths = self.source_paths(row[_SOURCE_ID] for row in rows)
        return [
            {**row, "source": paths[source_id]}
            for row in rows
            if (source_id := row[_SOURCE_ID]) in paths
        ]

    def clear_wiki_mentions(self) -> bool:
        """Drop every wiki mention row (a full rebuild starts from nothing)."""
        return self.clear_table(WIKI_MENTIONS_TABLE, "1 = 1")

    def has_wiki_mentions(self) -> bool:
        """Whether any wiki mention row exists.

        A cold store or one migrated from the file-only index has none, which
        forces a refresh to rebuild in full and seed the table before an
        incremental pass can aggregate over it.
        """
        table = self.open_table(WIKI_MENTIONS_TABLE)
        return table is not None and table.count_rows() > 0

    def _memories_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("owner", pa.utf8()),
                pa.field("shared", pa.bool_()),
                pa.field("kind", pa.utf8()),
                pa.field("source", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
                pa.field("created_at", pa.utf8()),
                pa.field("updated_at", pa.utf8()),
            ]
        )

    def _duplicate_memory_id_unlocked(self, table: LanceTable, record: MemoryRow) -> str | None:
        """Return the id of a near-duplicate same-owner, same-kind memory, if any."""
        if table.count_rows() == 0:
            return None
        predicate = (
            f"owner = '{escape_sql_string(record.owner)}' "
            f"AND kind = '{escape_sql_string(record.kind)}'"
        )
        rows = (
            table.search(record.vector)
            .distance_type(_VECTOR_METRIC)
            .where(predicate)
            .limit(1)
            .to_list()
        )
        if rows and rows[0].get("_distance", 1.0) <= self._config.memory_dedup_distance:
            return str(rows[0]["id"])
        return None

    def _evict_overflow_unlocked(self, table: LanceTable, owner: str) -> None:
        """Delete oldest memories for *owner* so an incoming insert stays within the cap."""
        cap = self._config.memory_max_per_owner
        predicate = f"owner = '{escape_sql_string(owner)}'"
        rows = table.search().where(predicate).limit(None).to_list()
        if len(rows) < cap:
            return
        rows.sort(key=lambda r: r.get("created_at", ""))
        for row in rows[: len(rows) - (cap - 1)]:
            _safe_delete_unlocked(table, f"id = '{escape_sql_string(str(row['id']))}'")

    def add_memory(self, record: MemoryRow) -> str:
        """Insert *record*, or update the nearest same-owner duplicate in place.

        Returns the stored id. Raises ``EmbeddingModelMismatchError`` when the store
        was built under a different embedding model, and ``ValueError`` on a vector
        dimension mismatch.
        """
        if len(record.vector) != self._config.embedding_dim:
            raise ValueError(
                f"Memory vector dimension mismatch: expected "
                f"{self._config.embedding_dim}, got {len(record.vector)}"
            )
        with self._write_lock():
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            db = self.get_db()
            table = ensure_table(db, MEMORIES_TABLE, self._memories_schema())
            duplicate_id = self._duplicate_memory_id_unlocked(table, record)
            if duplicate_id is not None and _safe_delete_unlocked(
                table, f"id = '{escape_sql_string(duplicate_id)}'"
            ):
                # Only reuse the id once the old row is actually gone; a swallowed
                # delete failure would otherwise leave two rows with the same id.
                record.id = duplicate_id
            self._evict_overflow_unlocked(table, record.owner)
            table.add([record.model_dump(mode="json")])
            if self.get_meta() is None:
                self._write_meta_unlocked(
                    embedding_model=embedding_model, embedding_dim=embedding_dim
                )
            return record.id

    def get_memories(
        self,
        *,
        owner_predicate: str,
        kind: MemoryKind | None = None,
    ) -> list[MemoryRow]:
        """Return memories matching *owner_predicate* and optional *kind*, newest first."""
        table = self.open_table(MEMORIES_TABLE)
        if table is None:
            return []
        clauses = [f"({owner_predicate})"]
        if kind is not None:
            clauses.append(f"kind = '{escape_sql_string(kind)}'")
        rows = table.search().where(" AND ".join(clauses)).limit(None).to_list()
        memories = [MemoryRow(**r) for r in rows]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories

    def search_memories(
        self,
        query_vector: Vector,
        *,
        owner_predicate: str,
        top_k: int,
        max_distance: float,
    ) -> list[MemoryRow]:
        """Vector-recall FACT memories within *max_distance*, best first."""
        table = self.open_table(MEMORIES_TABLE)
        if table is None or top_k <= 0:
            return []
        self._ensure_embedding_compat()
        predicate = f"({owner_predicate}) AND kind = '{MemoryKind.FACT}'"
        rows = (
            table.search(query_vector)
            .distance_type(_VECTOR_METRIC)
            .where(predicate)
            .limit(top_k)
            .to_list()
        )
        return [MemoryRow(**r) for r in rows if r.get("_distance", 1.0) <= max_distance]

    def update_memory(self, memory_id: str, *, shared: bool, owner: str) -> bool:
        """Set the *shared* flag on *owner*'s memory. Returns True when found and owned.

        The ``owner`` predicate scopes the mutation to the caller's namespace so an
        agent cannot flip another owner's (or the human's) memory.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return False
            predicate = self._owned_memory_predicate(memory_id, owner)
            rows = table.search().where(predicate).limit(1).to_list()
            if not rows:
                return False
            record = MemoryRow(**rows[0])
            record.shared = shared
            record.updated_at = datetime.now(UTC).isoformat()
            # If the delete fails, do not add the modified copy: that would leave
            # two rows for one id. Report not-updated instead.
            if not _safe_delete_unlocked(table, predicate):
                return False
            table.add([record.model_dump(mode="json")])
            return True

    def delete_memory(self, memory_id: str, *, owner: str) -> bool:
        """Delete *owner*'s memory by id. Returns True when a matching row was deleted.

        The ``owner`` predicate scopes the delete to the caller's namespace so an
        agent cannot destroy another owner's (or the human's) memory.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return False
            predicate = self._owned_memory_predicate(memory_id, owner)
            if not table.search().where(predicate).limit(1).to_list():
                return False
            # Report the real outcome: a swallowed delete failure must not be
            # reported as a successful forget.
            return _safe_delete_unlocked(table, predicate)

    @staticmethod
    def _owned_memory_predicate(memory_id: str, owner: str) -> str:
        """SQL predicate matching a single memory id within *owner*'s namespace."""
        return f"id = '{escape_sql_string(memory_id)}' AND owner = '{escape_sql_string(owner)}'"

    def rebuild_memory_embeddings(self, embed: Callable[[list[str]], list[Vector]]) -> int:
        """Re-embed every memory under the current model, recreating the table.

        The vector column dimension is immutable, so a different-dim model needs a
        fresh table; recreating unconditionally also covers the same-dim case. Memory
        text is human-authored and re-embeddable, so no data is lost. Returns the count.

        The snapshot, embed, and table rebuild all run under the write lock so a
        concurrent ``add_memory`` cannot commit into the read-then-drop window and
        be erased; it either lands before the snapshot or blocks until the rebuild
        finishes.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return 0
            rows = table.search().limit(None).to_list()
            if not rows:
                return 0
            memories = [MemoryRow(**r) for r in rows]
            vectors = embed([m.text for m in memories])
            for memory, vector in zip(memories, vectors, strict=True):
                # MemoryRow serializes to JSON on write, which has no ndarray encoding.
                memory.vector = vector.tolist()
            db = self.get_db()
            db.drop_table(MEMORIES_TABLE)
            new_table = ensure_table(db, MEMORIES_TABLE, self._memories_schema())
            new_table.add([m.model_dump(mode="json") for m in memories])
        return len(memories)

    def close(self) -> None:
        """Release the database connection and reset state."""
        self._db = None
        self._fts_ready = False
        self._title_fts_ready = False
        self._scalar_ready = False

    def drop_all(self) -> None:
        """Drop every table except ``_memories`` -- used by rebuild.

        Memory is user-authored data with no on-disk source, not derived from
        documents, so a rebuild preserves it. Only a factory reset (which deletes
        the data directory) clears it.
        """
        with self._write_lock():
            self._fts_ready = False
            self._title_fts_ready = False
            self._scalar_ready = False
            db = self.get_db()
            # Before the tables: a record that outlives them would describe the next ones.
            self.forget_wiki_refreshed()
            dropped = [name for name in table_names(db) if name != MEMORIES_TABLE]
            # The source rows go first: a drop that stops part way leaves no file unchanged.
            for name in sorted(dropped, key=lambda name: name != SOURCES_TABLE):
                db.drop_table(name)
        self._invalidate_source_cache()
