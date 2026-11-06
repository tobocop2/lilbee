"""Search-result post-processing shared by CLI, HTTP, and MCP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.data.store import SearchChunk
    from lilbee.retrieval.query.formatting import SourceRef


def resolve_vault_path(source_filename: str) -> str | None:
    """Return *source_filename* as a vault-relative path, or None if unresolvable.

    Resolves symlinks on both sides and rejects ``..`` escapes from
    ``documents_dir``.
    """
    if cfg.vault_base is None:
        return None
    from lilbee.data.ingest.discovery import resolve_source_path

    try:
        vault_base = cfg.vault_base.resolve()
        documents_dir = cfg.documents_dir.resolve()
        source_path = resolve_source_path(source_filename).resolve()
        source_path.relative_to(documents_dir)
        relative_docs_dir = documents_dir.relative_to(vault_base)
    except (OSError, ValueError):
        return None
    if not source_path.is_file():
        return None
    return (relative_docs_dir / source_path.relative_to(documents_dir)).as_posix()


def clean_result(result: SearchChunk, ref: SourceRef | None = None) -> dict:
    """Return SearchChunk as a JSON dict, stamping vault_path when resolvable and the
    ``label`` and ``citation_number`` of *ref*, the number only when it has one."""
    payload = result.model_dump(exclude={"vector"}, exclude_none=True)
    vault_path = resolve_vault_path(result.source)
    if vault_path is not None:
        payload["vault_path"] = vault_path
    if ref is not None:
        payload["label"] = ref.label
        if ref.citation_number is not None:
            payload["citation_number"] = ref.citation_number
    return payload


def clean_results(
    chunks: Sequence[SearchChunk], answered_from: Sequence[SearchChunk] = ()
) -> list[dict]:
    """*chunks* as JSON dicts, each with the label of its source.

    *answered_from* is the retrieved list an answer was written from. A chunk whose
    source is in it also carries the ``[n]`` the answer cites that source by, so a
    cited subset keeps its true numbers. Search results pass none and carry no number.
    """
    refs = _source_refs(chunks, answered_from)
    return [clean_result(c, refs[c.source]) for c in chunks]


def source_labels(chunks: Sequence[SearchChunk]) -> dict[str, str]:
    """The label of each source among *chunks*, as ``clean_results`` stamps it."""
    return {source: ref.label for source, ref in _source_refs(chunks, ()).items()}


def _source_refs(
    chunks: Sequence[SearchChunk], answered_from: Sequence[SearchChunk]
) -> dict[str, SourceRef]:
    """The ref of each source among *chunks*, numbered only by *answered_from*."""
    from lilbee.app.titles import stored_web_titles
    from lilbee.retrieval.query.formatting import source_refs

    titles = stored_web_titles({c.source for c in (*chunks, *answered_from)})
    refs = {r.source: replace(r, citation_number=None) for r in source_refs([*chunks], titles)}
    refs.update((r.source, r) for r in source_refs([*answered_from], titles))
    return refs
