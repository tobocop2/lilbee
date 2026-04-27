"""Search-result post-processing shared by CLI, HTTP, and MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.data.store import SearchChunk


def resolve_vault_path(source_filename: str) -> str | None:
    """Return *source_filename* as a vault-relative path, or None if unresolvable.

    Resolves symlinks on both sides and rejects ``..`` escapes from
    ``documents_dir``.
    """
    if cfg.vault_base is None:
        return None
    try:
        vault_base = cfg.vault_base.resolve()
        documents_dir = cfg.documents_dir.resolve()
        source_path = (cfg.documents_dir / source_filename).resolve()
        source_path.relative_to(documents_dir)
        relative_docs_dir = documents_dir.relative_to(vault_base)
    except (OSError, ValueError):
        return None
    if not source_path.is_file():
        return None
    return (relative_docs_dir / source_path.relative_to(documents_dir)).as_posix()


def clean_result(result: SearchChunk) -> dict:
    """Return SearchChunk as a JSON dict, stamping vault_path when resolvable."""
    payload = result.model_dump(exclude={"vector"}, exclude_none=True)
    vault_path = resolve_vault_path(result.source)
    if vault_path is not None:
        payload["vault_path"] = vault_path
    return payload
