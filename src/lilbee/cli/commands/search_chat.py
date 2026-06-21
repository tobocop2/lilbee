"""Search, ask, chat, and topics commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.table import Table

from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
    model_option,
    num_ctx_option,
    repeat_penalty_option,
    seed_option,
    temperature_option,
    top_k_sampling_option,
    top_p_option,
)
from lilbee.cli.commands._shared import CHUNK_PREVIEW_LEN
from lilbee.cli.helpers import (
    announce_cold_start,
    announce_ready,
    auto_sync,
    json_output,
)
from lilbee.core.config import cfg
from lilbee.data.store import EmbeddingModelMismatchError, SearchScope, scope_to_chunk_type
from lilbee.providers.base import ProviderError
from lilbee.providers.roles import WorkerRole

# How many top concepts to show inline before truncating with a ``+N more`` tail.
_TOPIC_PREVIEW_LIMIT = 5

_EMBED_MISMATCH_ADOPT_HINT = (
    "Run `lilbee use-embedder {model}` to search this index with its embedder."
)
_EMBED_MISMATCH_REBUILD_HINT = (
    "This index needs a {dim}-dim embedder; run `lilbee rebuild` to re-embed it "
    "under your current model."
)


def _exit_embedding_mismatch(exc: EmbeddingModelMismatchError) -> NoReturn:
    """Print a surface-appropriate mismatch error and exit non-zero.

    Headless: never switches embedder silently. Names the index's embedder and,
    when adoptable (same dim), the one command that makes it searchable.
    """
    hint = (
        _EMBED_MISMATCH_ADOPT_HINT.format(model=exc.persisted_model)
        if exc.dims_match
        else _EMBED_MISMATCH_REBUILD_HINT.format(dim=exc.persisted_dim)
    )
    if cfg.json_mode:
        json_output({"error": str(exc), "hint": hint, "persisted_model": exc.persisted_model})
        raise SystemExit(1)
    console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
    console.print(hint)
    raise SystemExit(1)


_scope_option = typer.Option(
    SearchScope.BOTH,
    "--scope",
    "-s",
    help="Restrict the pool to raw chunks, wiki pages, or both (default).",
    case_sensitive=False,
)


def _display_score(result: dict[str, Any]) -> float:
    """Relevance score, else distance, else 0.0. Explicit None checks keep a
    legitimate 0.0 from falling through a truthy ``or`` chain."""
    score = result.get("relevance_score")
    if score is None:
        score = result.get("distance")
    return 0.0 if score is None else score


def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results"),
    scope: SearchScope = _scope_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    if not query or not query.strip():
        if cfg.json_mode:
            json_output({"error": "query must not be empty"})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] query must not be empty")
        raise SystemExit(1)

    err = announce_cold_start(WorkerRole.EMBED, str(cfg.embedding_model))
    try:
        results = get_services().searcher.search(
            query,
            top_k=top_k or cfg.top_k,
            chunk_type=scope_to_chunk_type(scope),
        )
        announce_ready(err, WorkerRole.EMBED)
    except EmbeddingModelMismatchError as exc:
        _exit_embedding_mismatch(exc)
    except Exception as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
    # Apply the same relevance cutoff the REST and MCP search paths use, so the
    # CLI doesn't surface lower-relevance chunks the API would suppress.
    results = [r for r in results if r.distance is None or r.distance <= cfg.max_distance]
    cleaned = [clean_result(r) for r in results]

    if cfg.json_mode:
        json_output({"command": "search", "query": query, "results": cleaned})
        return

    if not cleaned:
        console.print("No results found.")
        return

    has_relevance = any("relevance_score" in r for r in cleaned)
    table = Table(title="Search Results")
    table.add_column("Source", style=theme.ACCENT)
    table.add_column("Chunk", max_width=80)
    score_label = "Score" if has_relevance else "Distance"
    table.add_column(score_label, justify="right", style=theme.MUTED)

    for r in cleaned:
        chunk_text = r["chunk"]
        preview = chunk_text[:CHUNK_PREVIEW_LEN]
        if len(chunk_text) > CHUNK_PREVIEW_LEN:
            preview += "..."
        table.add_row(r["source"], preview, f"{_display_score(r):.4f}")
    console.print(table)


def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    scope: SearchScope = _scope_option,
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = global_option,
    temperature: float | None = temperature_option,
    top_p: float | None = top_p_option,
    top_k_sampling: int | None = top_k_sampling_option,
    repeat_penalty: float | None = repeat_penalty_option,
    num_ctx: int | None = num_ctx_option,
    seed: int | None = seed_option,
) -> None:
    """Ask a one-shot question (auto-syncs first)."""
    apply_overrides(
        data_dir=data_dir,
        model=model,
        use_global=use_global,
        temperature=temperature,
        top_p=top_p,
        top_k_sampling=top_k_sampling,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        seed=seed,
    )

    try:
        from lilbee.app.settings import apply_settings_update
        from lilbee.modelhub.models import ensure_chat_model

        pulled = ensure_chat_model()
        if pulled is not None:
            apply_settings_update({"chat_model": pulled})
        get_services().embedder.validate_model()
        if cfg.json_mode:
            from rich.console import Console as _QuietConsole

            auto_sync(_QuietConsole(quiet=True))
        else:
            auto_sync(console)

        chunk_type = scope_to_chunk_type(scope)

        if cfg.json_mode:
            result = get_services().searcher.ask_raw(question, chunk_type=chunk_type)
            json_output(
                {
                    "command": "ask",
                    "question": question,
                    "answer": result.answer,
                    "sources": [clean_result(s) for s in result.sources],
                    "cited_sources": [clean_result(s) for s in result.cited_sources],
                }
            )
            return

        err = announce_cold_start(WorkerRole.CHAT, str(cfg.chat_model))
        stream = get_services().searcher.ask_stream(question, chunk_type=chunk_type)
        first = True
        for token in stream:
            if first:
                announce_ready(err, WorkerRole.CHAT)
                first = False
            console.print(token.content, end="")
        console.print()
    except EmbeddingModelMismatchError as exc:
        _exit_embedding_mismatch(exc)
    except (RuntimeError, ProviderError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None


def use_embedder(
    ref: str = typer.Argument(
        ..., help="Embedding model ref to adopt (copy it from a downloaded index's error)."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Switch to embedder REF, downloading it if needed, without rebuilding the index."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    from lilbee.app.models import adopt_embedder
    from lilbee.catalog.compat import UnsupportedArchError

    try:
        result = adopt_embedder(ref)
    except (RuntimeError, ValueError, OSError, UnsupportedArchError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None

    if cfg.json_mode:
        json_output(
            {"command": "use-embedder", "model": result.model, "status": result.status.value}
        )
        return
    console.print(f"Now embedding with [{theme.ACCENT}]{result.model}[/{theme.ACCENT}].")


def chat(
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = global_option,
    temperature: float | None = temperature_option,
    top_p: float | None = top_p_option,
    top_k_sampling: int | None = top_k_sampling_option,
    repeat_penalty: float | None = repeat_penalty_option,
    num_ctx: int | None = num_ctx_option,
    seed: int | None = seed_option,
) -> None:
    """Interactive chat loop. Press S in the TUI to sync pending documents."""
    apply_overrides(
        data_dir=data_dir,
        model=model,
        use_global=use_global,
        temperature=temperature,
        top_p=top_p,
        top_k_sampling=top_k_sampling,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        seed=seed,
    )

    if cfg.json_mode:
        json_output({"error": "Chat requires a terminal, not --json"})
        raise SystemExit(1)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] Chat requires a terminal.")
        raise SystemExit(1)
    from lilbee.cli.tui import run_tui

    run_tui()


def topics(
    query: str = typer.Argument(None, help="Optional query to find related concepts."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show top concept communities or concepts related to a query."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        msg = "Concept graph requires: pip install 'lilbee[graph]'"
        if cfg.json_mode:
            json_output({"error": msg})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]{msg}[/{theme.ERROR}]")
        raise SystemExit(1)

    if not cfg.concept_graph:
        if cfg.json_mode:
            json_output({"error": "Concept graph is disabled (LILBEE_CONCEPT_GRAPH=false)"})
            raise SystemExit(1)
        console.print(
            f"[{theme.ERROR}]Concept graph is disabled.[/{theme.ERROR}] "
            "Enable with LILBEE_CONCEPT_GRAPH=true"
        )
        raise SystemExit(1)

    if not get_services().concepts.get_graph():
        if cfg.json_mode:
            json_output({"error": "Concept graph not available"})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]Concept graph not available.[/{theme.ERROR}]")
        raise SystemExit(1)

    if query:
        _topics_for_query(query)
    else:
        _topics_overview(top_k)


def _topics_for_query(query: str) -> None:
    """Show concepts related to a query."""
    cg = get_services().concepts
    concepts = cg.extract_concepts(query)
    related = cg.expand_query(query)
    all_concepts = concepts + [r for r in related if r not in concepts]

    if cfg.json_mode:
        json_output({"command": "topics", "query": query, "concepts": all_concepts})
        return
    if not all_concepts:
        console.print("No concepts found for this query.")
        return
    console.print(f"Concepts related to [{theme.ACCENT}]{query}[/{theme.ACCENT}]:")
    for c in all_concepts:
        console.print(f"  {c}")


def _topics_overview(top_k: int) -> None:
    """Show top concept communities."""
    from dataclasses import asdict

    communities = get_services().concepts.top_communities(k=top_k)
    if cfg.json_mode:
        json_output({"command": "topics", "communities": [asdict(c) for c in communities]})
        return
    if not communities:
        console.print("No concept communities found. Try syncing some documents first.")
        return
    table = Table(title="Concept Communities")
    table.add_column("Cluster", justify="right", style=theme.MUTED)
    table.add_column("Size", justify="right")
    table.add_column("Top Concepts", style=theme.ACCENT)
    for comm in communities:
        preview = ", ".join(comm.concepts[:_TOPIC_PREVIEW_LIMIT])
        if len(comm.concepts) > _TOPIC_PREVIEW_LIMIT:
            preview += f" (+{len(comm.concepts) - _TOPIC_PREVIEW_LIMIT} more)"
        table.add_row(str(comm.cluster_id), str(comm.size), preview)
    console.print(table)
