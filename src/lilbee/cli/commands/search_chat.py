"""Search, ask, chat, and topics commands."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.table import Table

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
    auto_sync,
    clean_result,
    json_output,
)
from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.data.store import SearchScope, scope_to_chunk_type
from lilbee.providers.base import ProviderError

_scope_option = typer.Option(
    SearchScope.BOTH,
    "--scope",
    "-s",
    help="Restrict the pool to raw chunks, wiki pages, or both (default).",
    case_sensitive=False,
)


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

    try:
        results = get_services().searcher.search(
            query,
            top_k=top_k or cfg.top_k,
            chunk_type=scope_to_chunk_type(scope),
        )
    except Exception as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
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
        score = r.get("relevance_score") or r.get("distance") or 0
        table.add_row(r["source"], preview, f"{score:.4f}")
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
        from lilbee.models import ensure_chat_model

        ensure_chat_model()
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
                }
            )
            return

        for token in get_services().searcher.ask_stream(question, chunk_type=chunk_type):
            console.print(token.content, end="")
        console.print()
    except (RuntimeError, ProviderError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None


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
    """Interactive chat loop (auto-syncs first)."""
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

    run_tui(auto_sync=True)


def topics(
    query: str = typer.Argument(None, help="Optional query to find related concepts."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show top concept communities or concepts related to a query."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    from lilbee.concepts import concepts_available

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
        preview = ", ".join(comm.concepts[:5])
        if len(comm.concepts) > 5:
            preview += f" (+{len(comm.concepts) - 5} more)"
        table.add_row(str(comm.cluster_id), str(comm.size), preview)
    console.print(table)
