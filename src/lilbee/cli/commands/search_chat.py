"""Search, ask, chat, and topics commands."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.table import Table

from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    chat_model_overridden,
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
from lilbee.cli.log_routing import route_diagnostics_to_log_file
from lilbee.core.config import cfg
from lilbee.data.store import EmbeddingModelMismatchError, SearchScope, scope_to_chunk_type
from lilbee.providers.base import ProviderError
from lilbee.providers.roles import WorkerRole

# How many top concepts to show inline before truncating with a ``+N more`` tail.
_TOPIC_PREVIEW_LIMIT = 5
# Upper bound on retrieved results, matching the REST search route's le=100 cap.
_MAX_TOP_K = 100

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


def _swap_stale_models_to_installed(chat_overridden: bool = False) -> None:
    """Swap a stale chat/embedding ref to an installed model for this run only.

    The persisted config is never rewritten from a one-shot command; durable
    swaps belong to the TUI's interactive startup canonicalization. An explicit
    --model override is honored as given, so an unusable value surfaces the
    engine's not-installed error instead of a silent substitute. The notice
    prints to stderr in every mode: stdout stays answer-only and JSON stays
    parseable.
    """
    from rich.console import Console

    from lilbee.app.settings import apply_ephemeral_model_swap
    from lilbee.modelhub.model_manager import (
        ValidationResult,
        canonicalize_chat_model,
        canonicalize_embedding_model,
    )

    checks = [(canonicalize_embedding_model(), "embedding_model", "Embedding")]
    if not chat_overridden:
        checks.insert(0, (canonicalize_chat_model(), "chat_model", "Chat"))
    err = Console(stderr=True)
    for canon, field, label in checks:
        if canon.status == ValidationResult.OK or canon.effective == canon.original:
            continue
        apply_ephemeral_model_swap(field, canon.effective)
        err.print(
            f"{label} model {canon.original!r} is unavailable ({canon.reason}); "
            f"using installed {canon.effective!r} for this run.",
            style=theme.WARNING,
        )


_MD_FILE_LINK_RE = re.compile(r"\[([^\]]+)\]\((file://[^)]+)\)")


def _print_answer_stream(stream: Any, on_first_token: Callable[[], None]) -> None:
    """Stream an answer to stdout verbatim, then render its Sources block.

    Tokens print with markup off: model text is data, and Rich markup would eat
    the ``[label]`` of every markdown source link (and let the model restyle the
    terminal). On a terminal the Sources block's ``[label](file://...)`` links
    become OSC 8 hyperlinks, clickable even when the path wraps; piped output
    and legacy Windows consoles keep the raw markdown so the URL survives (Rich
    emits no OSC 8 on the legacy path, which would drop it). The marker can span
    tokens, so a marker-sized tail is held back until it can be classified.
    """
    from rich.markup import escape

    from lilbee.retrieval.query.formatting import SOURCES_BLOCK_MARKER

    buf = ""
    hold = len(SOURCES_BLOCK_MARKER)
    in_sources = False
    flushed = False
    try:
        for token in stream:
            on_first_token()
            if token.is_reasoning:
                # Reasoning is not filtered by StreamingCitationFilter, so a
                # thinking trace drafting a Sources: list must never trip the
                # marker scan; it prints live and bypasses the buffer.
                console.print(token.content, end="", markup=False)
                continue
            buf += token.content
            if in_sources:
                continue
            if SOURCES_BLOCK_MARKER in buf:
                head, _, buf = buf.partition(SOURCES_BLOCK_MARKER)
                console.print(head, end="", markup=False)
                in_sources = True
            elif len(buf) > hold:
                console.print(buf[:-hold], end="", markup=False)
                buf = buf[-hold:]
        flushed = True
    finally:
        if not flushed and buf and not in_sources:
            # An exception escaped the stream mid-answer: the held tail is
            # real answer text; print it before the error surfaces.
            console.print(buf, markup=False)
    if not in_sources:
        console.print(buf, markup=False)
        return
    console.print(SOURCES_BLOCK_MARKER, end="", markup=False)
    if not console.is_terminal or console.legacy_windows:
        console.print(buf, markup=False, highlight=False)
        return
    parts: list[str] = []
    last = 0
    for m in _MD_FILE_LINK_RE.finditer(buf):
        parts.append(escape(buf[last : m.start()]))
        parts.append(f"[link={m.group(2)}]{escape(m.group(1))}[/link]")
        last = m.end()
    parts.append(escape(buf[last:]))
    console.print("".join(parts), highlight=False)


def _reject_if_empty(value: str, label: str) -> None:
    """Exit with a uniform error if *value* is empty/whitespace (matches REST)."""
    if value and value.strip():
        return
    msg = f"{label} must not be empty"
    if cfg.json_mode:
        json_output({"error": msg})
        raise SystemExit(1)
    console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {msg}")
    raise SystemExit(1)


def _display_score(result: dict[str, Any]) -> float:
    """Relevance score, else distance, else 0.0. Explicit None checks keep a
    legitimate 0.0 from falling through a truthy ``or`` chain."""
    score = result.get("relevance_score")
    if score is None:
        score = result.get("distance")
    return 0.0 if score is None else score


def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", min=1, help="Number of results"),
    scope: SearchScope = _scope_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    route_diagnostics_to_log_file()

    _reject_if_empty(query, "query")

    err = announce_cold_start(WorkerRole.EMBED, str(cfg.embedding_model))
    try:
        results = get_services().searcher.search(
            query,
            top_k=min(top_k or cfg.top_k, _MAX_TOP_K),
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
    no_sync: bool = typer.Option(
        False, "--no-sync", help="Skip the pre-answer auto-sync (useful on large static corpora)."
    ),
) -> None:
    """Ask a one-shot question."""
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
    route_diagnostics_to_log_file()
    _reject_if_empty(question, "question")

    try:
        from lilbee.app.settings import apply_settings_update
        from lilbee.modelhub.models import ensure_chat_model

        pulled = ensure_chat_model()
        if pulled is not None:
            apply_settings_update({"chat_model": pulled})
        _swap_stale_models_to_installed(chat_overridden=chat_model_overridden())
        get_services().embedder.validate_model()
        if cfg.auto_sync and not no_sync:
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

        def _on_first_token() -> None:
            nonlocal first
            if first:
                announce_ready(err, WorkerRole.CHAT)
                first = False

        _print_answer_stream(stream, on_first_token=_on_first_token)
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
    # No diagnostics routing here: chat hands off to the TUI, which owns its
    # logging (tui.log). Routing first would latch captureWarnings and leave a
    # NOTSET cli.log handler running under the TUI.

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
