"""Wiki layer commands: build, update, lint, citations, status, prune, synthesize, drafts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.table import Table

from lilbee.app.services import get_services
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.core.security import PathTraversalError
from lilbee.wiki.shared import (
    INVALID_DRAFT_SLUG_ERROR,
    WikiSubdir,
)

if TYPE_CHECKING:
    from lilbee.wiki.entity_extractor import ExtractedEntity


wiki_app = typer.Typer(help="Wiki layer commands: generate, lint, citations, status, prune.")

# Citations table renders excerpts truncated to ``_CITATION_EXCERPT_MAX_CHARS``;
# the ellipsis insertion point is one ``...`` shorter so the visible string never
# exceeds the column width.
_CITATION_EXCERPT_MAX_CHARS = 60
_CITATION_EXCERPT_TRUNCATE_AT = 57

# Dry-run NER output previews the first ``_NER_DRY_RUN_PREVIEW_LIMIT`` sources
# per row, with ``", ..."`` appended when more were dropped.
_NER_DRY_RUN_PREVIEW_LIMIT = 3


def _count_md_files(directory: Path) -> int:
    """Count markdown files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.md")))


def _fail_wiki_disabled() -> None:
    """Emit the standard wiki-disabled message in the caller's output mode."""
    if cfg.json_mode:
        json_output({"error": msg.CMD_WIKI_DISABLED})
        return
    console.print(msg.CMD_WIKI_DISABLED)


@wiki_app.command(name="lint")
def wiki_lint(
    wiki_source: str = typer.Argument("", help="Wiki page path (empty = lint all)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Lint wiki pages for stale citations, missing sources, and unmarked claims."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.lint import lint_all as _lint_all
    from lilbee.wiki.lint import lint_wiki_page

    store = get_services().store
    if wiki_source:
        issues = lint_wiki_page(wiki_source, store)
    else:
        report = _lint_all(store)
        issues = report.issues

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_lint",
                "issues": [i.to_dict() for i in issues],
                "total": len(issues),
            }
        )
        return

    if not issues:
        console.print("No issues found.")
        return

    table = Table(title="Wiki Lint Issues")
    table.add_column("Page", style=theme.ACCENT)
    table.add_column("Severity")
    table.add_column("Message")
    for issue in issues:
        sev_style = theme.ERROR if issue.severity.value == "error" else theme.WARNING
        sev_text = f"[{sev_style}]{issue.severity.value}[/{sev_style}]"
        table.add_row(issue.wiki_source, sev_text, issue.message)
    console.print(table)


@wiki_app.command(name="citations")
def wiki_citations(
    wiki_source: str = typer.Argument(..., help="Wiki page path, e.g. wiki/summaries/doc.md."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show citations for a wiki page."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    records = get_services().store.get_citations_for_wiki(wiki_source)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_citations",
                "wiki_source": wiki_source,
                "citations": [dict(r) for r in records],
                "total": len(records),
            }
        )
        return

    if not records:
        console.print(f"No citations found for [{theme.ACCENT}]{wiki_source}[/{theme.ACCENT}]")
        return

    table = Table(title=f"Citations: {wiki_source}")
    table.add_column("Key", style=theme.ACCENT)
    table.add_column("Source")
    table.add_column("Type", style=theme.MUTED)
    table.add_column("Excerpt", max_width=_CITATION_EXCERPT_MAX_CHARS)
    for rec in records:
        excerpt = (
            rec["excerpt"][:_CITATION_EXCERPT_TRUNCATE_AT] + "..."
            if len(rec["excerpt"]) > _CITATION_EXCERPT_MAX_CHARS
            else rec["excerpt"]
        )
        table.add_row(rec["citation_key"], rec["source_filename"], rec["claim_type"], excerpt)
    console.print(table)


@wiki_app.command(name="status")
def wiki_status(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show wiki layer status: page counts and lint summary."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not wiki_root.exists():
        if cfg.json_mode:
            json_output({"wiki_enabled": cfg.wiki, "pages": 0, "issues": 0})
            return
        console.print("Wiki directory does not exist yet. Run sync with wiki enabled.")
        return

    summaries = _count_md_files(wiki_root / WikiSubdir.SUMMARIES)
    drafts = _count_md_files(wiki_root / WikiSubdir.DRAFTS)

    from lilbee.wiki.lint import lint_all as _lint_all

    report = _lint_all(get_services().store)

    if cfg.json_mode:
        json_output(
            {
                "wiki_enabled": cfg.wiki,
                WikiSubdir.SUMMARIES: summaries,
                WikiSubdir.DRAFTS: drafts,
                "pages": summaries + drafts,
                "lint_errors": report.error_count,
                "lint_warnings": report.warning_count,
            }
        )
        return

    color = "green" if cfg.wiki else "red"
    label = "enabled" if cfg.wiki else "disabled"
    console.print(f"Wiki: [{color}]{label}[/{color}]")
    console.print(f"  Summaries: [{theme.LABEL}]{summaries}[/{theme.LABEL}]")
    console.print(f"  Drafts:    [{theme.LABEL}]{drafts}[/{theme.LABEL}]")
    if report.error_count or report.warning_count:
        console.print(
            f"  Lint: [{theme.ERROR}]{report.error_count} error(s)[/{theme.ERROR}], "
            f"[{theme.WARNING}]{report.warning_count} warning(s)[/{theme.WARNING}]"
        )
    else:
        console.print("  Lint: all clean")


@wiki_app.command(name="synthesize")
def wiki_synthesize(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Generate synthesis pages for concept clusters spanning 3+ sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
        return
    from lilbee.wiki.generation import generate_synthesis_pages

    svc = get_services()
    paths = generate_synthesis_pages(svc.provider, svc.store, svc.clusterer)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_synthesize",
                "paths": [str(p) for p in paths],
                "count": len(paths),
            }
        )
        return

    if not paths:
        console.print("No synthesis pages generated (need 3+ sources per cluster).")
        return

    console.print(f"Generated [{theme.LABEL}]{len(paths)}[/{theme.LABEL}] synthesis pages:")
    for path in paths:
        console.print(f"  {path}")


@wiki_app.command(name="prune")
def wiki_prune(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Prune stale and orphaned wiki pages."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_prune",
                "records": [r.to_dict() for r in report.records],
                "archived": report.archived_count,
                "flagged": report.flagged_count,
            }
        )
        return

    if not report.records:
        console.print("No pages pruned.")
        return

    table = Table(title="Wiki Prune Results")
    table.add_column("Page", style=theme.ACCENT)
    table.add_column("Action")
    table.add_column("Reason")
    for rec in report.records:
        action_style = theme.ERROR if rec.action.value == "archived" else theme.WARNING
        action_text = f"[{action_style}]{rec.action.value}[/{action_style}]"
        table.add_row(rec.wiki_source, action_text, rec.reason)
    console.print(table)


@wiki_app.command(name="build")
def wiki_build(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Run extraction only; skip every LLM call. Prints the NER entity candidates. "
            "LLM-curated concept pages require a build call and are not shown in dry-run."
        ),
    ),
) -> None:
    """Build the concept and entity wiki across all ingested sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
        return

    if dry_run:
        from lilbee.data.store import SearchChunk
        from lilbee.wiki.entity_extractor import get_entity_extractor

        svc = get_services()
        chunks: list[SearchChunk] = []
        for record in svc.store.get_sources():
            chunks.extend(svc.store.get_chunks_by_source(record["filename"]))
        extractor = get_entity_extractor(cfg.wiki_entity_mode, svc.provider, cfg)
        entities = extractor.extract(chunks)
        _wiki_build_dry_run_output(entities)
        return

    from lilbee.wiki import run_full_build

    result = run_full_build(cfg)

    if cfg.json_mode:
        json_output({"command": "wiki_build", **result})
        return

    pages = result["paths"]
    if not pages:
        console.print("No concept or entity pages generated.")
        return

    console.print(
        f"Generated [{theme.LABEL}]{result['count']}[/{theme.LABEL}] "
        f"wiki pages from {result['entities']} extracted records:"
    )
    for path in pages:
        console.print(f"  {path}")


_DRY_RUN_CONCEPT_NOTE = (
    "Note: LLM-curated concepts are not shown in --dry-run. "
    "Run `lilbee wiki build` to see which concepts the LLM proposes."
)


def _wiki_build_dry_run_output(entities: list[ExtractedEntity]) -> None:
    """Render the extraction result as JSON or table without calling any LLM.

    Concepts come from the per-source batched LLM call, so listing
    them would require the call we are trying to avoid. The dry-run
    surface is NER-entity only, with a trailing note so a user who
    expected concepts in the output knows why they are missing.
    """
    rows: list[dict[str, Any]] = [
        {
            "slug": e.slug,
            "label": e.label,
            "kind": e.kind.value,
            "type_hint": e.type_hint,
            "mentions": len(e.chunk_refs),
            "sources": sorted({r.source for r in e.chunk_refs}),
        }
        for e in entities
    ]

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_build",
                "dry_run": True,
                "entities": rows,
                "count": len(rows),
                "note": _DRY_RUN_CONCEPT_NOTE,
            }
        )
        return

    if not rows:
        console.print("No candidate entities extracted. Run sync first.")
        console.print(f"[{theme.MUTED}]{_DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")
        return

    table = Table(title=f"Wiki build dry-run ({len(rows)} NER entity candidates)")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Type")
    table.add_column("Mentions")
    table.add_column("Sources")
    for row in rows:
        sources_list: list[str] = row["sources"]
        table.add_row(
            str(row["slug"]),
            str(row["kind"]),
            str(row["type_hint"]),
            str(row["mentions"]),
            ", ".join(sources_list[:_NER_DRY_RUN_PREVIEW_LIMIT])
            + (", ..." if len(sources_list) > _NER_DRY_RUN_PREVIEW_LIMIT else ""),
        )
    console.print(table)
    console.print(
        f"Dry run: [{theme.LABEL}]{len(rows)}[/{theme.LABEL}] candidate entities. "
        "No LLM calls were made."
    )
    console.print(f"[{theme.MUTED}]{_DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")


@wiki_app.command(name="update")
def wiki_update(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Refresh the concept and entity wiki after an ingest.

    Currently a full rebuild. The incremental touched-slug regeneration
    lands in the ingest-hook task and will re-route this command then.
    """
    wiki_build(data_dir=data_dir, use_global=use_global, dry_run=False)


drafts_app = typer.Typer(help="Review wiki drafts: list, diff, accept, reject.")
wiki_app.add_typer(drafts_app, name="drafts")


@drafts_app.command(name="list")
def wiki_drafts_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List pending wiki drafts with drift, faithfulness, and pairing info."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import PendingKind, list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_drafts_list",
                "drafts": [d.to_dict() for d in drafts],
                "total": len(drafts),
            }
        )
        return

    if not drafts:
        console.print("No drafts pending review.")
        return

    table = Table(title="Wiki Drafts")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Drift")
    table.add_column("Faithfulness")
    table.add_column("Published?", style=theme.MUTED)
    for d in drafts:
        kind = d.pending_kind or PendingKind.DRIFT
        drift = f"{d.drift_ratio:.0%}" if d.drift_ratio is not None else "-"
        faith = f"{d.faithfulness_score:.2f}" if d.faithfulness_score is not None else "-"
        published = "yes" if d.published_exists else "no"
        table.add_row(d.slug, kind, drift, faith, published)
    console.print(table)


def _draft_slug_error() -> None:
    """Report a rejected (traversal) draft slug generically, without leaking paths."""
    message = INVALID_DRAFT_SLUG_ERROR
    if cfg.json_mode:
        json_output({"error": message})
    else:
        console.print(f"[{theme.ERROR}]{message}[/{theme.ERROR}]")
    raise typer.Exit(1) from None


@drafts_app.command(name="diff")
def wiki_drafts_diff(
    slug: str = typer.Argument(..., help="Draft slug (e.g. chevrolet)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show a unified diff of the draft against its published counterpart."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_diff", "slug": slug, "diff": diff})
        return
    console.print(diff or "(no differences)")


@drafts_app.command(name="accept")
def wiki_drafts_accept(
    slug: str = typer.Argument(..., help="Draft slug to accept."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Overwrite the published page with the draft and re-index its chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import accept_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        result = accept_draft(slug, wiki_root, get_services().store)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_accept", **result.to_dict()})
        return
    console.print(
        f"Accepted [{theme.ACCENT}]{slug}[/{theme.ACCENT}] -> "
        f"{result.moved_to} ({result.reindexed_chunks} chunks re-indexed)"
    )


@drafts_app.command(name="reject")
def wiki_drafts_reject(
    slug: str = typer.Argument(..., help="Draft slug to reject."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete the draft file. Does not touch the published page or index."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import reject_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        reject_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_reject", "slug": slug})
        return
    console.print(f"Rejected [{theme.ACCENT}]{slug}[/{theme.ACCENT}]")
