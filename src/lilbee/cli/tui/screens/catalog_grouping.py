"""Row-grouping helpers and the GridSection container for CatalogScreen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.model_list import ModelListSection
from lilbee.runtime.hardware import FitLevel


@dataclass
class GridSection:
    """A named group of rows for the grid view."""

    heading: str
    rows: list[CatalogRow]


TASK_BUCKET_ORDER = (ModelTask.CHAT, ModelTask.EMBEDDING, ModelTask.VISION, ModelTask.RERANK)
PICKS_SECTION_HEADING = "★ Picks"


def row_cache_signature(row: CatalogRow) -> tuple[str, bool]:
    """Pair (name, installed-flag) for the per-tab cache key.

    Frontier rows don't carry an ``installed`` field; they're keyed as
    if installed=False since each frontier entry is provider-managed
    rather than on-disk.
    """
    if row.kind == CatalogRowKind.FRONTIER:
        return (row.name, False)
    return (row.name, row.installed)


def _is_runnable_pick(row: LocalCatalogRow) -> bool:
    """Whether the engine supports the row and the machine can hold it.

    Rows with no fit chip are excluded: an unknown size cannot be promised.
    """
    return (
        row.compat is ModelCompat.SUPPORTED
        and row.fit is not None
        and row.fit.level is not FitLevel.WONT_RUN
    )


def _backfill_sort_key(row: LocalCatalogRow) -> tuple[int, str]:
    """Rank a backfilled Discover pick: most downloaded first, then alphabetical.

    Every candidate already runs here, so popularity separates them rather
    than fit.
    """
    return (-row.sort_downloads, row.name.lower())


def for_you_by_role(rows: list[LocalCatalogRow]) -> list[LocalCatalogRow]:
    """Runnable picks grouped by role: chat, embedding, vision, rerank.

    Featured rows lead, best fit first. A role whose featured rows cannot run
    backfills with the most downloaded row that does, so a card that does not
    fit is replaced rather than dropped. A role with nothing runnable yields
    no pick.
    """
    runnable = [r for r in rows if _is_runnable_pick(r)]
    out: list[LocalCatalogRow] = []
    for task in TASK_BUCKET_ORDER:
        candidates = [r for r in runnable if r.task == task]
        featured = [r for r in candidates if r.featured]
        if featured:
            out.append(min(featured, key=for_you_sort_key))
        elif candidates:
            out.append(min(candidates, key=_backfill_sort_key))
    return out


def for_you_sort_key(row: LocalCatalogRow) -> tuple[int, str]:
    """Rank Discover 'For You' rows: best fit first, unknown fit last, then alphabetical.

    Curation is applied before the sort, so featured isn't in the key.
    """
    from lilbee.runtime.hardware import FIT_RANK

    unknown_fit_rank = len(FIT_RANK)
    rank = unknown_fit_rank if row.fit is None else FIT_RANK[row.fit.level]
    return (rank, row.name.lower())


def group_frontier_rows(
    frontier_rows: list[FrontierCatalogRow],
) -> list[ModelListSection]:
    """Group frontier rows into provider-headed sections.

    Section order follows :data:`PROVIDER_KEYS` (the canonical display
    order); providers absent from PROVIDER_KEYS land at the tail in
    alphabetical order. Rows within each section are alphabetical.
    """
    if not frontier_rows:
        return []
    from lilbee.providers.sdk_backend import PROVIDER_KEYS

    per_provider: dict[str, list[FrontierCatalogRow]] = {}
    for row in frontier_rows:
        per_provider.setdefault(row.provider, []).append(row)
    canonical_order = [label for _, _, _, label in PROVIDER_KEYS]
    ordered = [p for p in canonical_order if p in per_provider]
    extras = sorted(set(per_provider) - set(canonical_order))
    sections: list[ModelListSection] = []
    for provider in [*ordered, *extras]:
        rows = sorted(per_provider[provider], key=lambda r: r.name.lower())
        sections.append(ModelListSection(heading=provider, rows=list(rows)))
    return sections


def group_task_rows_with_picks(
    task_rows: list[LocalCatalogRow], task_label: str
) -> list[GridSection]:
    """Per-tab grouping: ★ Picks pinned, then Installed, then the rest.

    Lifts featured rows out of their task bucket into a dedicated pinned
    section at the top of the tab. Today's behavior interleaved them at
    the top of the task bucket; the redesign treats curation as its own
    layer so the eye lands on Picks first instead of having to scan past
    them to find non-featured rows.

    Pre-condition: caller has already filtered ``task_rows`` to a single
    task (the active per-task tab).
    """
    picks: list[CatalogRow] = []
    installed: list[CatalogRow] = []
    others: list[CatalogRow] = []
    for row in task_rows:
        if row.featured:
            picks.append(row)
        elif row.installed:
            installed.append(row)
        else:
            others.append(row)
    return [
        GridSection(PICKS_SECTION_HEADING, picks),
        GridSection(msg.HEADING_INSTALLED, installed),
        GridSection(task_label, others),
    ]


def flatten_sections(sections: list[GridSection], heading: str) -> list[GridSection]:
    """Collapse *sections* into a single section, preserving row order.

    Used while a search filter is active: every mounted section costs a heading
    plus a whole card row even when it holds one match.
    """
    rows = [row for section in sections for row in section.rows]
    if not rows:
        return []
    return [GridSection(heading, rows)]


def group_rows_for_grid(local_rows: list[LocalCatalogRow]) -> list[GridSection]:
    """Group local rows into sections for the grid view.

    Layout: Installed first, then one section per task. Featured rows live
    at the top of their task section (recognizable by the ``pick`` pill);
    no separate "Our picks" bucket so the catalog reads as a single
    task-organized list.
    """
    installed: list[CatalogRow] = []
    by_task: dict[str, list[CatalogRow]] = {task: [] for task in TASK_BUCKET_ORDER}
    extras: dict[str, list[CatalogRow]] = {}
    for row in local_rows:
        if row.installed:
            installed.append(row)
            continue
        bucket = by_task.get(row.task)
        if bucket is not None:
            bucket.append(row)
        else:
            extras.setdefault(row.task, []).append(row)
    # Within each task bucket: featured first (preserving their input order),
    # then the rest in their incoming order. Stable so HF rank from the API
    # is preserved among non-featured rows.
    for bucket in by_task.values():
        bucket.sort(key=lambda r: not cast("LocalCatalogRow", r).featured)
    for bucket in extras.values():
        bucket.sort(key=lambda r: not cast("LocalCatalogRow", r).featured)
    return [
        GridSection(msg.HEADING_INSTALLED, installed),
        *[GridSection(task.capitalize(), by_task[task]) for task in TASK_BUCKET_ORDER],
        *[GridSection(task.capitalize(), extras[task]) for task in extras],
    ]
