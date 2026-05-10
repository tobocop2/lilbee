"""Row-grouping helpers and the GridSection container for CatalogScreen."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.catalog.types import ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.model_list import ModelListSection


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


def for_you_sort_key(row: LocalCatalogRow) -> tuple[int, str]:
    """Rank Discover 'For You' rows: best fit first, then alphabetical.

    Fit rank: FITS=0, TIGHT=1, WONT_RUN=2, no chip=3. Featured-only
    callers already filtered, so featured isn't in the key.
    """
    from lilbee.runtime.hardware import FitLevel

    if row.fit is None:
        rank = 3
    elif row.fit.level is FitLevel.FITS:
        rank = 0
    elif row.fit.level is FitLevel.TIGHT:
        rank = 1
    else:
        rank = 2
    return (rank, row.name.lower())


def group_frontier_rows(
    frontier_rows: list[FrontierCatalogRow],
) -> list[ModelListSection]:
    """Group frontier rows into provider-headed sections, alphabetical within."""
    if not frontier_rows:
        return []
    per_provider: dict[str, list[FrontierCatalogRow]] = {}
    for row in frontier_rows:
        per_provider.setdefault(row.provider, []).append(row)
    sections: list[ModelListSection] = []
    for provider in sorted(per_provider):
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
        bucket.sort(key=lambda r: not getattr(r, "featured", False))
    for bucket in extras.values():
        bucket.sort(key=lambda r: not getattr(r, "featured", False))
    return [
        GridSection(msg.HEADING_INSTALLED, installed),
        *[GridSection(task.capitalize(), by_task[task]) for task in TASK_BUCKET_ORDER],
        *[GridSection(task.capitalize(), extras[task]) for task in extras],
    ]
