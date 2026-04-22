"""Wiki structure: turn kreuzberg's DocumentStructure into a heading-only tree.

kreuzberg's ``result.document`` is a fine-grained reading-order tree where every
paragraph, list item, and table cell is its own node. Wiki generation only
cares about the heading backbone (Title / Heading / Group-with-heading). This
module walks the kreuzberg tree and emits ``WikiNode`` records describing that
backbone, preserving parent-child links and page ranges so later stages can
drive map-reduce summarization over it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# kreuzberg content types that introduce a heading boundary.
_HEADING_NODE_TYPES: frozenset[str] = frozenset({"title", "heading", "group"})

_TITLE_LEVEL = 0  # kreuzberg "title" variants become the root wiki node.
_DEFAULT_HEADING_LEVEL = 99  # group without an explicit heading_level sorts to leaves.

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]")
_SLUG_COLLAPSE_RE = re.compile(r"-+")


@dataclass(frozen=True)
class WikiNode:
    """One node of the wiki tree (chapter / section / page).

    ``slug`` is unique within a source and encodes the ancestor chain, e.g.
    ``02-brakes/03-abs``. ``parent_slug`` links back to the ancestor; a node
    with ``parent_slug=None`` is the root for its source.
    """

    slug: str
    parent_slug: str | None
    depth: int
    ordinal: int
    title: str
    page_start: int
    page_end: int
    kind: str
    kreuzberg_node_id: str


def wiki_node_to_dict(node: WikiNode) -> dict[str, Any]:
    """Serialize a WikiNode to the dict shape CLI and MCP emit to clients."""
    return {
        "slug": node.slug,
        "parent_slug": node.parent_slug,
        "depth": node.depth,
        "title": node.title,
        "kind": node.kind,
        "page_start": node.page_start,
        "page_end": node.page_end,
    }


def deserialize_document(document_json: str) -> dict[str, Any] | None:
    """Parse a persisted DocumentStructure JSON blob, returning ``None`` on failure."""
    try:
        parsed = json.loads(document_json)
    except json.JSONDecodeError:
        log.warning("Stored DocumentStructure JSON is not parseable; skipping")
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def walk_structure_to_wiki_nodes(document: dict[str, Any]) -> list[WikiNode]:
    """Filter a kreuzberg DocumentStructure down to the heading backbone.

    Returns ``WikiNode`` records in reading order with stable slugs and
    parent links. Each node's ``page_start..page_end`` range covers every
    descendant (heading or otherwise) in the kreuzberg tree, so leaf
    chunks can be matched to the enclosing heading by page number.
    """
    nodes = document.get("nodes") or []
    if not nodes:
        return []

    indexed = list(enumerate(nodes))
    out: list[WikiNode] = []
    slug_stack: list[tuple[int, str]] = []  # (level, slug) for in-progress ancestors
    sibling_ordinals: dict[str | None, int] = {}

    for idx, node in indexed:
        level = _heading_level(node)
        if level is None:
            continue
        title = _heading_title(node)
        if not title:
            continue

        while slug_stack and slug_stack[-1][0] >= level:
            slug_stack.pop()
        parent_slug = slug_stack[-1][1] if slug_stack else None
        ordinal = sibling_ordinals.get(parent_slug, 0)
        sibling_ordinals[parent_slug] = ordinal + 1
        slug = _compose_slug(title, ordinal, parent_slug)

        page_start = _node_page(node, "page") or 0
        page_end = _descendant_page_end(indexed, idx, nodes) or page_start
        kind = _classify_kind(level, parent_slug)

        out.append(
            WikiNode(
                slug=slug,
                parent_slug=parent_slug,
                depth=level,
                ordinal=ordinal,
                title=title,
                page_start=page_start,
                page_end=page_end,
                kind=kind,
                kreuzberg_node_id=str(node.get("id", "")),
            )
        )
        slug_stack.append((level, slug))

    return out


def _heading_level(node: dict[str, Any]) -> int | None:
    """Return the heading level for *node*, or None if it is not a heading."""
    content = node.get("content") or {}
    node_type = content.get("node_type")
    if node_type not in _HEADING_NODE_TYPES:
        return None
    if node_type == "title":
        return _TITLE_LEVEL
    if node_type == "heading":
        level = content.get("level")
        if isinstance(level, int):
            return level
        return _DEFAULT_HEADING_LEVEL
    # group: only counts when it carries an explicit heading_level + heading_text
    heading_level = content.get("heading_level")
    heading_text = content.get("heading_text")
    if isinstance(heading_level, int) and heading_text:
        return heading_level
    return None


def _heading_title(node: dict[str, Any]) -> str:
    content = node.get("content") or {}
    node_type = content.get("node_type")
    if node_type in {"title", "heading"}:
        return str(content.get("text") or "").strip()
    if node_type == "group":
        return str(content.get("heading_text") or "").strip()
    return ""


def _node_page(node: dict[str, Any], key: str) -> int | None:
    value = node.get(key)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _descendant_page_end(
    indexed: list[tuple[int, dict[str, Any]]],
    idx: int,
    all_nodes: list[dict[str, Any]],
) -> int:
    """Return the largest ``page`` value reachable from *idx* via child links."""
    node = all_nodes[idx]
    visited: set[int] = set()
    frontier: list[int] = [idx]
    max_page = _node_page(node, "page_end") or _node_page(node, "page") or 0
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        if current < 0 or current >= len(all_nodes):
            continue
        child = all_nodes[current]
        child_page = _node_page(child, "page_end") or _node_page(child, "page") or 0
        if child_page > max_page:
            max_page = child_page
        children = child.get("children") or []
        for child_idx in children:
            if isinstance(child_idx, int) and child_idx not in visited:
                frontier.append(child_idx)
    return max_page


def _classify_kind(level: int, parent_slug: str | None) -> str:
    """Map a heading level to a wiki-node kind label.

    ``title`` variants (``level == 0``) become ``"root"`` regardless of parenting;
    otherwise any outermost heading is a ``"chapter"`` and nested ones are
    ``"section"``. Kind is advisory metadata for renderers, not used by walking.
    """
    if level == _TITLE_LEVEL:
        return "root"
    if parent_slug is None:
        return "chapter"
    return "section"


def _compose_slug(title: str, ordinal: int, parent_slug: str | None) -> str:
    """Build a stable slug from a heading title, prefixed by its ordinal for uniqueness."""
    cleaned = title.lower().replace(" ", "-").replace("/", "--")
    cleaned = _SLUG_CLEAN_RE.sub("", cleaned)
    cleaned = _SLUG_COLLAPSE_RE.sub("-", cleaned).strip("-")
    if not cleaned:
        cleaned = "section"
    prefixed = f"{ordinal + 1:02d}-{cleaned}"
    return f"{parent_slug}/{prefixed}" if parent_slug else prefixed
