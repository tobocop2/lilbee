"""Wiki ``[[link]]`` rewriter.

Post-processing pass that rewrites concept and entity surface forms to
Obsidian-style ``[[slug]]`` links in the body of a page. Code fences,
YAML frontmatter, and the citation block are left untouched so the
rewriter can run repeatedly over the same file without corrupting
either the provenance trail or illustrative code blocks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lilbee.wiki.grammar import CITATION_BLOCK_COMMENT, CODE_FENCE_RE


@dataclass(frozen=True)
class CompiledRewriter:
    """Precompiled artifacts shared across a batch of page rewrites.

    The regex compile + longest-first sort are O(M log M) in the
    surface-map size. When a build rewrites P pages, computing these
    once and reusing them across iterations cuts the per-page cost
    to a single ``pattern.sub`` pass. The ``lookup`` is wrapped in a
    read-only view so callers can't accidentally poison a shared
    rewriter mid-loop.
    """

    pattern: re.Pattern[str]
    lookup: Mapping[str, str]


def compile_rewriter(surface_to_slug: dict[str, str]) -> CompiledRewriter | None:
    """Compile the regex + lowercase lookup for a surface-to-slug map.

    Returns ``None`` when the map is empty so the caller can short-circuit.
    """
    if not surface_to_slug:
        return None
    return CompiledRewriter(
        pattern=_compile_surface_pattern(surface_to_slug),
        lookup=MappingProxyType(
            {surface.lower(): slug for surface, slug in surface_to_slug.items()}
        ),
    )


def rewrite_wiki_links(
    content: str,
    surface_to_slug: dict[str, str],
    skip_slug: str | None = None,
) -> str:
    """Return *content* with slug surface forms rewritten to ``[[slug]]``.

    *surface_to_slug* maps the human-readable surface form (e.g.
    ``"tire pressure"``) to its slug (``"tire-pressure"``). Matching is
    case-insensitive, respects word boundaries, and skips occurrences
    already wrapped in ``[[...]]``. When two surface forms overlap
    (e.g. ``"ford"`` and ``"ford motor company"``) the longer form
    wins, since the alternation regex is ordered longest-first.

    *skip_slug* suppresses self-links: a match that resolves to this
    slug is left as raw text. Callers pass the owning page's slug so
    ``braking.md`` does not gain a ``[[braking]]`` reference to itself.
    Filtering in the replace callback is O(1) per match; pre-filtering
    the dict would be O(M) per page.

    For batch work over many pages, call :func:`compile_rewriter` once
    and pass the result to :func:`apply_rewriter` to skip the per-call
    compile + sort.
    """
    rewriter = compile_rewriter(surface_to_slug)
    if rewriter is None or not content:
        return content
    return apply_rewriter(content, rewriter, skip_slug)


def apply_rewriter(
    content: str,
    rewriter: CompiledRewriter,
    skip_slug: str | None = None,
) -> str:
    """Apply a precompiled rewriter to *content*, returning the rewritten text."""
    if not content:
        return content

    ending_newline = content.endswith("\n")
    lines = content.splitlines()
    rewritten = [
        _rewrite_line(line, rewriter.pattern, rewriter.lookup, skip_slug) if writable else line
        for line, writable in _classify_lines(lines)
    ]
    result = "\n".join(rewritten)
    if ending_newline:
        result += "\n"
    return result


def _compile_surface_pattern(surface_to_slug: dict[str, str]) -> re.Pattern[str]:
    """Compile one alternation regex ordered longest-first.

    Longest-first matters so ``"ford motor company"`` beats ``"ford"``
    when both are in the slug set. The lookbehind blocks matching
    inside an existing ``[[...]]`` link by rejecting a preceding ``[``,
    and the lookahead blocks matching inside the closing ``]]``.
    """
    sorted_surfaces = sorted(surface_to_slug, key=len, reverse=True)
    alternation = "|".join(re.escape(s) for s in sorted_surfaces if s)
    return re.compile(
        r"(?<![\w\[])(" + alternation + r")(?![\w\]])",
        re.IGNORECASE,
    )


def _rewrite_line(
    line: str,
    pattern: re.Pattern[str],
    lookup: Mapping[str, str],
    skip_slug: str | None = None,
) -> str:
    def replace(match: re.Match[str]) -> str:
        slug = lookup[match.group(0).lower()]
        if slug == skip_slug:
            return match.group(0)
        return f"[[{slug}]]"

    return pattern.sub(replace, line)


def _classify_lines(lines: list[str]) -> list[tuple[str, bool]]:
    """Tag each line with whether it's part of a rewritable body region."""
    tagged: list[tuple[str, bool]] = []
    in_frontmatter = False
    in_code_fence = False
    in_citation = False

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if idx == 0 and stripped == "---":
            in_frontmatter = True
            tagged.append((line, False))
            continue
        if in_frontmatter:
            tagged.append((line, False))
            if stripped == "---":
                in_frontmatter = False
            continue

        # The citation block is terminal: once its comment marker appears
        # every following line is citation, so ``in_citation`` never resets.
        if stripped == CITATION_BLOCK_COMMENT:
            in_citation = True
        if in_citation:
            tagged.append((line, False))
            continue

        if CODE_FENCE_RE.match(stripped):
            tagged.append((line, False))
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            tagged.append((line, False))
            continue

        tagged.append((line, True))
    return tagged
