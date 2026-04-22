"""Wiki ``[[link]]`` rewriter.

Post-processing pass that rewrites concept and entity surface forms to
Obsidian-style ``[[slug]]`` links in the body of a page. Code fences,
YAML frontmatter, and the citation block are left untouched so the
rewriter can run repeatedly over the same file without corrupting
either the provenance trail or illustrative code blocks.
"""

from __future__ import annotations

import re

from lilbee.wiki.shared import CITATION_BLOCK_COMMENT

_CODE_FENCE_PREFIX = "```"


def rewrite_wiki_links(content: str, surface_to_slug: dict[str, str]) -> str:
    """Return *content* with slug surface forms rewritten to ``[[slug]]``.

    *surface_to_slug* maps the human-readable surface form (e.g.
    ``"tire pressure"``) to its slug (``"tire-pressure"``). Matching is
    case-insensitive, respects word boundaries, and skips occurrences
    already wrapped in ``[[...]]``. When two surface forms overlap
    (e.g. ``"ford"`` and ``"ford motor company"``) the longer form
    wins, since the alternation regex is ordered longest-first.
    """
    if not surface_to_slug or not content:
        return content

    pattern = _compile_surface_pattern(surface_to_slug)
    lookup = {surface.lower(): slug for surface, slug in surface_to_slug.items()}

    ending_newline = content.endswith("\n")
    lines = content.splitlines()
    rewritten = [
        _rewrite_line(line, pattern, lookup) if writable else line
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
    lookup: dict[str, str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"[[{lookup[match.group(0).lower()]}]]"

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

        if stripped.startswith(_CODE_FENCE_PREFIX):
            tagged.append((line, False))
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            tagged.append((line, False))
            continue

        tagged.append((line, True))
    return tagged
