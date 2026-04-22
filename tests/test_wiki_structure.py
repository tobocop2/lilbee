"""Tests for wiki/structure.py: turning kreuzberg's DocumentStructure into wiki nodes."""

from __future__ import annotations

import json

from lilbee.wiki.structure import (
    WikiNode,
    _heading_title,
    deserialize_document,
    walk_structure_to_wiki_nodes,
)


def _heading(level: int, text: str, page: int = 1, children: list[int] | None = None) -> dict:
    return {
        "id": f"node-{text.replace(' ', '-').lower()}",
        "content": {"node_type": "heading", "level": level, "text": text},
        "page": page,
        "children": children or [],
    }


def _title(text: str, page: int = 1, children: list[int] | None = None) -> dict:
    return {
        "id": f"title-{text.replace(' ', '-').lower()}",
        "content": {"node_type": "title", "text": text},
        "page": page,
        "children": children or [],
    }


def _paragraph(text: str, page: int = 1) -> dict:
    return {
        "id": f"p-{abs(hash(text)) % 10000}",
        "content": {"node_type": "paragraph", "text": text},
        "page": page,
        "children": [],
    }


def _group(text: str, level: int, page: int = 1, children: list[int] | None = None) -> dict:
    return {
        "id": f"g-{text.replace(' ', '-').lower()}",
        "content": {
            "node_type": "group",
            "heading_level": level,
            "heading_text": text,
        },
        "page": page,
        "children": children or [],
    }


class TestDeserializeDocument:
    def test_parses_valid_json(self):
        raw = json.dumps({"nodes": []})
        assert deserialize_document(raw) == {"nodes": []}

    def test_returns_none_on_invalid_json(self):
        assert deserialize_document("{not json") is None

    def test_returns_none_when_not_a_dict(self):
        assert deserialize_document("[1, 2, 3]") is None


class TestWalkStructure:
    def test_empty_document_returns_empty(self):
        assert walk_structure_to_wiki_nodes({"nodes": []}) == []

    def test_missing_nodes_key_returns_empty(self):
        assert walk_structure_to_wiki_nodes({}) == []

    def test_single_heading_becomes_chapter(self):
        doc = {"nodes": [_heading(1, "Introduction", page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        node = result[0]
        assert isinstance(node, WikiNode)
        assert node.title == "Introduction"
        assert node.slug == "01-introduction"
        assert node.parent_slug is None
        assert node.depth == 1
        assert node.kind == "chapter"
        assert node.page_start == 1

    def test_title_variant_becomes_root_kind(self):
        doc = {"nodes": [_title("Top", page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].kind == "root"
        assert result[0].depth == 0
        assert result[0].parent_slug is None

    def test_nested_headings_create_parent_links(self):
        doc = {
            "nodes": [
                _heading(1, "Chapter A", page=1),
                _heading(2, "Section One", page=2),
                _heading(2, "Section Two", page=3),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 3
        chapter, s1, s2 = result
        assert chapter.parent_slug is None
        assert s1.parent_slug == chapter.slug
        assert s2.parent_slug == chapter.slug
        assert s1.slug == "01-chapter-a/01-section-one"
        assert s2.slug == "01-chapter-a/02-section-two"
        assert s1.kind == "section"
        assert s2.kind == "section"

    def test_heading_without_text_is_skipped(self):
        doc = {
            "nodes": [
                {
                    "id": "x",
                    "content": {"node_type": "heading", "level": 1, "text": "   "},
                    "page": 1,
                    "children": [],
                },
                _heading(1, "Real", page=2),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].title == "Real"

    def test_group_with_heading_info_counts_as_heading(self):
        doc = {"nodes": [_group("Intro Group", level=1, page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].title == "Intro Group"
        assert result[0].depth == 1

    def test_group_without_heading_info_is_skipped(self):
        doc = {
            "nodes": [
                {
                    "id": "x",
                    "content": {"node_type": "group"},
                    "page": 1,
                    "children": [],
                },
                _heading(1, "Real", page=2),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].title == "Real"

    def test_paragraphs_are_not_headings(self):
        doc = {
            "nodes": [
                _heading(1, "Chapter", page=1),
                _paragraph("Body text.", page=1),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1

    def test_depth_pops_when_new_sibling_matches_ancestor_level(self):
        """Two level-1 headings should be siblings, not parent and child."""
        doc = {
            "nodes": [
                _heading(1, "First", page=1),
                _heading(2, "One A", page=2),
                _heading(1, "Second", page=3),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        parents = [n.parent_slug for n in result]
        assert parents == [None, result[0].slug, None]

    def test_page_end_covers_descendants(self):
        """A chapter's page_end should span any deeper-numbered descendant pages."""
        doc = {
            "nodes": [
                _heading(1, "Chapter", page=1, children=[1, 2]),
                _paragraph("Body on page 3.", page=3),
                _paragraph("Body on page 7.", page=7),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert result[0].page_end == 7

    def test_kreuzberg_node_id_is_preserved(self):
        doc = {"nodes": [_heading(1, "Chapter", page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        assert result[0].kreuzberg_node_id == "node-chapter"

    def test_slug_strips_non_alphanumeric(self):
        doc = {"nodes": [_heading(1, "What's Next: 2026!", page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        # Non-alphanumeric removed, spaces/punctuation collapse to single dashes.
        assert result[0].slug == "01-whats-next-2026"

    def test_heading_without_level_sorts_as_leaf(self):
        """A heading with no level hint lands deep in the tree, not at the top."""
        doc = {
            "nodes": [
                _heading(1, "Chapter", page=1),
                {
                    "id": "x",
                    "content": {"node_type": "heading", "text": "Mystery heading"},
                    "page": 2,
                    "children": [],
                },
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 2
        assert result[1].parent_slug == result[0].slug

    def test_duplicate_titles_get_distinct_ordinals(self):
        """Two headings with the same text must still produce distinct slugs."""
        doc = {
            "nodes": [
                _heading(1, "Chapter", page=1),
                _heading(1, "Chapter", page=5),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert result[0].slug == "01-chapter"
        assert result[1].slug == "02-chapter"

    def test_empty_slug_falls_back_to_section(self):
        """Headings whose text is only punctuation still produce a valid slug."""
        doc = {"nodes": [_heading(1, "!!!", page=1)]}
        result = walk_structure_to_wiki_nodes(doc)
        assert result[0].slug == "01-section"

    def test_circular_child_reference_does_not_loop(self):
        """A malformed tree with a self-referential child index must terminate."""
        doc = {
            "nodes": [
                {
                    "id": "h1",
                    "content": {"node_type": "heading", "level": 1, "text": "Bad"},
                    "page": 1,
                    "page_end": 1,
                    "children": [0],
                }
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].page_end >= 1

    def test_shared_descendant_visited_only_once(self):
        """A DAG where two paths reach the same descendant shouldn't re-visit it."""
        doc = {
            "nodes": [
                {
                    "id": "h1",
                    "content": {"node_type": "heading", "level": 1, "text": "Root"},
                    "page": 1,
                    "children": [1, 2],
                },
                {
                    "id": "p1",
                    "content": {"node_type": "paragraph", "text": "A"},
                    "page": 2,
                    "children": [3],
                },
                {
                    "id": "p2",
                    "content": {"node_type": "paragraph", "text": "B"},
                    "page": 3,
                    "children": [3],
                },
                {
                    "id": "p3",
                    "content": {"node_type": "paragraph", "text": "Shared"},
                    "page": 9,
                    "children": [],
                },
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert result[0].page_end == 9

    def test_child_index_out_of_bounds_is_ignored(self):
        """A child index pointing past the nodes array must not raise."""
        doc = {
            "nodes": [
                {
                    "id": "h1",
                    "content": {"node_type": "heading", "level": 1, "text": "Root"},
                    "page": 1,
                    "children": [999],
                }
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].page_end == 1

    def test_unknown_content_type_produces_empty_title(self):
        """Unknown node_type values should not produce wiki nodes (empty title)."""
        doc = {
            "nodes": [
                {
                    "id": "odd",
                    "content": {"node_type": "page_break"},
                    "page": 1,
                    "children": [],
                },
                _heading(1, "Real", page=2),
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].title == "Real"

    def test_duplicate_child_index_skipped_on_revisit(self):
        """A children list containing the same index twice must not revisit it."""
        doc = {
            "nodes": [
                {
                    "id": "h1",
                    "content": {"node_type": "heading", "level": 1, "text": "Chapter"},
                    "page": 1,
                    "children": [1, 1],
                },
                {
                    "id": "p",
                    "content": {"node_type": "paragraph", "text": "Body"},
                    "page": 5,
                    "children": [],
                },
            ]
        }
        result = walk_structure_to_wiki_nodes(doc)
        assert len(result) == 1
        assert result[0].page_end == 5


class TestHeadingTitleDirect:
    """Cover the defensive fallback in _heading_title for unknown content types."""

    def test_unknown_node_type_returns_empty_string(self):
        node = {"content": {"node_type": "not-a-real-type"}}
        assert _heading_title(node) == ""

    def test_title_without_text_returns_empty_string(self):
        node = {"content": {"node_type": "title"}}
        assert _heading_title(node) == ""
