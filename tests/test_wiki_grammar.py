"""Tests for the wiki markdown grammar constants.

These tests pin the shape of the patterns that the wiki contract
depends on. If a change here fails a test, every wiki page written
under the previous grammar will break on re-parse, so the test is a
gate against accidental drift.
"""

from __future__ import annotations

from lilbee.wiki import grammar


class TestStructuralDelimiters:
    def test_citation_block_sep_is_triple_dash(self) -> None:
        assert grammar.CITATION_BLOCK_SEP == "---"

    def test_citation_block_comment_matches_writer_output(self) -> None:
        assert grammar.CITATION_BLOCK_COMMENT.startswith("<!--")
        assert grammar.CITATION_BLOCK_COMMENT.endswith("-->")
        assert "auto-generated" in grammar.CITATION_BLOCK_COMMENT

    def test_code_fence_prefix_is_triple_backtick(self) -> None:
        assert grammar.CODE_FENCE_PREFIX == "```"


class TestCitationPatterns:
    def test_cite_re_matches_inline_ref(self) -> None:
        match = grammar.CITE_RE.search("see [^src3] for details")
        assert match is not None
        assert match.group(1) == "src3"

    def test_cite_re_rejects_non_src_prefix(self) -> None:
        assert grammar.CITE_RE.search("[^note1]") is None

    def test_footnote_re_matches_definition_line(self) -> None:
        text = "[^src1]: python-docs/typing.md, lines 12-45"
        match = grammar.FOOTNOTE_RE.match(text)
        assert match is not None
        assert match.group(1) == "src1"
        assert match.group(2) == "python-docs/typing.md, lines 12-45"

    def test_footnote_re_is_multiline(self) -> None:
        text = "body line\n[^src2]: source.md\nmore body"
        matches = list(grammar.FOOTNOTE_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "src2"

    def test_inference_re_matches_marker(self) -> None:
        assert grammar.INFERENCE_RE.search("this is [*inference*] text") is not None

    def test_inference_re_rejects_plain_text(self) -> None:
        assert grammar.INFERENCE_RE.search("this is inference text") is None


class TestLinkAndStructurePatterns:
    def test_wiki_link_re_captures_slug(self) -> None:
        match = grammar.WIKI_LINK_RE.search("see [[tire-pressure]] above")
        assert match is not None
        assert match.group(1) == "tire-pressure"

    def test_wiki_link_re_rejects_single_brackets(self) -> None:
        assert grammar.WIKI_LINK_RE.search("[not-a-link]") is None

    def test_code_fence_re_matches_backtick_and_tilde(self) -> None:
        assert grammar.CODE_FENCE_RE.match("```python") is not None
        assert grammar.CODE_FENCE_RE.match("~~~") is not None

    def test_code_fence_re_rejects_prose(self) -> None:
        assert grammar.CODE_FENCE_RE.match("normal paragraph") is None

    def test_h1_re_matches_heading_and_captures_title(self) -> None:
        match = grammar.H1_RE.match("# My Title")
        assert match is not None
        assert match.group(1) == "My Title"

    def test_h1_re_strips_trailing_hashes(self) -> None:
        match = grammar.H1_RE.match("# My Title ##")
        assert match is not None
        assert match.group(1) == "My Title"

    def test_h1_re_rejects_h2(self) -> None:
        assert grammar.H1_RE.match("## Subheading") is None
