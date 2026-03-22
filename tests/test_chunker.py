"""Tests for text and code chunking."""

import tempfile
from pathlib import Path

from lilbee.chunker import chunk_text


class TestChunkText:
    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("This is a short paragraph.")
        assert len(chunks) == 1
        assert "short paragraph" in chunks[0]

    def test_multiple_paragraphs_all_present(self):
        paragraphs = [f"Paragraph number {i} with some content." for i in range(50)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for p in paragraphs:
            assert any(p in c for c in chunks), f"Missing: {p}"

    def test_consecutive_chunks_overlap(self):
        paragraphs = [f"Paragraph {i} with enough words to fill space." for i in range(30)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=15)
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i + 1].split())
            assert words_a & words_b, f"No overlap between chunk {i} and {i + 1}"

    def test_long_sentence_splits(self):
        long = " ".join(["word"] * 2000)
        chunks = chunk_text(long, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1


class TestHardSplitWords:
    """Cover hard_split_words — the last-resort splitting path."""

    def test_no_separators_forces_word_split(self):
        """A string with no paragraph/sentence separators that needs splitting at word level."""
        from lilbee.chunker import hard_split_words

        # Directly test the function
        text = " ".join(["word"] * 500)
        segments = hard_split_words(text, max_tokens=20)
        assert len(segments) > 1
        # All segments non-empty
        for seg in segments:
            assert len(seg) > 0

    def test_single_long_token_no_spaces(self):
        """A string with no spaces at all forces hard_split_words via _split_to_segments."""
        # No separators at all: no \n\n, no ". ", no " "
        long_text = "a" * 5000
        chunks = chunk_text(long_text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 1


class TestTailOverlap:
    """Ensure overlap logic actually runs."""

    def test_chunks_have_overlap_content(self):
        # Many short paragraphs, small chunk size to force multiple chunks + overlap
        text = "\n\n".join([f"Para {i} has some content here." for i in range(40)])
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=15)
        assert len(chunks) > 2


class TestSegmentsEmpty:
    def test_segments_returns_empty(self):
        """Cover the `not segments` early return in chunk_text."""
        from unittest.mock import patch

        from lilbee.chunker import chunk_text

        with patch("lilbee.chunker._split_to_segments", return_value=[]):
            result = chunk_text("Some text")
            assert result == []


class TestChunkMarkdown:
    def test_splits_on_headings(self):
        from lilbee.chunker import chunk_markdown

        md = "# Intro\n\nHello world.\n\n## Details\n\nSome details here."
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert "# Intro" in chunks[0]
        assert "Hello world" in chunks[0]
        assert "## Details" in chunks[1]
        assert "Some details here" in chunks[1]

    def test_heading_hierarchy_prepended(self):
        from lilbee.chunker import chunk_markdown

        md = "# Top\n\n## Sub\n\nContent under sub."
        chunks = chunk_markdown(md)
        assert any("# Top > ## Sub" in c for c in chunks)

    def test_no_headings_falls_back(self):
        from lilbee.chunker import chunk_markdown

        md = "Just plain text without any headings."
        chunks = chunk_markdown(md)
        assert len(chunks) >= 1
        assert "Just plain text" in chunks[0]

    def test_empty_text(self):
        from lilbee.chunker import chunk_markdown

        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []

    def test_content_before_first_heading(self):
        from lilbee.chunker import chunk_markdown

        md = "Preamble text.\n\n# First Section\n\nSection body."
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert "Preamble" in chunks[0]
        assert "# First Section" in chunks[1]

    def test_heading_only_no_body_falls_back(self):
        from lilbee.chunker import chunk_markdown

        md = "# Heading Only\n\n"
        chunks = chunk_markdown(md)
        # No body text under heading — sections list is empty bodies, falls back
        assert len(chunks) >= 0  # doesn't crash

    def test_nested_headings(self):
        from lilbee.chunker import chunk_markdown

        md = "# A\n\nA body\n\n## B\n\nB body\n\n### C\n\nC body\n\n## D\n\nD body"
        chunks = chunk_markdown(md)
        # Should have 4 sections: A, B, C, D
        assert len(chunks) == 4
        # C should have full path
        c_chunk = next(c for c in chunks if "C body" in c)
        assert "# A" in c_chunk
        assert "## B" in c_chunk
        assert "### C" in c_chunk
        # D should NOT have B or C in its path (they were popped)
        d_chunk = next(c for c in chunks if "D body" in c)
        assert "# A" in d_chunk
        assert "## D" in d_chunk
        assert "## B" not in d_chunk


class TestCodeChunker:
    def test_python_function_extraction(self):
        from lilbee.code_chunker import chunk_code

        code = '''
def hello():
    """Say hello."""
    print("hello")

def goodbye(name: str) -> str:
    """Say goodbye."""
    return f"goodbye {name}"

class Greeter:
    def greet(self):
        return "hi"
'''
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_code(path)
            assert len(chunks) >= 2
            for c in chunks:
                assert c.line_start > 0
                assert c.line_end >= c.line_start
        finally:
            path.unlink()

    def test_unsupported_extension_falls_back(self):
        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("some content\n" * 100)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_code(path)
            assert len(chunks) >= 1
        finally:
            path.unlink()

    def test_supported_extensions_include_common_langs(self):
        from lilbee.code_chunker import supported_extensions

        exts = supported_extensions()
        expected = {".py", ".js", ".go", ".rs", ".ts", ".rb"}
        missing = expected - exts
        assert not missing, f"Missing extensions: {missing}"

    def test_definition_types_languages_have_extensions(self):
        """Every language in DEFINITION_TYPES must have at least one extension in EXT_TO_LANG."""
        from lilbee.languages import DEFINITION_TYPES, EXT_TO_LANG

        langs_with_ext = set(EXT_TO_LANG.values())
        missing = sorted(set(DEFINITION_TYPES) - langs_with_ext)
        assert not missing, (
            f"Languages in DEFINITION_TYPES without any EXT_TO_LANG entry: {missing}"
        )

    def test_no_definitions_falls_back(self):
        """File with no functions/classes falls back to token chunking."""
        from lilbee.code_chunker import chunk_code

        # Python file with only comments and assignments — no function/class defs
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("# Just a comment\n" * 50 + "x = 1\ny = 2\n" * 50)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_code(path)
            assert len(chunks) >= 1
        finally:
            path.unlink()

    def testget_parser_returns_none_for_bad_language(self):
        """Cover get_parser returning None for unknown language."""
        from lilbee.code_chunker import get_parser

        result = get_parser("totally_fake_lang_xyz")
        assert result is None

    def testget_parser_returns_parser_for_valid_language(self):
        """Verify get_parser returns a working parser."""
        from lilbee.code_chunker import get_parser

        parser = get_parser("python")
        assert parser is not None

    def test_no_parser_falls_back(self):
        """When parser is None (bad language), falls back to token chunking."""
        from unittest.mock import patch

        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def hello():\n    pass\n")
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.code_chunker.get_parser", return_value=None):
                chunks = chunk_code(path)
                assert len(chunks) >= 1
        finally:
            path.unlink()

    def test_ruby_ast_chunking(self):
        """New language (Ruby) uses AST chunking via tree-sitter-language-pack."""
        from lilbee.code_chunker import chunk_code

        code = """
class Greeter
  def initialize(name)
    @name = name
  end

  def greet
    "Hello, #{@name}!"
  end
end

def standalone_function
  puts "hi"
end
"""
        with tempfile.NamedTemporaryFile(suffix=".rb", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_code(path)
            assert len(chunks) >= 2
            texts = " ".join(c.chunk for c in chunks)
            assert "Greeter" in texts
            assert "standalone_function" in texts
        finally:
            path.unlink()

    def testcollect_definitions_with_container(self):
        """Cover collect_definitions container path (line 98-99)."""
        from unittest.mock import MagicMock

        from lilbee.code_chunker import collect_definitions

        # Build mock AST: root -> container_child -> definition_grandchild
        grandchild = MagicMock()
        grandchild.type = "function_definition"
        grandchild.start_byte = 0
        grandchild.end_byte = 10
        grandchild.start_point = MagicMock(row=0)
        grandchild.end_point = MagicMock(row=2)

        container = MagicMock()
        container.type = "block"  # In _CONTAINERS
        container.children = [grandchild]

        root = MagicMock()
        root.children = [container]

        source = b"def hello():\n    pass\n"
        def_types = frozenset({"function_definition"})

        results = collect_definitions(root, source, def_types)
        assert len(results) == 1

    def testfind_line_not_found(self):
        """Cover find_line returning default when needle not found."""
        from lilbee.code_chunker import find_line

        result = find_line("nonexistent", ["line1", "line2"], 0)
        assert result == 1

    def testfind_line_empty_needle(self):
        """Cover find_line with empty needle."""
        from lilbee.code_chunker import find_line

        result = find_line("", ["line1", "line2"], 0)
        assert result == 1
