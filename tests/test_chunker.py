"""Tests for text and code chunking."""

import tempfile
from pathlib import Path

from lilbee.chunker import _CHUNK_POSITION_PREFIX_LEN, chunk_pages, chunk_text


class TestChunkPositionPrefixLen:
    def test_constant_exists_and_is_positive(self):
        assert isinstance(_CHUNK_POSITION_PREFIX_LEN, int)
        assert _CHUNK_POSITION_PREFIX_LEN > 0


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


class TestChunkPages:
    def test_empty_pages(self):
        assert chunk_pages([]) == []

    def test_single_page(self):
        pages = [{"page": 1, "text": "Some content on page one. It has details."}]
        result = chunk_pages(pages)
        assert len(result) >= 1
        assert result[0].page_start == 1
        assert result[0].page_end == 1

    def test_multi_page_valid_ranges(self):
        pages = [
            {"page": 1, "text": "Page one content.\n\n"},
            {"page": 2, "text": "Page two content.\n\n"},
            {"page": 3, "text": "Page three content.\n\n"},
        ]
        result = chunk_pages(pages)
        assert len(result) >= 1
        for r in result:
            assert 1 <= r.page_start <= r.page_end <= 3


class TestHardSplitWords:
    """Cover _hard_split_words — the last-resort splitting path."""

    def test_no_separators_forces_word_split(self):
        """A string with no paragraph/sentence separators that needs splitting at word level."""
        from lilbee.chunker import _hard_split_words

        # Directly test the function
        text = " ".join(["word"] * 500)
        segments = _hard_split_words(text, max_tokens=20)
        assert len(segments) > 1
        # All segments non-empty
        for seg in segments:
            assert len(seg) > 0

    def test_single_long_token_no_spaces(self):
        """A string with no spaces at all forces _hard_split_words via _split_to_segments."""
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


class TestChunkPagesEdge:
    def test_chunk_not_found_in_text(self):
        """Cover the pos == -1 fallback branch in chunk_pages."""
        from unittest.mock import patch

        from lilbee.chunker import chunk_pages

        pages = [{"page": 1, "text": "Hello world. Some content here."}]
        # Mock chunk_text to return a chunk that doesn't exist in the full text
        with patch("lilbee.chunker.chunk_text", return_value=["NONEXISTENT_CHUNK"]):
            result = chunk_pages(pages)
            assert len(result) == 1
            assert result[0].page_start >= 1


class TestPagesForRange:
    def test_no_overlap_falls_back_to_first_page(self):
        """When char range doesn't overlap any boundary, return first page."""
        from lilbee.chunker import _pages_for_range

        boundaries = [(0, 100, 1), (100, 200, 2)]
        # Range entirely outside all boundaries
        ps, pe = _pages_for_range(300, 400, boundaries)
        assert ps == 1
        assert pe == 1


class TestChunkPagesTracking:
    def test_chunks_from_later_pages_have_correct_page_start(self):
        """Chunks from page 200 should not have page_start=1."""
        from lilbee.chunker import chunk_pages

        header = "##### **Maintenance and Specifications**\n\n"
        pages = [
            {"page": 1, "text": header + "Introduction content " * 50},
            {"page": 2, "text": header + "Chapter two content " * 50},
            {"page": 100, "text": header + "Unique page 100 content " * 50},
            {"page": 200, "text": header + "Unique page 200 content " * 50},
        ]
        chunks = chunk_pages(pages)
        page_200_chunks = [c for c in chunks if "page 200 content" in c.chunk]
        assert page_200_chunks, "Should have chunks from page 200"
        for c in page_200_chunks:
            assert c.page_start >= 200, f"Expected page_start >= 200, got {c.page_start}"


class TestSegmentsEmpty:
    def test_segments_returns_empty(self):
        """Cover the `not segments` early return in chunk_text (line 97)."""
        from unittest.mock import patch

        from lilbee.chunker import chunk_text

        with patch("lilbee.chunker._split_to_segments", return_value=[]):
            result = chunk_text("Some text")
            assert result == []


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
        assert ".py" in exts
        assert ".js" in exts
        assert ".go" in exts

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

    def test_load_language_failure(self):
        """Cover _load_language returning None on bad module (exception path)."""
        from lilbee.code_chunker import _load_language

        result = _load_language("nonexistent_module_xyz", "fake")
        assert result is None

    def test_load_language_no_function(self):
        """Cover _load_language returning None when module has no language function."""
        from unittest.mock import MagicMock, patch

        from lilbee.code_chunker import _load_language

        mock_mod = MagicMock(spec=[])  # Module with no attributes
        with patch("importlib.import_module", return_value=mock_mod):
            result = _load_language("fake_module", "fake")
            assert result is None

    def test_get_parser_returns_none_for_bad_language(self):
        """Cover _get_parser returning None when _load_language fails."""
        from lilbee.code_chunker import _get_parser, _parsers

        # Use a key not in cache
        cache_key = "totally_fake_lang_xyz"
        _parsers.pop(cache_key, None)
        result = _get_parser("nonexistent_module", cache_key)
        assert result is None

    def test_parser_cache_hit(self):
        """Second call for same language uses cache."""
        from lilbee.code_chunker import _get_parser, _parsers

        # First call populates cache
        p1 = _get_parser("tree_sitter_python", "python")
        assert p1 is not None
        assert "python" in _parsers
        # Second call hits cache
        p2 = _get_parser("tree_sitter_python", "python")
        assert p2 is p1

    def test_no_parser_falls_back(self):
        """When parser is None (bad language), falls back to token chunking."""
        from unittest.mock import patch

        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def hello():\n    pass\n")
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.code_chunker._get_parser", return_value=None):
                chunks = chunk_code(path)
                assert len(chunks) >= 1
        finally:
            path.unlink()

    def test_collect_definitions_with_container(self):
        """Cover _collect_definitions container path (line 98-99)."""
        from unittest.mock import MagicMock

        from lilbee.code_chunker import _collect_definitions

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

        results = _collect_definitions(root, source, def_types)
        assert len(results) == 1

    def test_find_line_not_found(self):
        """Cover _find_line returning default when needle not found."""
        from lilbee.code_chunker import _find_line

        result = _find_line("nonexistent", ["line1", "line2"], 0)
        assert result == 1

    def test_find_line_empty_needle(self):
        """Cover _find_line with empty needle."""
        from lilbee.code_chunker import _find_line

        result = _find_line("", ["line1", "line2"], 0)
        assert result == 1
