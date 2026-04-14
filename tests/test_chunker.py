"""Tests for text chunking behavior.

These tests verify chunking invariants regardless of the underlying
implementation.
"""

import tempfile
from pathlib import Path

import pytest

from lilbee.chunk import chunk_text


class TestChunkText:
    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("This is a short paragraph.")
        assert len(chunks) >= 1
        assert "short paragraph" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        paragraphs = [f"Paragraph number {i} with detailed content here." for i in range(50)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_multiple_paragraphs_all_present(self):
        paragraphs = [f"Unique paragraph {i} with specific content." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text)
        joined = " ".join(chunks)
        for p in paragraphs:
            assert p in joined, f"Missing: {p}"

    def test_long_sentence_splits(self):
        sentence = "word " * 500
        chunks = chunk_text(sentence)
        assert len(chunks) >= 1

    def test_plain_text_no_heading_context(self):
        text = "Just plain text without any markdown headings."
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        assert "plain text" in chunks[0]


class TestMarkdownChunking:
    def test_splits_on_headings(self):
        md = (
            "# Intro\n\nHello world paragraph with enough text.\n\n"
            "## Details\n\nSome details here with more content."
        )
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert len(chunks) >= 1

    def test_heading_hierarchy_prepended(self):
        md = "# Top\n\nTop content here with text.\n\n## Sub\n\nContent under sub section."
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert any("Top" in c and "Sub" in c for c in chunks)

    def test_nested_headings(self):
        md = (
            "# A\n\nA body text here.\n\n"
            "## B\n\nB body text here.\n\n"
            "### C\n\nC body text here.\n\n"
            "## D\n\nD body text here."
        )
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "A" in joined
        assert "D" in joined

    def test_content_before_first_heading(self):
        md = "Preamble text content.\n\n# First Section\n\nSection body content."
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Preamble" in joined
        assert "Section body" in joined

    def test_empty_markdown(self):
        assert chunk_text("", mime_type="text/markdown", heading_context=True) == []


@pytest.mark.xdist_group("tree_sitter")
class TestCodeChunker:
    """Tree-sitter code chunker tests — grouped to avoid fork-unsafe C parser collisions."""

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
            assert len(chunks) >= 1
            joined = "\n".join(c.chunk for c in chunks)
            assert "hello" in joined
        finally:
            path.unlink()

    def test_unsupported_extension_returns_fallback(self):
        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".xyz_unsupported", mode="w", delete=False) as f:
            f.write("some content here")
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_code(path)
            assert isinstance(chunks, list)
        finally:
            path.unlink()

    def test_is_code_file_common_extensions(self):
        from lilbee.code_chunker import is_code_file

        assert is_code_file(Path("main.py"))
        assert is_code_file(Path("app.js"))
        assert is_code_file(Path("lib.rs"))
        assert is_code_file(Path("server.go"))

    def test_is_code_file_non_code(self):
        from lilbee.code_chunker import is_code_file

        assert not is_code_file(Path("photo.png"))
        assert not is_code_file(Path("document.pdf"))

    def test_detect_language_python(self):
        from lilbee.code_chunker import _detect_language

        result = _detect_language(Path("main.py"))
        assert result is not None
        assert "python" in result.lower()

    def test_ensure_language_exception_returns_false(self):
        from unittest.mock import patch

        from lilbee.code_chunker import _ensure_language

        with patch("lilbee.code_chunker.has_language", side_effect=RuntimeError("boom")):
            assert _ensure_language("python") is False

    def test_find_line_no_match_returns_start(self):
        from lilbee.code_chunker import find_line

        lines = ["aaa", "bbb", "ccc"]
        assert find_line("zzz", lines, 0) == 1

    def test_extract_symbols_non_list_structure(self):
        from lilbee.code_chunker import _extract_symbols

        assert _extract_symbols({"structure": "not a list"}, "code") == []

    def test_extract_symbols_non_dict_entry(self):
        from lilbee.code_chunker import _extract_symbols

        result = {"structure": ["not a dict", {"name": "fn", "kind": "function", "span": {}}]}
        symbols = _extract_symbols(result, "code")
        assert len(symbols) == 1
        assert symbols[0].name == "fn"

    def test_ensure_language_false_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.code_chunker._ensure_language", return_value=False):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
        finally:
            path.unlink()

    def test_process_exception_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.code_chunker.process", side_effect=RuntimeError("parse fail")):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
        finally:
            path.unlink()

    def test_empty_symbols_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.code_chunker._extract_symbols", return_value=[]):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
        finally:
            path.unlink()


class TestHeadingContextNoDuplicate:
    def test_heading_context_no_duplicate(self):
        """kreuzberg >= 4.8.5 should not duplicate headings with prepend_heading_context."""
        md = "# Title\n\n" + "Word " * 500 + "\n\n## Section\n\n" + "More " * 500
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        for c in chunks:
            parts = c.split("\n\n", 2)
            if len(parts) >= 2:
                ctx_last = parts[0].rsplit(" > ", 1)[-1].strip()
                assert parts[1].strip() != ctx_last, f"Duplicate heading in chunk: {c[:100]}"


class TestChunkTextEmptyResult:
    def test_returns_empty_when_no_chunks(self):
        from unittest.mock import MagicMock, patch

        from lilbee.chunk import chunk_text

        mock_result = MagicMock()
        mock_result.chunks = []
        with patch("kreuzberg.extract_bytes_sync", return_value=mock_result):
            assert chunk_text("some text") == []
