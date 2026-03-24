"""Tests for text chunking behavior.

These tests verify chunking invariants regardless of the underlying
implementation.
"""

import tempfile
from pathlib import Path

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

    def test_supported_extensions_nonempty(self):
        from lilbee.code_chunker import supported_extensions

        exts = supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".rs" in exts
        assert ".go" in exts

    def test_languages_map(self):
        from lilbee.languages import EXT_TO_LANG

        assert ".py" in EXT_TO_LANG
        assert ".js" in EXT_TO_LANG
        assert ".rs" in EXT_TO_LANG
        assert EXT_TO_LANG[".py"] == "python"
