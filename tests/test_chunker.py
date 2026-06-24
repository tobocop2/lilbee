"""Tests for text chunking behavior.

These tests verify chunking invariants regardless of the underlying
implementation.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lilbee.data.chunk import chunk_text


@dataclass
class _FakeMeta:
    """Stand-in for tree-sitter ChunkContext.metadata in chunk_code tests."""

    symbols_defined: list[str] = field(default_factory=list)


@dataclass
class _FakeTSChunk:
    """Stand-in for a tree-sitter result.chunks entry (size-bounded CodeChunk)."""

    content: str
    start_line: int
    end_line: int
    symbols: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> _FakeMeta:
        return _FakeMeta(symbols_defined=self.symbols)


@dataclass
class _FakeResult:
    """Stand-in for tree-sitter ProcessResult exposing only .chunks."""

    chunks: list[_FakeTSChunk]


class TestChunkText:
    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("This is a short paragraph.")
        assert len(chunks) >= 1
        assert "short paragraph" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        """Multi-topic input forces topic breaks so semantic can't merge into one chunk."""
        topics = [
            "Solar panels convert sunlight into electricity via photovoltaic cells.",
            "The FDA approved a new clinical trial for a diabetes treatment.",
            "Quantum computers use qubits and entanglement for parallel computation.",
            "Ancient Roman aqueducts used gravity to transport water across cities.",
        ]
        paragraphs = [topics[i % len(topics)] + f" Variant {i}." for i in range(60)]
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

    def test_semantic_disabled_uses_char_budget(self, monkeypatch):
        """When cfg.semantic_chunking is False, chunker falls back to fixed char budget."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        chunks = chunk_text("Plain text chunked without the semantic branch.")
        assert chunks
        assert "Plain text" in " ".join(chunks)

    def test_use_semantic_false_bypasses_semantic(self, monkeypatch):
        """Caller can opt out of semantic chunking even when cfg has it enabled."""
        from lilbee.core.config import cfg
        from lilbee.data.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        bypassed = build_chunking_config(use_semantic=False)
        assert bypassed.chunker_type == "text"
        enabled = build_chunking_config()
        assert enabled.chunker_type == "semantic"


class TestBuildChunkingConfig:
    def test_semantic_enabled_uses_semantic_chunker_with_embedding(self, monkeypatch):
        """Semantic path requires an EmbeddingConfig or kreuzberg silently falls back."""
        from lilbee.core.config import cfg
        from lilbee.data.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "topic_threshold", 0.6)
        result = build_chunking_config()
        assert result.chunker_type == "semantic"
        assert result.topic_threshold == pytest.approx(0.6, abs=1e-5)
        assert result.embedding is not None

    def test_semantic_respects_max_chars_when_embedding_present(self, monkeypatch):
        """With an embedding attached kreuzberg honors max_characters on the semantic path."""
        from lilbee.core.config import cfg
        from lilbee.data.chunk import CHARS_PER_TOKEN, build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        result = build_chunking_config()
        assert result.max_characters == 512 * CHARS_PER_TOKEN
        assert result.embedding is not None
        assert result.embedding.model == "fast"

    def test_char_budget_when_disabled(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.chunk import CHARS_PER_TOKEN, build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        result = build_chunking_config()
        assert result.chunker_type == "text"
        assert result.max_characters == 512 * CHARS_PER_TOKEN
        assert result.overlap == 100 * CHARS_PER_TOKEN
        assert result.embedding is None

    def test_disabled_does_not_attach_embedding(self, monkeypatch):
        """When semantic is off, no EmbeddingConfig is built; avoids the ONNX download."""
        from lilbee.core.config import cfg
        from lilbee.data.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        result = build_chunking_config()
        assert result.embedding is None

    def test_download_progress_off_when_globally_suppressed(self, monkeypatch):
        """quiet/JSON modes suppress HF progress bars; the embedding config mirrors that."""
        from lilbee.data.chunk import _show_download_progress

        monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        assert _show_download_progress() is False

    def test_download_progress_on_when_not_suppressed(self, monkeypatch):
        from lilbee.data.chunk import _show_download_progress

        monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        assert _show_download_progress() is True

    def test_download_progress_default_off_when_unset(self, monkeypatch):
        """lilbee defaults the env var on at import, so the bar stays off by default."""
        from lilbee.data.chunk import _show_download_progress

        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
        # Unset env reads as "not disabled" -> progress allowed; lilbee's __init__
        # sets it to "1" in real runs, so this documents the bare-helper contract.
        assert _show_download_progress() is True

    def test_heading_path_shares_char_budget(self, monkeypatch):
        """The heading-aware path uses the same token->char budget as the default path."""
        from lilbee.core.config import cfg
        from lilbee.data.chunk import CHARS_PER_TOKEN, _char_budget

        monkeypatch.setattr(cfg, "chunk_size", 256)
        monkeypatch.setattr(cfg, "chunk_overlap", 40)
        max_chars, max_overlap = _char_budget()
        assert max_chars == 256 * CHARS_PER_TOKEN
        assert max_overlap == 40 * CHARS_PER_TOKEN


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
    """Tree-sitter code chunker tests: grouped to avoid fork-unsafe C parser collisions."""

    def test_python_function_extraction(self):
        from lilbee.data.code_chunker import chunk_code

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
        from lilbee.data.code_chunker import chunk_code

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
        from lilbee.data.code_chunker import is_code_file

        assert is_code_file(Path("main.py"))
        assert is_code_file(Path("app.js"))
        assert is_code_file(Path("lib.rs"))
        assert is_code_file(Path("server.go"))

    def test_is_code_file_non_code(self):
        from lilbee.data.code_chunker import is_code_file

        assert not is_code_file(Path("photo.png"))
        assert not is_code_file(Path("document.pdf"))

    def test_detect_language_python(self):
        from lilbee.data.code_chunker import _detect_language

        result = _detect_language(Path("main.py"))
        assert result is not None
        assert "python" in result.lower()

    def test_ensure_language_exception_returns_false(self):
        from unittest.mock import patch

        from lilbee.data.code_chunker import _ensure_language

        with patch("lilbee.data.code_chunker.has_language", side_effect=RuntimeError("boom")):
            assert _ensure_language("python") is False

    def test_find_line_no_match_returns_start(self):
        from lilbee.data.code_chunker import find_line

        lines = ["aaa", "bbb", "ccc"]
        assert find_line("zzz", lines, 0) == 1

    def test_fallback_line_tracking_does_not_skip_chunk_start(self):
        """The next search must begin at a chunk's actual start line, not one past
        it: an overlapping chunk that re-includes the previous start line would
        otherwise be skipped and mis-located (1-based start fed as a 0-based index)."""
        from unittest.mock import patch

        from lilbee.data import code_chunker

        text = "FIRST\nsecond"
        with patch.object(code_chunker, "chunk_text", return_value=["FIRST\nsecond", "FIRST"]):
            chunks = code_chunker._fallback_chunks(text)
        assert chunks[0].line_start == 1
        assert chunks[1].line_start == 1

    def test_ensure_language_false_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.data.code_chunker._ensure_language", return_value=False):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
        finally:
            path.unlink()

    def test_process_exception_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        # Force _ensure_language True so the try/except in chunk_code is
        # actually entered; otherwise CI hosts without tree-sitter Python
        # preloaded short-circuit to the no-language fallback first.
        try:
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", side_effect=RuntimeError("parse fail")),
            ):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
        finally:
            path.unlink()

    def test_empty_chunks_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            # process() yielding no size-bounded chunks must fall back to text
            # chunking rather than returning nothing.
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", return_value=_FakeResult([])),
            ):
                chunks = chunk_code(path)
                assert isinstance(chunks, list)
                assert chunks  # fell back to non-empty text chunks
        finally:
            path.unlink()

    def test_ensure_language_returns_true_when_already_loaded(self):
        """The early-return branch fires when has_language already says yes."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import _ensure_language

        with patch("lilbee.data.code_chunker.has_language", return_value=True) as has:
            assert _ensure_language("python") is True
            has.assert_called_once_with("python")

    def test_ensure_language_runs_install_when_not_loaded(self):
        """Cover the init()-then-recheck branch deterministically; without this,
        coverage of that line drifts whenever the tree-sitter language pack
        ships pre-installed for a given Python version."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import PackConfig, _ensure_language

        with (
            patch("lilbee.data.code_chunker.has_language", side_effect=[False, True]) as has,
            patch("lilbee.data.code_chunker.init") as init_mock,
        ):
            assert _ensure_language("python") is True
            assert has.call_count == 2
            init_mock.assert_called_once_with(PackConfig(languages=["python"]))

    def test_chunk_code_empty_source_returns_empty(self):
        """Empty (whitespace-only) source short-circuits before tree-sitter."""
        from lilbee.data.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("\n   \n")
            f.flush()
            path = Path(f.name)

        try:
            assert chunk_code(path) == []
        finally:
            path.unlink()

    def test_chunk_code_emits_chunks_from_result(self):
        """chunk_code consumes the parser's size-bounded result.chunks (not the
        unbounded structure tree): the header names the relative source and the
        chunk's symbols, and the content is passed through verbatim. Mocked so the
        test is independent of whether tree-sitter parses on this CI host."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        result = _FakeResult(
            [
                _FakeTSChunk(
                    "def hello():\n    return 1\n", start_line=0, end_line=2, symbols=["hello"]
                )
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def hello():\n    return 1\n")
            f.flush()
            path = Path(f.name)

        try:
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", return_value=result),
            ):
                chunks = chunk_code(path, source_name="pkg/mod.py")
        finally:
            path.unlink()

        assert len(chunks) == 1
        first = chunks[0]
        assert "# File: pkg/mod.py | hello (lines 1-2)" in first.chunk
        assert "def hello" in first.chunk
        assert first.line_start == 1
        assert first.line_end == 2
        assert first.chunk_index == 0

    def test_chunk_header_omits_symbols_and_never_says_none(self):
        """A symbol-free (anonymous) chunk omits the symbol segment entirely
        rather than rendering the literal string 'None' (bb-ziks.62)."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        result = _FakeResult([_FakeTSChunk("x = 1\n", start_line=0, end_line=1, symbols=[])])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", return_value=result),
            ):
                chunks = chunk_code(path, source_name="m.py")
        finally:
            path.unlink()
        assert chunks[0].chunk.startswith("# File: m.py (lines 1-1)")
        assert "None" not in chunks[0].chunk
        assert "|" not in chunks[0].chunk

    def test_header_uses_relative_source_name_not_absolute_path(self):
        """The header carries the relative source name, never the host's absolute
        path, so an exported corpus does not leak the operator's disk layout
        (bb-ziks.19)."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        result = _FakeResult([_FakeTSChunk("code\n", start_line=0, end_line=1, symbols=["s"])])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("code\n")
            f.flush()
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", return_value=result),
            ):
                chunks = chunk_code(path, source_name="src/x.py")
        finally:
            path.unlink()
        assert "src/x.py" in chunks[0].chunk
        assert str(path) not in chunks[0].chunk

    def test_non_ascii_code_not_corrupted(self):
        """Non-ASCII identifiers/strings survive chunking: the prior code sliced a
        str with tree-sitter UTF-8 byte offsets, mis-slicing every symbol after
        the first multibyte char (bb-7jg1.4)."""
        from lilbee.data.code_chunker import chunk_code

        code = (
            'def greet(náme):\n    return f"Hallo {náme}"\n\n'
            'class Wörker:\n    def café(self):\n        return "résumé"\n'
        )
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)
        try:
            chunks = chunk_code(path)
        finally:
            path.unlink()
        joined = "\n".join(c.chunk for c in chunks)
        # Assert the exact multi-byte spans, not bare identifiers: byte-offset
        # slicing of a str shifts after the first multibyte char, so the full
        # declarations would be mangled even though a stray identifier might survive.
        assert "def greet(náme):" in joined
        assert 'return f"Hallo {náme}"' in joined
        assert "class Wörker:" in joined
        assert "def café(self):" in joined
        assert 'return "résumé"' in joined

    def test_line_end_derived_from_content_not_parser_end_line(self):
        """A chunk whose final line has no trailing newline must report that line as
        line_end. tree-sitter's end_line counts newline-terminated lines, so it
        under-reports here; line_end is derived from the content instead."""
        from unittest.mock import patch

        from lilbee.data.code_chunker import chunk_code

        # Content spans two physical lines but ends without a trailing newline;
        # the fake's end_line=1 mirrors the real parser's under-count.
        result = _FakeResult(
            [_FakeTSChunk("def a():\n    return 1", start_line=0, end_line=1, symbols=["a"])]
        )
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def a():\n    return 1")
            f.flush()
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.code_chunker.process", return_value=result),
            ):
                chunks = chunk_code(path, source_name="m.py")
        finally:
            path.unlink()
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 2
        assert "lines 1-2" in chunks[0].chunk

    def test_real_parser_line_range_spans_no_trailing_newline_file(self):
        """Against the real parser, the chunk line range reaches the file's final
        line even when the file has no trailing newline."""
        from lilbee.data.code_chunker import chunk_code

        # 5 physical lines (L1..L5), no trailing newline on the last.
        code = "def alpha():\n    return 1\n\ndef beta():\n    return 2"
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)
        try:
            chunks = chunk_code(path)
        finally:
            path.unlink()
        assert chunks
        assert min(c.line_start for c in chunks) == 1
        assert max(c.line_end for c in chunks) == 5


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

        from lilbee.data.chunk import chunk_text

        mock_result = MagicMock()
        mock_result.chunks = []
        with patch("kreuzberg.extract_bytes_sync", return_value=mock_result):
            assert chunk_text("some text") == []
