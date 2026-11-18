"""Tests for text chunking behavior.

These tests verify chunking invariants regardless of the underlying
implementation.
"""

import re
import tempfile
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.data.extract.chunk import chunk_text
from tests._mock_effects import repeat_last
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def mock_services():
    """Mock services for every chunker test: the sizer reads the provider's embed cap
    and counts through the tokenizer registered for that provider."""
    import lilbee.app.services as svc_mod

    services = make_mock_services()
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


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
        from lilbee.data.extract.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        bypassed = build_chunking_config(use_semantic=False)
        assert bypassed.chunker_type == "text"
        enabled = build_chunking_config()
        assert enabled.chunker_type == "semantic"


class TestBuildChunkingConfig:
    def test_semantic_enabled_uses_semantic_chunker_with_embedding(self, monkeypatch):
        """Semantic path requires an EmbeddingConfig or xberg silently falls back."""
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "topic_threshold", 0.6)
        result = build_chunking_config()
        assert result.chunker_type == "semantic"
        assert result.topic_threshold == pytest.approx(0.6, abs=1e-5)
        assert result.embedding is not None

    def test_semantic_respects_max_chars_when_embedding_present(self, monkeypatch):
        """With an embedding attached xberg honors max_characters on the semantic path."""
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import CHARS_PER_TOKEN, build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        result = build_chunking_config()
        assert result.max_characters == 512 * CHARS_PER_TOKEN
        assert result.embedding is not None
        # Boundary detection routes to lilbee's embedder via xberg's plugin backend
        # (registered in lilbee.data.extract.backends.registry), not the ONNX preset.
        assert result.embedding.model.type == "plugin"
        assert str(result.embedding.model) == '{"type":"plugin","name":"lilbee"}'

    def test_token_budget_when_semantic_disabled(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        result = build_chunking_config()
        assert result.chunker_type == "text"
        assert (result.max_characters, result.overlap) == (512, 100)
        assert str(result.sizing) == _current_provider_sizing()
        assert result.embedding is None

    def test_disabled_does_not_attach_embedding(self, monkeypatch):
        """When semantic is off, no EmbeddingConfig is built; avoids the ONNX download."""
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        result = build_chunking_config()
        assert result.embedding is None

    def test_heading_path_shares_char_budget(self, monkeypatch):
        """The heading-aware path uses the same token->char budget as the default path."""
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import CHARS_PER_TOKEN, _char_budget

        monkeypatch.setattr(cfg, "chunk_size", 256)
        monkeypatch.setattr(cfg, "chunk_overlap", 40)
        max_chars, max_overlap = _char_budget()
        assert max_chars == 256 * CHARS_PER_TOKEN
        assert max_overlap == 40 * CHARS_PER_TOKEN


_BGE_SMALL_CAP = 504
"""bge-small-en-v1.5 serves a 512 window; the cap is one margin below it."""
_TOKEN = re.compile(r"\w{1,4}|[^\w\s]")


def _wordpiece_like(text: str) -> int:
    """A tokenizer that splits words into pieces of four letters and counts punctuation."""
    return len(_TOKEN.findall(text))


def _heavy_at(extra: int):
    """A tokenizer of one token per character and *extra* more for each ``@``."""
    return lambda text: len(text) + extra * text.count("@")


_PROSE = " ".join(
    f"Paragraph {i}: the aqueduct carried water across the valley to the city on a slope."
    for i in range(300)
)
_DENSE = "\n".join(
    f"{i:04x}:" + "".join(f"{(i * 7 + j) % 16:x}" for j in range(64)) for i in range(400)
)


@pytest.fixture
def bge_small_embedder(mock_services):
    """The mock provider as a 512-window embedder that counts with its own tokenizer."""
    mock_services.provider.embed_token_cap.return_value = _BGE_SMALL_CAP
    mock_services.provider.count_tokens.side_effect = _wordpiece_like
    return mock_services.provider


class TestChunkBudgetUnderTheEmbedWindow:
    """When the embedder's window is below the character budget, chunks are sized in
    the embedder's own tokens at the cap; a wide window leaves the character budget."""

    def test_binding_window_selects_tokenizer_sizing_at_the_cap(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import _size_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 300)
        mock_services.provider.embed_token_cap.return_value = _BGE_SMALL_CAP
        max_size, overlap, sizing = _size_params()
        assert (max_size, overlap) == (_BGE_SMALL_CAP, _BGE_SMALL_CAP // 2)
        assert str(sizing) == _current_provider_sizing()

    def test_wide_window_keeps_chunk_size(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import _size_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        mock_services.provider.embed_token_cap.return_value = 2048
        max_size, overlap, _sizing = _size_params()
        assert (max_size, overlap) == (512, 100)

    def test_character_budget_is_never_cut_to_the_cap(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import CHARS_PER_TOKEN, _char_budget

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        mock_services.provider.embed_token_cap.return_value = _BGE_SMALL_CAP
        assert _char_budget() == (512 * CHARS_PER_TOKEN, 100 * CHARS_PER_TOKEN)

    def test_no_managed_embedder_keeps_chunk_size(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import _size_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        mock_services.provider.embed_token_cap.return_value = None
        assert _size_params()[0] == 512

    def test_chunks_fit_the_cap_in_the_embedders_tokens_without_shrinking_prose(
        self, monkeypatch, bge_small_embedder
    ):
        """Under the real bge-small cap, no chunk exceeds it under the tokenizer the
        chunker sized with, and prose chunks keep near the configured size."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)

        prose = chunk_text(_PROSE, use_semantic=False)
        dense = chunk_text(_DENSE, use_semantic=False)

        assert max(_wordpiece_like(c) for c in prose + dense) <= _BGE_SMALL_CAP
        assert max(len(c) for c in prose) > 2 * _BGE_SMALL_CAP
        assert max(len(c) for c in dense) <= _BGE_SMALL_CAP * 4

    def test_tokenizer_is_registered_for_the_provider_on_first_use(
        self, monkeypatch, bge_small_embedder
    ):
        import xberg

        from lilbee.core.config import cfg
        from lilbee.data.extract.backends.tokenizer import provider_tokenizers

        monkeypatch.setattr(cfg, "chunk_size", 512)
        before = set(xberg.list_tokenizer_backends())
        chunk_text("a short line", use_semantic=False)
        added = set(xberg.list_tokenizer_backends()) - before
        assert added == {provider_tokenizers.name_for(bge_small_embedder)}
        bge_small_embedder.count_tokens.assert_called()


def _current_provider_sizing() -> str:
    """The sizing that names the tokenizer registered for the current provider."""
    from lilbee.app.services import get_services
    from lilbee.data.extract.backends.tokenizer import provider_tokenizers

    name = provider_tokenizers.name_for(get_services().provider)
    return f'{{"type":"tokenizer","model":"{name}"}}'


class TestTokenSizing:
    """The plain and heading chunkers go through lilbee's registered tokenizer
    backend, so chunk_size is a real token budget."""

    def test_size_params_is_raw_token_budget_with_sizing(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import _size_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        max_size, overlap, sizing = _size_params()
        assert (max_size, overlap) == (512, 100)
        assert str(sizing) == _current_provider_sizing()

    def test_overlap_capped_at_half_the_token_budget(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import _size_params

        monkeypatch.setattr(cfg, "chunk_size", 100)
        monkeypatch.setattr(cfg, "chunk_overlap", 90)
        _max_size, overlap, _sizing = _size_params()
        assert overlap == 50

    def test_semantic_chunker_keeps_the_character_budget(self, monkeypatch):
        """The semantic chunker sizes by characters (xberg's ChunkSizing has no
        effect on that path)."""
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import CHARS_PER_TOKEN, build_chunking_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        result = build_chunking_config()
        assert result.chunker_type == "semantic"
        assert result.max_characters == 512 * CHARS_PER_TOKEN

    def test_heading_path_attaches_tokenizer_sizing(self, monkeypatch):
        from lilbee.data.extract.chunk import chunk_text

        captured = {}

        def fake_extract_document(data, mime_type, *, config):
            chunking = config.chunking
            captured["sizing"] = str(chunking.sizing)
            captured["chunker_type"] = chunking.chunker_type
            return SimpleNamespace(chunks=[])

        monkeypatch.setattr("lilbee.data.extract.xberg.extract_document", fake_extract_document)
        chunk_text("# Heading\n\nbody text", mime_type="text/markdown", heading_context=True)
        assert captured["sizing"] == _current_provider_sizing()
        assert captured["chunker_type"] == "markdown"


class TestChunksFitTheEmbedderTokenLimit:
    """Chunks are sized by the embedder's own token count under the default config."""

    def test_dense_text_never_exceeds_chunk_size_tokens(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "semantic_chunking", False)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(cfg, "chunk_overlap", 100)
        mock_services.provider.embed_token_cap.return_value = None
        mock_services.provider.count_tokens.side_effect = _wordpiece_like

        chunks = chunk_text(_DENSE, use_semantic=False)

        assert len(chunks) > 1
        assert max(_wordpiece_like(c) for c in chunks) <= 512

    def test_breadcrumb_counts_inside_the_budget(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 128)
        monkeypatch.setattr(cfg, "chunk_overlap", 20)
        mock_services.provider.embed_token_cap.return_value = None
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        body = "\n\n".join(
            f"Paragraph {i}: the aqueduct carried water to the city." for i in range(80)
        )
        doc = "# Roman Water Supply\n\n## Aqueducts and Channels\n\n" + body

        chunks = chunk_text(doc, mime_type="text/markdown", heading_context=True)

        assert len(chunks) > 1
        assert any(c.startswith("# Roman Water Supply > ## Aqueducts and Channels") for c in chunks)
        assert max(_wordpiece_like(c) for c in chunks) <= 128

    def test_a_heading_path_over_half_the_budget_is_dropped(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        monkeypatch.setattr(cfg, "chunk_overlap", 8)
        mock_services.provider.embed_token_cap.return_value = None
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        heading = " ".join(f"section{i}" for i in range(40))
        body = "\n\n".join(
            f"Paragraph {i}: the aqueduct carried water to the city." for i in range(30)
        )

        chunks = chunk_text(
            f"# {heading}\n\n{body}", mime_type="text/markdown", heading_context=True
        )

        assert len(chunks) > 1
        assert all(heading not in c for c in chunks)
        assert max(_wordpiece_like(c) for c in chunks) <= 64


def _headings(*texts: str) -> SimpleNamespace:
    """A heading context of one level-1 heading per text."""
    return SimpleNamespace(headings=[SimpleNamespace(level=1, text=t) for t in texts])


def _words(n: int, stem: str = "w") -> str:
    return " ".join(f"{stem}{i}" for i in range(n))


class TestFitBreadcrumb:
    """The heading path fits inside the budget, or goes when it takes over half of it."""

    @pytest.fixture(autouse=True)
    def _word_counter(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_overlap", 0)
        mock_services.provider.embed_token_cap.return_value = None
        mock_services.provider.count_tokens.side_effect = lambda text: len(text.split())

    def test_a_path_over_half_the_budget_leaves_the_chunk_bare(self):
        from lilbee.data.extract.chunk import _fit_breadcrumb

        content = _words(8)

        assert _fit_breadcrumb(content, _headings(_words(6, "p")), 10) == [content]

    def test_a_path_of_exactly_half_the_budget_stays_on_every_piece(self):
        from lilbee.data.extract.chunk import _fit_breadcrumb

        pieces = _fit_breadcrumb(_words(8), _headings(_words(4, "p")), 10)

        assert len(pieces) > 1
        assert all(p.startswith("# p0 p1 p2 p3\n\n") for p in pieces)
        assert max(len(p.split()) for p in pieces) <= 10

    def test_a_chunk_that_exactly_fills_the_budget_with_its_path_is_not_re_split(self, monkeypatch):
        from lilbee.data.extract import chunk

        def no_re_split(*_args):
            raise AssertionError("a chunk that fits was re-split")

        monkeypatch.setattr(chunk, "_extract_chunks", no_re_split)

        assert chunk._fit_breadcrumb(_words(5), _headings(_words(4, "p")), 10) == [
            "# p0 p1 p2 p3\n\n" + _words(5)
        ]

    def test_every_re_split_piece_carries_the_path(self):
        from lilbee.data.extract.chunk import _fit_breadcrumb

        content = _words(20)

        pieces = _fit_breadcrumb(content, _headings("Install"), 10)

        assert len(pieces) > 2
        assert all(p.startswith("# Install\n\n") for p in pieces)
        assert " ".join(p.removeprefix("# Install\n\n") for p in pieces) == content


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

    def test_saved_reference_links_keep_their_brackets(self):
        """A crawled page's ``[[1]](url)`` references index as ``word[1][2]``."""
        from lilbee.crawler.save import normalize_crawled_markdown

        raw = (
            "# Page\n\nThe build workflow[[1]](https://w.org/a#cite_note-1)"
            "[[2]](https://w.org/a#cite_note-2) runs nightly."
        )
        md = normalize_crawled_markdown(raw)
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        joined = " ".join(chunks)
        assert "workflow[1][2] runs nightly" in joined
        assert "cite_note" not in joined


@pytest.mark.xdist_group("tree_sitter")
class TestCodeChunker:
    """Tree-sitter code chunker tests: grouped to avoid fork-unsafe C parser collisions."""

    def test_python_function_extraction(self):
        from lilbee.data.extract.code_chunker import chunk_code

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
        from lilbee.data.extract.code_chunker import chunk_code

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
        from lilbee.data.extract.code_chunker import is_code_file

        assert is_code_file(Path("main.py"))
        assert is_code_file(Path("app.js"))
        assert is_code_file(Path("lib.rs"))
        assert is_code_file(Path("server.go"))

    def test_is_code_file_non_code(self):
        from lilbee.data.extract.code_chunker import is_code_file

        assert not is_code_file(Path("photo.png"))
        assert not is_code_file(Path("document.pdf"))

    def test_detect_language_python(self):
        from lilbee.data.extract.code_chunker import _detect_language

        result = _detect_language(Path("main.py"))
        assert result is not None
        assert "python" in result.lower()

    def test_ensure_language_exception_returns_false(self):
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import _ensure_language

        with patch(
            "lilbee.data.extract.code_chunker.has_language", side_effect=RuntimeError("boom")
        ):
            assert _ensure_language("python") is False

    def test_find_line_no_match_returns_start(self):
        from lilbee.data.extract.code_chunker import find_line

        lines = ["aaa", "bbb", "ccc"]
        assert find_line("zzz", lines, 0) == 1

    def test_fallback_line_tracking_does_not_skip_chunk_start(self):
        """The next search must begin at a chunk's actual start line, not one past
        it: an overlapping chunk that re-includes the previous start line would
        otherwise be skipped and mis-located (1-based start fed as a 0-based index)."""
        from unittest.mock import patch

        from lilbee.data.extract import code_chunker

        text = "FIRST\nsecond"
        with patch.object(code_chunker, "chunk_text", return_value=["FIRST\nsecond", "FIRST"]):
            chunks = code_chunker._fallback_chunks(text)
        assert chunks[0].line_start == 1
        assert chunks[1].line_start == 1

    def test_ensure_language_false_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            with patch("lilbee.data.extract.code_chunker._ensure_language", return_value=False):
                chunks = chunk_code(path)
                # Truthiness, not isinstance: a regression returning [] would drop
                # the file's content from the index and still be a list.
                assert chunks
                assert "x = 1" in "\n".join(c.chunk for c in chunks)
        finally:
            path.unlink()

    def test_process_exception_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        # Force _ensure_language True so the try/except in chunk_code is
        # actually entered; otherwise CI hosts without tree-sitter Python
        # preloaded short-circuit to the no-language fallback first.
        try:
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch(
                    "lilbee.data.extract.code_chunker.process",
                    side_effect=RuntimeError("parse fail"),
                ),
            ):
                chunks = chunk_code(path)
                assert chunks
                assert "x = 1" in "\n".join(c.chunk for c in chunks)
        finally:
            path.unlink()

    def test_empty_chunks_triggers_fallback(self):
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n" * 20)
            f.flush()
            path = Path(f.name)

        try:
            # process() yielding no size-bounded chunks must fall back to text
            # chunking rather than returning nothing.
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=_FakeResult([])),
            ):
                chunks = chunk_code(path)
                assert chunks  # fell back to non-empty text chunks
                assert "x = 1" in "\n".join(c.chunk for c in chunks)
        finally:
            path.unlink()

    def test_ensure_language_returns_true_when_already_loaded(self):
        """The early-return branch fires when has_language already says yes."""
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import _ensure_language

        with patch("lilbee.data.extract.code_chunker.has_language", return_value=True) as has:
            assert _ensure_language("python") is True
            has.assert_called_once_with("python")

    def test_ensure_language_runs_install_when_not_loaded(self):
        """Cover the init()-then-recheck branch deterministically; without this,
        coverage of that line drifts whenever the tree-sitter language pack
        ships pre-installed for a given Python version."""
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import PackConfig, _ensure_language

        with (
            patch(
                "lilbee.data.extract.code_chunker.has_language",
                side_effect=repeat_last(False, True),
            ) as has,
            patch("lilbee.data.extract.code_chunker.init") as init_mock,
        ):
            assert _ensure_language("python") is True
            assert has.call_count == 2
            init_mock.assert_called_once_with(PackConfig(languages=["python"]))

    def test_chunk_code_empty_source_returns_empty(self):
        """Empty (whitespace-only) source short-circuits before tree-sitter."""
        from lilbee.data.extract.code_chunker import chunk_code

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

        from lilbee.data.extract.code_chunker import chunk_code

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
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result),
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

    def _chunk_with_parser_result(self, result, source_name="pkg/mod.py"):
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import chunk_code

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write("x = 1\n")
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result),
            ):
                return chunk_code(path, source_name=source_name)
        finally:
            path.unlink()

    def test_parser_chunk_over_the_token_budget_is_split_at_line_boundaries(
        self, monkeypatch, mock_services
    ):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 100)
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        body = "".join(f"value_{i:03d} = compute({i}, {i + 1})\n" for i in range(60))
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=60, symbols=["compute"])])

        chunks = self._chunk_with_parser_result(result)

        assert len(chunks) > 1
        assert max(_wordpiece_like(c.chunk) for c in chunks) <= 100
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert chunks[0].line_start == 1
        assert chunks[-1].line_end == 60
        assert all(a.line_end + 1 == b.line_start for a, b in pairwise(chunks))
        assert all("# File: pkg/mod.py | compute (lines" in c.chunk for c in chunks)
        body_lines = [ln for c in chunks for ln in c.chunk.split("\n\n", 1)[1].splitlines()]
        assert body_lines == body.splitlines()

    def test_one_long_line_over_the_budget_is_split_mid_line(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 100)
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        line = "data = [" + ", ".join(str(i) for i in range(300)) + "]\n"
        result = _FakeResult([_FakeTSChunk(line, start_line=4, end_line=5, symbols=[])])

        chunks = self._chunk_with_parser_result(result)

        assert len(chunks) > 1
        assert max(_wordpiece_like(c.chunk) for c in chunks) <= 100
        assert {c.line_start for c in chunks} == {5}
        assert "".join(c.chunk.split("\n\n", 1)[1] for c in chunks) == line

    def test_a_piece_that_cannot_shrink_is_kept(self, monkeypatch, mock_services):
        """One character over the budget on its own leaves nothing to split."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = _heavy_at(100)
        result = _FakeResult([_FakeTSChunk("@", start_line=0, end_line=1, symbols=[])])

        chunks = self._chunk_with_parser_result(result)

        assert len(chunks) == 1
        assert chunks[0].chunk == "# File: pkg/mod.py (lines 1-1)\n\n@"

    def test_a_file_name_over_half_the_budget_is_dropped_with_the_header(
        self, monkeypatch, mock_services
    ):
        """A long file name does not split a body into a chunk for each character."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = len
        body = "".join(f"row_{i:02d} = {i}\n" for i in range(40))
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=40, symbols=["row"])])

        chunks = self._chunk_with_parser_result(result, source_name="n" * 57 + ".py")

        assert 1 < len(chunks) < 20
        assert max(len(c.chunk) for c in chunks) <= 64
        assert all("# File:" not in c.chunk for c in chunks)
        assert "".join(c.chunk for c in chunks) == body
        assert (chunks[0].line_start, chunks[-1].line_end) == (1, 40)

    def test_a_symbol_list_over_half_the_budget_is_dropped_from_the_header(
        self, monkeypatch, mock_services
    ):
        """A class of 150 methods does not split into a chunk for each character."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        methods = [f"method_with_a_long_name_{i}" for i in range(150)]
        body = "class Wide:\n" + "".join(f"    def {m}(self):\n        return 1\n" for m in methods)
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=301, symbols=methods)])

        chunks = self._chunk_with_parser_result(result)

        assert 1 < len(chunks) < 100
        assert max(_wordpiece_like(c.chunk) for c in chunks) <= 64
        assert all(c.chunk.startswith("# File: pkg/mod.py (lines ") for c in chunks)
        body_lines = [ln for c in chunks for ln in c.chunk.split("\n\n", 1)[1].splitlines()]
        assert body_lines == body.splitlines()

    def _bodies(self, chunks):
        return [c.chunk.split("\n\n", 1)[1] for c in chunks]

    def test_a_chunk_that_exactly_fills_the_budget_is_not_split(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = lambda text: len(text.split())
        body = " ".join(["w"] * 59)
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=1)])

        assert len(self._chunk_with_parser_result(result)) == 1

    def test_a_header_of_exactly_half_the_budget_keeps_its_symbols(
        self, monkeypatch, mock_services
    ):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = lambda text: len(text.split())
        body = "w\n" * 40
        symbols = [f"s{i}" for i in range(26)]
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=40, symbols=symbols)])

        chunks = self._chunk_with_parser_result(result)

        assert len(chunks) == 2
        assert all(c.chunk.startswith("# File: pkg/mod.py | s0, s1, ") for c in chunks)

    def test_two_lines_over_the_budget_split_at_the_line_boundary(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = lambda text: len(text.split())
        long_line = " ".join(["w"] * 40) + "\n"
        short_line = " ".join(["w"] * 20) + "\n"
        result = _FakeResult([_FakeTSChunk(long_line + short_line, start_line=0, end_line=2)])

        chunks = self._chunk_with_parser_result(result)

        assert self._bodies(chunks) == [long_line, short_line]

    def test_four_lines_over_the_budget_split_in_the_middle(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = lambda text: len(text.split())
        line = " ".join(["w"] * 15) + "\n"
        result = _FakeResult([_FakeTSChunk(line * 4, start_line=0, end_line=4)])

        chunks = self._chunk_with_parser_result(result)

        assert self._bodies(chunks) == [line * 2, line * 2]

    def test_two_characters_over_the_budget_split_into_single_characters(
        self, monkeypatch, mock_services
    ):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = _heavy_at(20)
        result = _FakeResult([_FakeTSChunk("@@", start_line=0, end_line=1)])

        chunks = self._chunk_with_parser_result(result)

        assert self._bodies(chunks) == ["@", "@"]

    def test_one_long_line_splits_at_its_middle(self, monkeypatch, mock_services):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = _heavy_at(12)
        result = _FakeResult([_FakeTSChunk("@a@b@c", start_line=0, end_line=1)])

        chunks = self._chunk_with_parser_result(result)

        assert self._bodies(chunks) == ["@a@", "b@c"]

    def test_a_symbol_list_within_half_the_budget_stays_in_the_header(
        self, monkeypatch, mock_services
    ):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "chunk_size", 64)
        mock_services.provider.count_tokens.side_effect = _wordpiece_like
        body = "".join(f"value_{i:03d} = compute({i}, {i + 1})\n" for i in range(20))
        result = _FakeResult([_FakeTSChunk(body, start_line=0, end_line=20, symbols=["compute"])])

        chunks = self._chunk_with_parser_result(result)

        assert len(chunks) > 1
        assert all("# File: pkg/mod.py | compute (lines" in c.chunk for c in chunks)

    def test_chunk_code_budget_is_bytes_not_tokens(self):
        """tree-sitter's chunk_max_size is documented in BYTES while
        cfg.chunk_size is a token budget, so the parser path must convert the
        same way the text path does -- otherwise code files split ~4x smaller
        than prose, and the parser and fallback paths disagree for one file."""
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import CHARS_PER_TOKEN
        from lilbee.data.extract.code_chunker import chunk_code

        result = _FakeResult([_FakeTSChunk("x = 1\n", start_line=0, end_line=1, symbols=[])])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            path = Path(f.name)

        try:
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result) as proc,
                patch("lilbee.data.extract.code_chunker.ProcessConfig") as config_cls,
            ):
                chunk_code(path, source_name="m.py")
        finally:
            path.unlink()

        proc.assert_called_once()
        assert config_cls.call_args.kwargs["chunk_max_size"] == cfg.chunk_size * CHARS_PER_TOKEN

    def test_chunk_header_omits_symbols_and_never_says_none(self):
        """A symbol-free (anonymous) chunk omits the symbol segment entirely
        rather than rendering the literal string 'None' (bb-ziks.62)."""
        from unittest.mock import patch

        from lilbee.data.extract.code_chunker import chunk_code

        result = _FakeResult([_FakeTSChunk("x = 1\n", start_line=0, end_line=1, symbols=[])])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result),
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

        from lilbee.data.extract.code_chunker import chunk_code

        result = _FakeResult([_FakeTSChunk("code\n", start_line=0, end_line=1, symbols=["s"])])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("code\n")
            f.flush()
            path = Path(f.name)
        try:
            with (
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result),
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
        from lilbee.data.extract.code_chunker import chunk_code

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

        from lilbee.data.extract.code_chunker import chunk_code

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
                patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
                patch("lilbee.data.extract.code_chunker.process", return_value=result),
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
        from lilbee.data.extract.code_chunker import chunk_code

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


class TestHeadingBreadcrumb:
    def test_breadcrumb_prefixes_each_chunk(self):
        """Every chunk starts with its heading path in ``# A > ## B`` form."""
        md = "# Title\n\n" + "Word " * 500 + "\n\n## Section\n\n" + "More " * 500
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert chunks[0].startswith("# Title\n\n")
        assert chunks[-1].startswith("# Title > ## Section\n\nMore ")

    def test_no_breadcrumb_without_headings(self):
        md = "Plain paragraph text. " * 60
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert chunks
        assert all(c.startswith("Plain paragraph") for c in chunks)

    def test_breadcrumb_is_not_repeated_as_body(self):
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

        from lilbee.data.extract.chunk import chunk_text

        mock_result = MagicMock()
        mock_result.chunks = []
        with patch("lilbee.data.extract.xberg.extract_document", return_value=mock_result):
            assert chunk_text("some text") == []
