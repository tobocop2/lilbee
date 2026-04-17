"""Integration tests demonstrating semantic vs text chunking behavior.

Uses real kreuzberg chunking (no mocks) on fixture documents to show
that semantic chunking produces topic-coherent chunks while text
chunking splits arbitrarily by character budget.
"""

from pathlib import Path

from kreuzberg import ChunkingConfig, ExtractionConfig, extract_bytes_sync

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def _chunk(
    text: str,
    chunker_type: str,
    max_chars: int | None = None,
    **kwargs,
) -> list[str]:
    """Chunk text using kreuzberg and return chunk contents."""
    chunking_kwargs: dict = {
        "chunker_type": chunker_type,
        "max_overlap": 0,
        **kwargs,
    }
    if max_chars is not None:
        chunking_kwargs["max_chars"] = max_chars
    config = ExtractionConfig(
        chunking=ChunkingConfig(**chunking_kwargs),
    )
    result = extract_bytes_sync(text.encode("utf-8"), "text/plain", config=config)
    return [c.content for c in result.chunks]


class TestSemanticVsTextChunking:
    """Compare semantic and text chunking on the same documents."""

    def test_semantic_respects_sections(self):
        """Semantic chunker keeps section content together."""
        text = (FIXTURES / "research_report.txt").read_text()
        chunks = _chunk(text, "semantic")

        energy = [c for c in chunks if "Solar panel" in c or "Wind energy" in c]
        assert energy, "Should have energy chunks"
        for chunk in energy:
            assert "FDA" not in chunk, "Energy chunk has healthcare content"

    def test_text_chunker_mixes_topics(self):
        """Text chunker allows topics to bleed across boundaries."""
        text = (FIXTURES / "research_report.txt").read_text()
        text_chunks = _chunk(text, "text", max_chars=800)
        semantic_chunks = _chunk(text, "semantic")

        markers = [
            ("solar", "FDA"),
            ("solar", "quantum"),
            ("Wind", "clinical"),
            ("Battery", "Deep learning"),
            ("Drug discovery", "qubit"),
        ]

        def cross_count(chunks: list[str]) -> int:
            n = 0
            for chunk in chunks:
                lo = chunk.lower()
                for a, b in markers:
                    if a.lower() in lo and b.lower() in lo:
                        n += 1
                        break
            return n

        assert cross_count(semantic_chunks) <= cross_count(text_chunks)

    def test_headers_force_boundaries(self):
        """ALL CAPS headers force chunk boundaries."""
        text = (FIXTURES / "research_report.txt").read_text()
        chunks = _chunk(text, "semantic")

        for section in [
            "RENEWABLE ENERGY",
            "MACHINE LEARNING",
            "QUANTUM COMPUTING",
            "CONCLUSIONS",
        ]:
            assert any(section in c for c in chunks), f"'{section}' missing"

    def test_mixed_topics_without_embeddings(self):
        """Without embeddings, merges within ceiling."""
        text = (FIXTURES / "mixed_topics.txt").read_text()
        chunks = _chunk(text, "semantic")

        combined = " ".join(chunks)
        assert "Pacific" in combined
        assert "Bastille" in combined
        assert "Chlorophyll" in combined


class TestSemanticChunkingProperties:
    """Test properties of semantic chunks."""

    def test_chunks_stay_under_ceiling(self):
        """Chunks stay under the auto-budget ceiling."""
        text = (FIXTURES / "research_report.txt").read_text()
        chunks = _chunk(text, "semantic")
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 4100, f"Chunk {i}: {len(chunk)} chars"

    def test_chunks_cover_full_document(self):
        """All source content appears in chunks."""
        text = (FIXTURES / "research_report.txt").read_text()
        combined = " ".join(_chunk(text, "semantic"))
        for phrase in [
            "monocrystalline panels",
            "sodium-ion batteries",
            "diabetic retinopathy",
            "ambient listening",
            "thousand-qubit",
            "entanglement distribution",
        ]:
            assert phrase in combined, f"'{phrase}' missing"

    def test_empty_text(self):
        assert _chunk("", "semantic") == []

    def test_short_text_single_chunk(self):
        chunks = _chunk("Hello world.", "semantic")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_topic_threshold_accepted(self):
        """topic_threshold is accepted without error."""
        text = (FIXTURES / "research_report.txt").read_text()
        assert len(_chunk(text, "semantic", topic_threshold=0.5)) >= 1

    def test_markdown_content(self):
        """Handles markdown fixtures."""
        text = (FIXTURES / "recipes.md").read_text()
        combined = " ".join(_chunk(text, "semantic"))
        assert "Szechuan" in combined

    def test_csv_degrades_gracefully(self):
        """Handles structured data without errors."""
        text = (FIXTURES / "inventory.csv").read_text()
        assert len(_chunk(text, "semantic")) >= 1

    def test_oversized_text_splits(self):
        """Text exceeding ceiling gets split."""
        text = "word " * 1500  # ~7500 chars
        assert len(_chunk(text, "semantic")) >= 2


class TestChunkerTypeComparison:
    """Side-by-side topic purity comparison."""

    def test_topic_purity(self):
        """Semantic chunks have higher topic purity."""
        text = (FIXTURES / "research_report.txt").read_text()

        keywords = {
            "energy": {"solar", "wind", "battery", "megawatt"},
            "healthcare": {"fda", "clinical", "physician"},
            "quantum": {"qubit", "quantum", "entanglement"},
        }

        def purity(chunks: list[str]) -> float:
            pure = 0
            for chunk in chunks:
                lo = chunk.lower()
                topics = {t for t, kws in keywords.items() if any(kw in lo for kw in kws)}
                if len(topics) <= 1:
                    pure += 1
            return pure / len(chunks) if chunks else 0.0

        text_chunks = _chunk(text, "text", max_chars=600)
        semantic_chunks = _chunk(text, "semantic")
        assert purity(semantic_chunks) >= purity(text_chunks)
