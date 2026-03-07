"""RAG accuracy tests using a generated test PDF with known facts.

Requires Ollama running with mistral and nomic-embed-text models pulled.
"""

from pathlib import Path

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod


def _models_available() -> bool:
    """Check that both embedding and chat models are available."""
    try:
        import ollama

        from lilbee.embedder import embed

        embed("test")  # fastembed, no Ollama needed
        ollama.chat(model=cfg.CHAT_MODEL, messages=[{"role": "user", "content": "hi"}])
        return True
    except Exception:
        return False


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="Ollama not running or required models not pulled",
)


def _generate_test_pdf(output_path: Path) -> None:
    """Generate a test PDF with known vehicle specs using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(output_path), pagesize=letter)
    _width, height = letter

    # Page 1 — Vehicle Specs
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "2024 Thunderbolt X500 — Vehicle Specifications")
    c.setFont("Helvetica", 12)
    y = height - 110
    specs = [
        "Engine: 3.8L V6 Turbocharged",
        "Horsepower: 320 HP @ 5,500 RPM",
        "Torque: 380 lb-ft @ 2,800 RPM",
        "Oil capacity: 6.5 quarts with filter",
        "Oil type: 5W-30 full synthetic",
        "Tire pressure: 35 PSI front, 33 PSI rear",
        "Fuel tank capacity: 19 gallons",
        "Transmission: 10-speed automatic",
    ]
    for line in specs:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    # Page 2 — Maintenance Schedule
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Maintenance Schedule")
    c.setFont("Helvetica", 12)
    y = height - 110
    schedule = [
        "Oil change interval: every 7,500 miles",
        "Oil filter replacement: every oil change",
        "Brake fluid flush: every 30,000 miles",
        "Transmission fluid: every 60,000 miles",
        "Spark plugs: every 100,000 miles",
        "Coolant flush: every 50,000 miles",
        "Air filter: every 15,000 miles",
    ]
    for line in schedule:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    # Page 3 — Troubleshooting
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Troubleshooting — Diagnostic Trouble Codes")
    c.setFont("Helvetica", 12)
    y = height - 110
    codes = [
        "P0301: Cylinder 1 misfire detected. Check spark plugs, ignition coil, fuel injector.",
        "P0420: Catalytic converter efficiency below threshold. Check O2 sensors first.",
        "P0171: System too lean (Bank 1). Check for vacuum leaks, MAF sensor, fuel pressure.",
        "P0300: Random/multiple cylinder misfire. Check spark plugs, wires, fuel system.",
    ]
    for line in codes:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    c.save()


@pytest.fixture(scope="module")
def ingested_db(tmp_path_factory):
    """Generate test PDF, ingest it, return the db path."""
    tmp = tmp_path_factory.mktemp("lilbee_test")
    db_dir = tmp / "lancedb"
    docs_dir = tmp / "documents" / "test-docs"
    docs_dir.mkdir(parents=True)

    # Patch config to use temp paths
    original_db = cfg.LANCEDB_DIR
    original_docs = cfg.DOCUMENTS_DIR
    cfg.LANCEDB_DIR = db_dir
    cfg.DOCUMENTS_DIR = docs_dir.parent
    store_mod.LANCEDB_DIR = db_dir

    # Generate and ingest test PDF
    pdf_path = docs_dir / "test_vehicle_specs.pdf"
    _generate_test_pdf(pdf_path)

    import asyncio

    from lilbee.ingest import sync

    asyncio.run(sync())

    yield tmp

    # Restore config
    cfg.LANCEDB_DIR = original_db
    cfg.DOCUMENTS_DIR = original_docs
    store_mod.LANCEDB_DIR = original_db


# Each test checks retrieval + answer accuracy
_TEST_CASES = [
    ("What is the oil capacity of the Thunderbolt X500?", ["6.5 quarts"]),
    ("How often should I change the oil?", ["7,500 miles", "7500"]),
    ("What does code P0301 mean?", ["misfire", "Cylinder 1"]),
    ("What is the tire pressure for the front tires?", ["35 PSI"]),
    ("What engine does the Thunderbolt X500 have?", ["3.8L", "V6"]),
    ("What is the torque rating?", ["380", "lb-ft"]),
    ("When should I replace the spark plugs?", ["100,000", "100000"]),
]


@requires_models
class TestRAGAccuracy:
    @pytest.mark.parametrize("question,expected_terms", _TEST_CASES)
    def test_answer_contains_expected_facts(self, ingested_db, question, expected_terms):
        from lilbee.query import ask

        answer = ask(question)
        answer_lower = answer.lower()
        # Strip commas so "7,500" matches "7500" and vice versa
        answer_no_commas = answer_lower.replace(",", "")
        for term in expected_terms:
            term_lower = term.lower()
            assert term_lower in answer_lower or term_lower.replace(",", "") in answer_no_commas, (
                f"Expected '{term}' in answer to '{question}', got:\n{answer}"
            )

    def test_source_attribution_present(self, ingested_db):
        from lilbee.query import ask

        answer = ask("What is the oil capacity?")
        assert "Sources:" in answer
        assert "test_vehicle_specs.pdf" in answer

    def test_retrieval_finds_relevant_chunks(self, ingested_db):
        from lilbee.query import search_context

        results = search_context("oil capacity")
        assert len(results) > 0
        chunks_text = " ".join(r["chunk"] for r in results)
        assert "6.5 quarts" in chunks_text
