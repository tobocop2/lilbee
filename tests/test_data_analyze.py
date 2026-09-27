"""Corpus analysis signals: file types, sampling, extraction with OCR off, cancel and progress."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.data import analyze
from lilbee.data.analyze import FileFailure, LanguageShare, collect_signals, sample_keys
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.progress import AnalyzeEvent, EventType
from tests.conftest import make_pdf


def _doc(*, pdf=None, text_format=False, tables=0, content="", languages=None, pages=0):
    """An ExtractedDocument stand-in carrying the fields analyze reads."""
    if pdf is not None:
        fmt = SimpleNamespace(pdf=SimpleNamespace(page_count=pdf[0], scanned_pages=pdf[1]))
    else:
        fmt = SimpleNamespace(pdf=None) if text_format else None
    return SimpleNamespace(
        metadata=SimpleNamespace(format=fmt),
        counts=SimpleNamespace(pages=pages, tables=tables),
        content=content,
        detected_languages=languages,
    )


class _FakeExtract:
    """Answers aextract_batch by file name and records every batch it was given."""

    def __init__(self, by_name):
        self.by_name = by_name
        self.batches: list[list[str]] = []

    async def __call__(self, items, config):
        self.batches.append([item.filename for item in items])
        return [self.by_name[item.filename] for item in items]


def _write(root: Path, name: str, data: bytes = b"x") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def corpus(tmp_path):
    names = ["scan.pdf", "tabled.pdf", "notes.txt", "readme.md", "bad.pdf", "main.py", "pic.png"]
    files = {name: _write(tmp_path, name) for name in names}
    files["gone.md"] = tmp_path / "gone.md"
    files["unknown.zzz"] = _write(tmp_path, "unknown.zzz")
    return files


@pytest.fixture
def fake(monkeypatch):
    fake = _FakeExtract(
        {
            "scan.pdf": _doc(pdf=(10, [1, 2, 3, 4])),
            "tabled.pdf": _doc(pdf=(None, None), pages=2, tables=3, languages=["eng"]),
            "notes.txt": _doc(text_format=True, content="a" * 100, languages=["deu", "eng"]),
            "readme.md": _doc(content="b" * 300, languages=["eng"]),
            "bad.pdf": RuntimeError("Invalid cross-reference table"),
        }
    )
    monkeypatch.setattr(analyze, "aextract_batch", fake)
    return fake


def test_signals_come_from_every_file_and_the_extracted_sample(corpus, fake):
    signals = asyncio.run(collect_signals(corpus))

    assert (signals.files_total, signals.documents_total, signals.files_counted) == (8, 6, 2)
    assert signals.files_read == 4
    assert signals.cap == cfg.analyze_max_files
    assert signals.file_types == {"pdf": 3, "md": 2, "txt": 1, "code": 1, "image": 1}
    assert signals.code_share == 1 / 8
    pdf = signals.pdf
    assert (pdf.files, pdf.pages, pdf.scanned_pages) == (2, 12, 4)
    assert pdf.scanned_share == (4 + 1) / (12 + 1)
    assert (pdf.files_with_tables, pdf.tables, pdf.median_pages) == (1, 3, 6.0)
    assert signals.median_chars == 200.0
    assert signals.languages == (LanguageShare("eng", 2 / 3), LanguageShare("deu", 1 / 3))
    assert signals.image_files == 1
    assert FileFailure("bad.pdf", "Invalid cross-reference table") in signals.failed


def test_failures_name_the_file_and_the_reason(corpus, fake):
    failed = {f.file: f.error for f in asyncio.run(collect_signals(corpus)).failed}
    assert failed["bad.pdf"] == "Invalid cross-reference table"
    assert "gone.md" in failed["gone.md"]
    assert set(failed) == {"bad.pdf", "gone.md"}


def test_code_images_and_unknown_files_are_never_extracted(corpus, fake):
    asyncio.run(collect_signals(corpus))
    extracted = {name for batch in fake.batches for name in batch}
    assert extracted == {"scan.pdf", "tabled.pdf", "notes.txt", "readme.md", "bad.pdf"}


def test_an_archive_is_counted_but_not_extracted(tmp_path, monkeypatch):
    fake = _FakeExtract({"a.md": _doc(content="x")})
    monkeypatch.setattr(analyze, "aextract_batch", fake)
    files = {"a.md": _write(tmp_path, "a.md"), "b.zip": _write(tmp_path, "b.zip")}
    signals = asyncio.run(collect_signals(files))
    assert signals.file_types == {"md": 1, "zip": 1}
    assert fake.batches == [["a.md"]]


def test_extraction_runs_in_batches_of_the_batch_size_with_one_event_each(corpus, fake):
    cfg.batch_extraction_size = 2
    events: list[tuple[EventType, AnalyzeEvent]] = []
    asyncio.run(collect_signals(corpus, on_progress=lambda t, e: events.append((t, e))))
    # gone.md shares the first batch and fails its read, so only bad.pdf reaches xberg
    assert fake.batches == [["bad.pdf"], ["notes.txt", "readme.md"], ["scan.pdf", "tabled.pdf"]]
    assert [(t, e.done, e.total) for t, e in events] == [
        (EventType.ANALYZE, 2, 6),
        (EventType.ANALYZE, 4, 6),
        (EventType.ANALYZE, 6, 6),
    ]
    assert events[-1][1].file == "tabled.pdf"


def test_cancel_between_batches_stops_before_the_next_batch(corpus, fake):
    cfg.batch_extraction_size = 2
    cancel = threading.Event()
    with pytest.raises(TaskCancelledError):
        asyncio.run(collect_signals(corpus, on_progress=lambda *_: cancel.set(), cancel=cancel))
    assert len(fake.batches) == 1


def test_cancel_after_the_last_batch_still_returns_no_signals(corpus, fake):
    cancel = threading.Event()
    with pytest.raises(TaskCancelledError):
        asyncio.run(collect_signals(corpus, on_progress=lambda *_: cancel.set(), cancel=cancel))
    assert len(fake.batches) == 1


def test_the_cap_is_spent_only_on_documents_that_are_extracted(tmp_path, monkeypatch):
    cfg.analyze_max_files = 2
    code = {f"src/{i:02}.py": _write(tmp_path, f"src/{i:02}.py") for i in range(12)}
    docs = {f"docs/d{i}.md": _write(tmp_path, f"docs/d{i}.md") for i in range(4)}
    fake = _FakeExtract({f"d{i}.md": _doc(content="x") for i in range(4)})
    monkeypatch.setattr(analyze, "aextract_batch", fake)
    signals = asyncio.run(collect_signals({**code, **docs}))
    assert [name for batch in fake.batches for name in batch] == ["d0.md", "d2.md"]
    assert (signals.files_total, signals.documents_total, signals.files_counted) == (16, 4, 12)
    assert (signals.files_read, signals.cap) == (2, 2)


def test_a_corpus_over_the_cap_extracts_an_even_stride_not_the_first_files(tmp_path, monkeypatch):
    cfg.analyze_max_files = 3
    names = [f"{root}{i}.md" for root in "abc" for i in range(4)]
    files = {f"{name[0]}/{name}": _write(tmp_path, f"{name[0]}/{name}") for name in names}
    fake = _FakeExtract({name: _doc(content="x") for name in names})
    monkeypatch.setattr(analyze, "aextract_batch", fake)
    asyncio.run(collect_signals(files))
    assert [name for batch in fake.batches for name in batch] == ["a0.md", "b0.md", "c0.md"]


def test_images_weigh_in_at_the_rate_the_documents_were_sampled(tmp_path, monkeypatch):
    cfg.analyze_max_files = 2
    pdfs = {f"p{i}.pdf": _write(tmp_path, f"p{i}.pdf") for i in range(4)}
    images = {f"i{i}.png": _write(tmp_path, f"i{i}.png") for i in range(4)}
    fake = _FakeExtract({name: _doc(pdf=(1, [])) for name in pdfs})
    monkeypatch.setattr(analyze, "aextract_batch", fake)
    signals = asyncio.run(collect_signals({**pdfs, **images}))
    assert (signals.pdf.pages, signals.image_files) == (2, 4)
    assert signals.pdf.scanned_share == 0.5


def test_sample_keeps_every_key_at_or_under_the_cap():
    assert sample_keys(["b", "a", "c"], 3) == ["a", "b", "c"]


def test_sample_over_the_cap_is_an_even_stride_over_sorted_keys():
    keys = [f"{root}/{i}" for root in "abcd" for i in range(5)]
    picked = sample_keys(list(reversed(keys)), 4)
    assert picked == ["a/0", "b/0", "c/0", "d/0"]
    assert sample_keys(keys, 4) == sample_keys(keys, 4)


def test_an_empty_corpus_gives_zero_shares_and_no_medians(fake):
    signals = asyncio.run(collect_signals({}))
    assert signals.files_total == 0
    assert signals.code_share == 0.0
    assert signals.pdf.scanned_share == 0.0
    assert signals.pdf.median_pages is None
    assert signals.median_chars is None
    assert signals.languages == ()
    assert fake.batches == []


def test_a_folder_of_images_only_is_all_scans(tmp_path, fake):
    signals = asyncio.run(collect_signals({"a.png": _write(tmp_path, "a.png")}))
    assert (signals.documents_total, signals.files_read, signals.image_files) == (0, 0, 1)
    assert signals.pdf.scanned_share == 1.0
    assert fake.batches == []


def test_ocr_language_supported_asks_the_tesseract_backend(monkeypatch):
    import xberg

    calls: list[tuple[str, str]] = []

    def _supports(backend: str, language: str) -> bool:
        calls.append((backend, language))
        return language == "deu"

    monkeypatch.setattr(xberg, "ocr_backend_supports_language", _supports)
    assert analyze.ocr_language_supported("deu") is True
    assert analyze.ocr_language_supported("zho") is False
    assert calls == [("tesseract", "deu"), ("tesseract", "zho")]


def test_real_extraction_reads_pages_and_language_with_ocr_off(tmp_path):
    english = (
        "The government announced new rules today that will apply to every citizen. " * 10
    ).encode()
    files = {
        "doc.pdf": _write(tmp_path, "doc.pdf", make_pdf(pages=3)),
        "notes.txt": _write(tmp_path, "notes.txt", english),
    }
    signals = asyncio.run(collect_signals(files))
    assert signals.failed == ()
    assert (signals.pdf.files, signals.pdf.pages, signals.pdf.scanned_pages) == (1, 3, 0)
    assert signals.languages == (LanguageShare("eng", 1.0),)
    assert signals.median_chars is not None
    assert signals.median_chars > len(english) / 2


def test_analysis_config_turns_ocr_off_and_languages_and_tables_on():
    from lilbee.data.extract.document import analysis_config

    cfg.extraction_timeout = 30
    config = analysis_config()
    assert config.disable_ocr is True
    assert config.ocr is not None and config.ocr.enabled is False
    assert config.language_detection is not None
    assert (config.language_detection.enabled, config.language_detection.detect_multiple) == (
        True,
        True,
    )
    assert config.pdf_options is not None and config.pdf_options.extract_tables is True
    assert config.pages is not None and config.pages.extract_pages is True
    assert config.layout is None
    assert config.chunking is None
    assert config.extraction_timeout_secs == 30


def test_collect_signals_extracts_with_the_analysis_config(corpus, monkeypatch):
    from lilbee.data.extract.document import analysis_config

    seen = []

    async def _capture(items, config):
        seen.append(config)
        return [RuntimeError("x") for _ in items]

    monkeypatch.setattr(analyze, "aextract_batch", _capture)
    asyncio.run(collect_signals(corpus))
    assert seen and all(config == analysis_config() for config in seen)
