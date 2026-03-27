# TODO

## Kreuzberg: per-page progress callback

kreuzberg's `extract_file()` is a single async call with no way to report progress during extraction. lilbee's vision OCR path has a nice per-page progress bar because we control the loop, but kreuzberg-based extraction (text + Tesseract) is a black box.

**Feature request for kreuzberg:** Add an optional progress callback to `extract_file()` — something like `on_page(page_num, total_pages)` — so callers can display per-page progress bars during PDF/EPUB extraction.

This would let lilbee show the same Rich progress bar for all PDF ingestion paths, not just vision OCR.
