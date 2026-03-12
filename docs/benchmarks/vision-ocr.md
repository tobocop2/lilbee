# Vision OCR Model Benchmarks

Benchmark results for local vision models used for scanned PDF OCR extraction.

## Test Setup

- **Date:** 2026-03-11
- **Hardware:** Apple M1 Pro, 32 GB RAM, macOS Sequoia 15.3.2
- **Software:** Ollama 0.17.7, Python 3.12
- **Test corpus:** Star Wars X-Wing Collector's Edition manual, page 5 — a dense scanned page with headings, body text, and a special note callout
- **Image:** PNG rasterized from PDF at 300 DPI (3,237,376 bytes)

## Methodology

Each model was cold-started (no prior inference in session) and run via the `ollama.chat()` Python API with the same prompt and image. Wall-clock time was measured from API call to response. Quality was assessed qualitatively by comparing output against the original scanned page.

**Prompt used:**
```
Extract ALL text from this page as clean markdown. Preserve table structure.
```

## Results

| Model | Params | Size | Time | Chars | Quality |
|-------|--------|------|------|-------|---------|
| **maternion/LightOnOCR-2** | 1B | 1.5 GB | **11.9s** | 3380 | **Best** — clean markdown with `#` headers, `>` blockquotes, `---` separators. Near-perfect accuracy. |
| deepseek-ocr | 3B | 6.7 GB | 17.4s | 3228 | Excellent — near-perfect plain text, preserves original line breaks. No markdown structure. |
| minicpm-v | 8B | 5.5 GB | 35.6s | 3335 | Good — some transcription errors ("spacedcraft", "Play special attention"), dropped a paragraph. |
| glm-ocr | 0.9B | 2.2 GB | 51.7s | 3183 | Good — accurate text, no markdown formatting. Surprisingly slow despite tiny parameter count. |
| granite3.2-vision:2b | 2B | 2.4 GB | 36.7s | 3375 | **Bad** — forced all content into a markdown table, garbled and truncated text throughout. |

## Key Findings

- **LightOnOCR-2 is the clear winner:** fastest, smallest, and produces the best-structured output with clean markdown formatting
- **DeepSeek-OCR is the accuracy runner-up** with very clean text extraction but no markdown formatting
- **Model size does not correlate with quality:** the 1B model beat the 8B model in both speed and accuracy
- **Vision OCR is only useful for scanned/image PDFs** — for text-based PDFs, Kreuzberg's text extraction is faster and more accurate

## Limitations

- Single test page (results may vary with different document types: handwriting, tables, forms, multi-column layouts)
- Single hardware configuration (Apple Silicon — results will differ on NVIDIA GPUs or other platforms)
- Qualitative quality assessment (no automated metrics like CER/WER)
- Cold-start timing only (subsequent runs may be faster due to model caching)
