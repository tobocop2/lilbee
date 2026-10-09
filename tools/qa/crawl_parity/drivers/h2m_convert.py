"""Convert HTML with html-to-markdown alone, with the options crawlberg passes for lilbee."""

from __future__ import annotations

import html_to_markdown
from _driver_io import serve_conversions


def convert(html: str, base_url: str) -> str:
    """The converter's markdown for *html*."""
    options = html_to_markdown.ConversionOptions(
        include_document_structure=True,
        preprocessing=html_to_markdown.PreprocessingOptions(
            enabled=True, preset="minimal", remove_navigation=False, remove_forms=False
        ),
        exclude_selectors=[],
        inline_data_media="alt_text_only",
        extract_metadata=False,
        base_url=base_url,
    )
    return str(html_to_markdown.convert(html, options).content)


if __name__ == "__main__":
    serve_conversions(convert)
