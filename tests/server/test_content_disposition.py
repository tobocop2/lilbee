"""The attachment header: an ASCII fallback name plus the exact UTF-8 name."""

from __future__ import annotations

import pytest

from lilbee.server.content_disposition import attachment_disposition


@pytest.mark.parametrize(
    ("filename", "header"),
    [
        ("notes.html", "attachment; filename=\"notes.html\"; filename*=UTF-8''notes.html"),
        ('a"b.html', "attachment; filename=\"a_b.html\"; filename*=UTF-8''a%22b.html"),
        ("a\\b.html", "attachment; filename=\"a_b.html\"; filename*=UTF-8''a%5Cb.html"),
        ("制动.md", "attachment; filename=\"__.md\"; filename*=UTF-8''%E5%88%B6%E5%8A%A8.md"),
        ("a\r\n\x85b.md", "attachment; filename=\"a___b.md\"; filename*=UTF-8''a%0D%0A%C2%85b.md"),
    ],
    ids=["plain", "quote", "backslash", "non-ascii", "control"],
)
def test_the_header_names_any_file_validly(filename, header):
    assert attachment_disposition(filename) == header
