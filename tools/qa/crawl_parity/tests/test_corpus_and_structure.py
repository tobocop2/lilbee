"""The synthetic corpus loads whole, and the structure signature sees each element."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.qa.crawl_parity import plants, structure
from tools.qa.crawl_parity.converter import declared_charset, decode
from tools.qa.crawl_parity.corpus import SYNTHETIC_DIR, AlternateWhen, Response, load_synthetic
from tools.qa.crawl_parity.model import Mode, Page
from tools.qa.crawl_parity.structure import Element
from tools.qa.crawl_parity.thresholds import THRESHOLDS_FILE, load_thresholds
from tools.qa.crawl_parity.tokens import markdown_tokens

DOCUMENT = """# Title

## Sub *part*

Text with [a link](http://example.test/a) and [another](/b).

- one
- two

```python
print("x")
```

| a | b |
|---|---|
| c | d |
"""


def test_the_synthetic_corpus_has_every_group_and_each_seed_is_a_record() -> None:
    corpus = load_synthetic()
    assert {seed.name for seed in corpus.seeds} == {"f", "b", "n", "c", "w"}
    assert len(corpus.records) > 100
    assert all(seed.path in corpus.records for seed in corpus.seeds)
    assert corpus.seed("w").modes == (Mode.HTTP,)
    assert len(corpus.html_pages()) > 80


def test_every_body_file_of_the_corpus_is_used_by_a_record() -> None:
    used = (SYNTHETIC_DIR / "corpus.toml").read_text(encoding="utf-8")
    files = [
        path for path in SYNTHETIC_DIR.rglob("*") if path.is_file() and path.name != "corpus.toml"
    ]
    assert len(files) > 100
    assert [path for path in files if path.relative_to(SYNTHETIC_DIR).as_posix() not in used] == []


def test_wire_bytes_and_alternates_are_loaded_as_written() -> None:
    corpus = load_synthetic()
    gzipped = corpus.records["/n/gzip"].response
    assert gzipped.body[:2] == b"\x1f\x8b"
    assert ("Content-Encoding", "gzip") in gzipped.headers
    sjis = corpus.records["/n/sjis-header"].response
    with pytest.raises(UnicodeDecodeError):
        sjis.body.decode("utf-8")
    assert "日本語" in sjis.body.decode("shift_jis")
    need = corpus.records["/c/need"]
    assert need.alternate is not None
    assert (need.alternate.when, need.alternate.response.status) == (AlternateWhen.NO_COOKIE, 403)
    flaky = corpus.records["/n/s429"].alternate
    assert flaky is not None and (flaky.when, flaky.count) == (AlternateWhen.FIRST_REQUESTS, 2)
    assert corpus.records["/n/slow3"].delay_ms == 3000
    assert corpus.records["/n/r301"].response.status == 301


def test_a_charset_comes_from_the_header_then_the_meta_tag_then_utf8() -> None:
    header = Response(
        200, (("Content-Type", "text/html; charset=iso-8859-1"),), "café".encode("latin-1")
    )
    meta = Response(
        200,
        (("Content-Type", "text/html"),),
        b'<meta charset="shift_jis">' + "語".encode("shift_jis"),
    )
    bare = Response(200, (), "é".encode())
    unknown = Response(200, (("Content-Type", "text/html; charset=no-such"),), b"plain")
    assert (declared_charset(header), decode(header)) == ("iso-8859-1", "café")
    assert declared_charset(meta) == "shift_jis" and decode(meta).endswith("語")
    assert (declared_charset(bare), decode(bare)) == ("utf-8", "é")
    assert decode(unknown) == "plain"


def test_the_signature_counts_each_element() -> None:
    found = structure.signature(DOCUMENT)
    assert {element: found.count(element) for element in Element} == {
        Element.HEADING: 2,
        Element.LINK: 2,
        Element.CODE_BLOCK: 1,
        Element.TABLE: 1,
        Element.LIST_ITEM: 2,
    }
    assert found.items[Element.HEADING] == {"h1 Title": 1, "h2 Sub part": 1}
    assert found.items[Element.LINK] == {"http://example.test/a [a link]": 1, "/b [another]": 1}
    assert found.items[Element.TABLE] == {"2 rows, 4 cells": 1}


def test_equal_documents_have_no_structure_delta_and_each_change_is_named() -> None:
    assert structure.compare(structure.signature(DOCUMENT), structure.signature(DOCUMENT)) == []
    changed = DOCUMENT.replace("## Sub", "### Sub").replace("(/b)", "(/c)")
    deltas = structure.compare(structure.signature(DOCUMENT), structure.signature(changed))
    assert [delta.element for delta in deltas] == [Element.HEADING, Element.LINK]
    assert deltas[0].only_reference == {"h2 Sub part": 1}
    assert deltas[0].only_other == {"h3 Sub part": 1}


def test_a_flattened_table_loses_the_table_and_keeps_every_word() -> None:
    flattened = plants.flatten_tables(DOCUMENT)
    assert structure.signature(flattened).count(Element.TABLE) == 0
    assert markdown_tokens(flattened).word_counts() == markdown_tokens(DOCUMENT).word_counts()


def test_sentence_plants_change_exactly_one_sentence() -> None:
    markdown = "# T\n\nFirst plain sentence here now.\n\nSecond one.\n"
    chosen = plants.plain_sentence(markdown)
    assert chosen == "First plain sentence here now."
    assert chosen not in plants.delete_sentence(markdown, chosen)
    assert plants.duplicate_sentence(markdown, chosen).count(chosen) == 2
    with pytest.raises(LookupError, match="no line that is plain text"):
        plants.plain_sentence("# T\n\nshort\n")


def test_page_plants_copy_and_do_not_change_their_input() -> None:
    original = {"/a": Page("u/a", "one"), "/b": Page("u/b", "two")}
    assert plants.without_page(original, "/a") == {"/b": Page("u/b", "two")}
    assert plants.with_markdown(original, "/a", "new")["/a"] == Page("u/a", "new")
    assert original["/a"].markdown == "one" and len(original) == 2


def test_every_threshold_is_proposed_and_an_unknown_key_is_refused(tmp_path: Path) -> None:
    text = THRESHOLDS_FILE.read_text(encoding="utf-8")
    values = [line for line in text.splitlines() if "=" in line and not line.startswith("#")]
    assert len(values) == 19
    assert text.count("# PROPOSED") == len(values)
    limits = load_thresholds()
    assert limits.parity.pages_lost_max == 0 and limits.speed.repeats == 3
    broken = tmp_path / "thresholds.toml"
    broken.write_text(text.replace("pages_lost_max", "pages_lost_maximum"), encoding="utf-8")
    with pytest.raises(TypeError, match="pages_lost_maximum"):
        load_thresholds(broken)
