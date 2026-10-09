"""Tokens: what counts as a word, what markdown syntax is parsed away, and what is never dropped."""

from __future__ import annotations

import unicodedata

from tools.qa.crawl_parity.tokens import (
    NORMALISATION_NOTES,
    Normalisation,
    markdown_text,
    markdown_tokens,
    missing,
    tokenize,
)


def test_words_and_symbols_are_separate_classes() -> None:
    tokens = tokenize("Hello, world_1 [ ] <!-- x -->")
    assert tokens.words == ("Hello", "world", "1", "x")
    assert tokens.symbols == (",", "_", "[", "]", "<!--", "-->")


def test_each_ideograph_and_kana_is_one_token() -> None:
    assert tokenize("日本語 テキスト abc").words == (
        "日",
        "本",
        "語",
        "テ",
        "キ",
        "ス",
        "ト",
        "abc",
    )


def test_composed_and_decomposed_text_give_the_same_tokens() -> None:
    composed = unicodedata.normalize("NFC", "café naïve")
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert tokenize(composed) == tokenize(decomposed)


def test_case_is_not_normalised() -> None:
    assert tokenize("Word").words != tokenize("word").words


def test_markdown_markers_and_link_targets_are_not_tokens() -> None:
    tokens = markdown_tokens("# Title\n\n* **bold** [text](http://example.test/target) `code`\n")
    assert tokens.words == ("Title", "bold", "text", "code")
    assert tokens.symbols == ()


def test_an_underscore_marker_left_unparsed_makes_no_new_word() -> None:
    assert markdown_tokens("a _marked_ word").words == markdown_tokens("a\\_marked\\_ word").words
    assert tokenize("snake_case_name").words == ("snake", "case", "name")


def test_alternative_text_of_touching_images_stays_separate_words() -> None:
    touching = "![first one](a.png)![second](b.png)tail"
    assert markdown_tokens(touching).words == ("first", "one", "second", "tail")


def test_image_alternative_text_is_text() -> None:
    assert markdown_tokens("![a photo](x.png)").words == ("a", "photo")


def test_code_block_and_raw_html_are_text() -> None:
    words = markdown_tokens("```\nfenced words\n```\n\n<!-- hidden note -->\n").words
    assert words == ("fenced", "words", "hidden", "note")


def test_a_row_with_more_cells_than_its_header_keeps_every_cell() -> None:
    table = "| a |\n|---|\n| b | surplus |\n"
    assert "surplus" in markdown_tokens(table).words


def test_table_syntax_is_not_a_symbol_and_other_symbols_stay() -> None:
    tokens = markdown_tokens("| a | b |\n|:--|--:|\n| c? | d |\n")
    assert tokens.words == ("a", "b", "c", "d")
    assert tokens.symbols == ("?",)


def test_line_breaks_separate_words() -> None:
    assert markdown_text("one\ntwo  \nthree").split() == ["one", "two", "three"]


def test_missing_counts_occurrences() -> None:
    have = tokenize("a b").word_counts()
    want = tokenize("a a b c").word_counts()
    assert missing(have, want) == {"a": 1, "c": 1}


def test_every_normalisation_has_a_note_for_the_report() -> None:
    assert set(NORMALISATION_NOTES) == set(Normalisation)
    assert len(NORMALISATION_NOTES) == 4
