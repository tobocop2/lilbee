"""Tests for the shared tokenization helpers in ``lilbee.text``."""

from __future__ import annotations

from lilbee.text import STOP_WORDS, tokenize, tokenize_unique


class TestStopWords:
    def test_contains_common_english_stopwords(self):
        for word in ("the", "and", "with", "from", "that", "this"):
            assert word in STOP_WORDS

    def test_is_frozen(self):
        # frozensets have no add method
        assert not hasattr(STOP_WORDS, "add")


class TestTokenize:
    def test_basic_alphanumeric(self):
        assert tokenize("python typing protocol") == ["python", "typing", "protocol"]

    def test_drops_stopwords_at_or_above_min_length(self):
        # "the", "and" are both short but exercise the stopword branch
        # for tokens that pass the length filter: "has" is 3 chars.
        assert tokenize("python has typing") == ["python", "typing"]
        assert tokenize("the quick brown fox") == ["quick", "brown", "fox"]

    def test_drops_short_non_stopword_tokens(self):
        # "py" is 2 characters, not a stopword — still dropped by length
        assert tokenize("py python") == ["python"]

    def test_strips_non_alphanumeric_characters(self):
        assert tokenize("kafka!!! streams. topics?") == ["kafka", "streams", "topics"]

    def test_preserves_duplicates_for_tf_counting(self):
        assert tokenize("python python python") == ["python", "python", "python"]

    def test_lowercases(self):
        assert tokenize("Python TYPING") == ["python", "typing"]

    def test_empty_input_returns_empty(self):
        assert tokenize("") == []

    def test_all_punctuation_returns_empty(self):
        assert tokenize("!!! ??? ...") == []


class TestTokenizeUnique:
    def test_returns_a_set(self):
        result = tokenize_unique("python python typing")
        assert isinstance(result, set)
        assert result == {"python", "typing"}

    def test_drops_stopwords_and_single_chars(self):
        assert tokenize_unique("a the python typing x") == {"python", "typing"}

    def test_lowercases(self):
        assert tokenize_unique("Python TYPING") == {"python", "typing"}

    def test_empty_input_returns_empty_set(self):
        assert tokenize_unique("") == set()
