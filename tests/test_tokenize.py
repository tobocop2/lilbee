"""Tests for the RAG query tokenizer and IDF weights."""

import math

import pytest

from lilbee.retrieval.query.tokenize import _idf_weights, _tokenize


class TestTokenize:
    def test_keeps_single_character_terms(self) -> None:
        assert "c" in _tokenize("How do I use C for this?")

    def test_keeps_hyphen_split_letters_and_digits(self) -> None:
        assert _tokenize("W-2") == ["w", "2"]

    def test_drops_empty_fragments(self) -> None:
        assert _tokenize("") == []
        assert _tokenize("...") == []


class TestIdfWeights:
    def test_caps_single_character_weight(self) -> None:
        weights = _idf_weights({"s", "deploy"}, [{"deploy"}, set(), set(), set(), set(), set()])
        assert weights["s"] == pytest.approx(1.0)
        assert weights["deploy"] == pytest.approx(math.log(6 / 2))

    def test_leaves_small_single_character_weight_uncapped(self) -> None:
        weights = _idf_weights({"r"}, [{"r"}, {"other"}, {"filler"}])
        assert weights["r"] == pytest.approx(math.log(3 / 2))
