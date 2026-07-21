"""Tests for the shared side_effect tail helper."""

from unittest import mock

import pytest

from tests._mock_effects import repeat_last


def test_surplus_calls_return_the_last_effect():
    m = mock.Mock(side_effect=repeat_last(1, 2))
    assert [m(), m(), m(), m()] == [1, 2, 2, 2]


def test_exception_effects_still_raise():
    m = mock.Mock(side_effect=repeat_last(ValueError("boom"), "ok"))
    with pytest.raises(ValueError, match="boom"):
        m()
    assert m() == "ok"
    assert m() == "ok"


def test_single_effect_repeats():
    m = mock.Mock(side_effect=repeat_last("only"))
    assert [m(), m()] == ["only", "only"]
