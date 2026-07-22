"""Task 1: implement slugify() in slugify_impl.py beside this file."""

from slugify_impl import slugify


def test_lowercases_and_hyphenates():
    assert slugify("Hello World") == "hello-world"


def test_strips_punctuation():
    assert slugify("What's up, Doc?") == "whats-up-doc"


def test_collapses_runs_of_separators():
    assert slugify("a  --  b") == "a-b"


def test_trims_edge_separators():
    assert slugify("  -hello-  ") == "hello"
