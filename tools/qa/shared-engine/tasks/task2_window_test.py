from task2_window_impl import last_n_lines


def test_returns_exactly_n_lines():
    assert last_n_lines("a\nb\nc\nd", 2) == ["c", "d"]


def test_short_text_returns_everything():
    assert last_n_lines("a\nb", 5) == ["a", "b"]


def test_single_line():
    assert last_n_lines("only", 1) == ["only"]
