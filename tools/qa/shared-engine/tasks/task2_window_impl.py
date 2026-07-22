"""Task 2: fix the off-by-one so the tests pass. Do not change the tests."""


def last_n_lines(text: str, n: int) -> list[str]:
    """The final *n* lines of *text*, fewer when text is shorter."""
    lines = text.splitlines()
    return lines[-(n - 1) :]  # BUG: drops one line
