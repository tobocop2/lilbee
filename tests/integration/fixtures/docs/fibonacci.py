"""Fibonacci sequence calculator."""


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively.
    Args:
        n: The position in the Fibonacci sequence (0-indexed).

    Returns:
        The nth Fibonacci number.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(10)
        55
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
