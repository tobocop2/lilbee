"""Mathematical utility functions."""

_fib_cache: dict[int, int] = {}


def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using memoised recursion."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    if n in _fib_cache:
        return _fib_cache[n]
    result = fibonacci(n - 1) + fibonacci(n - 2)
    _fib_cache[n] = result
    return result


def celsius_to_fahrenheit(temp: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return temp * 9.0 / 5.0 + 32.0
