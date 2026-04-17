# 1. pytest-xdist hang: four cascading root causes

## Status
Accepted

## Context
pytest-xdist parallelism froze at 99% completion on Python 3.11 across CI platforms. Multiple speculative fixes were attempted and reverted before root cause analysis revealed four independent issues compounding.

## Finding
The "hang" was not one bug but four cascading platform-specific thread lifecycle problems:

1. **Python 3.11 non-daemon executor threads**: `asyncio.run()` inside Textual's `@work(thread=True)` creates `ThreadPoolExecutor` threads. On Python 3.11 these are non-daemon (3.12+ marks them daemon at interpreter shutdown). Non-daemon threads block process exit.

2. **Litestar QueueListener thread accumulation**: `_drain_textual_threads` joined ALL new threads with 2s timeout, including litestar `QueueListener` daemon threads that accumulate across 3500+ tests. Hundreds of 2s joins caused apparent hangs.

3. **Windows pipe blocking**: `pipe_closed()` used blocking `os.read()` on Windows. Without `select()` (unavailable on Windows for pipes), an open pipe blocks forever. Fix: `os.set_blocking(False)` before reading, restore in `finally`.

4. **macOS llama-cpp lock/drain ordering**: `_LockedStreamIterator.close()` released the lock without finishing inference, leaving the model inconsistent on ARM64 macOS. Next streaming call deadlocked. Fix: drain the C-level iterator before releasing the lock.

## Consequences
- conftest.py patches `Thread.__init__` to mark executor worker threads as daemon on Python 3.11
- Thread drain only joins non-daemon threads
- Windows uses non-blocking I/O for pipe reads
- llama-cpp streaming always drains before lock release
- Correlation-based "fixes" (reducing parallelism, serial tests) were reverted because they masked rather than solved the issue
