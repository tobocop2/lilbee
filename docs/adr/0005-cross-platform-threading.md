# 5. Cross-platform threading quirks

## Status
Accepted

## Context
lilbee runs on macOS, Linux, and Windows across Python 3.11-3.13. Each platform has different threading and I/O behaviors that surfaced as intermittent hangs or deadlocks.

## Finding

**Windows: no select() for pipes.** `os.read()` on a pipe blocks indefinitely. Unix can use `select()` to check readiness, Windows cannot for pipes. Fix: `os.set_blocking(False)` before reading, catch `BlockingIOError`, restore in `finally`.

**macOS ARM64: GPU thread safety with llama-cpp.** Releasing a llama-cpp model lock without finishing inference leaves the Metal GPU context inconsistent. Next streaming call deadlocks. Fix: always drain the C-level iterator before releasing the lock.

**Python 3.11 vs 3.12: executor thread daemon status.** `asyncio.run()` creates `ThreadPoolExecutor` threads. On 3.11, these are non-daemon (blocking process exit). On 3.12+, interpreter shutdown handles this. Fix: runtime version detection and daemon thread patching in conftest.

## Consequences
- Platform-specific code paths are annotated with comments explaining why
- CI matrix covers all three platforms and Python 3.11/3.12/3.13
- These issues only surface under parallel test execution or long-running sessions
