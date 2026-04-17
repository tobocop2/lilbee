# 7. TUI worker thread testing patterns

## Status
Accepted

## Context
Textual runs `@work(thread=True)` background workers that race with test assertions. Initial approaches mocked the decorator or individual lifecycle methods (`on_mount`, `on_show`), but these were fragile across platforms.

## Finding
The testing approach evolved through several iterations:

1. Mock specific methods (`_scan_models`, `ModelBar.on_mount`, `ChatScreen.on_show`) - brittle, broke on refactors
2. Per-test executor shutdown to prevent thread accumulation - better but still had races
3. Shared `call_from_thread` wrapper that silently drops calls during app shutdown - helped but didn't solve testing
4. **Final approach**: extract the logic being tested into a regular method, mock that, let the `@work` lifecycle complete normally

## Decision
Mock implementations, not decorators. If a `@work`-decorated method calls `self._do_heavy_thing()`, mock `_do_heavy_thing` rather than trying to mock or suppress the worker machinery.

## Consequences
- Tests are resilient to Textual framework changes
- Worker lifecycle runs normally (no framework internals mocked)
- Extracted methods are independently testable without the TUI
- Slight increase in indirection (thin `@work` wrapper + implementation method)
