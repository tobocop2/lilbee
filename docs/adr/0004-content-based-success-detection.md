# 4. Content-based success detection over status flags

## Status
Accepted

## Context
crawl4ai sets `result.success = False` when sub-resource fetches fail (e.g., a favicon 404), even when the main page has valid markdown content. Relying on the success flag caused valid pages to be rejected.

## Decision
Trust the content, not the status flag. Check actual markdown content independently of the library's status indicator. If `result.markdown` has content, treat the crawl as successful regardless of `result.success`.

## Consequences
- Valid pages are no longer rejected due to unrelated sub-resource failures
- Integration tests with real crawl4ai (via pytest-httpserver) caught this. Mocked tests did not.
- General principle: third-party library status indicators can be misleading. Validate the actual content independently.
