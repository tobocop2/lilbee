# Security Policy

## Supported Versions

Only the latest release is supported with security updates.

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, use one of these methods:

1. **GitHub Private Vulnerability Reporting** — use the "Report a vulnerability" button on the
   [Security tab](https://github.com/tobocop2/lilbee/security/advisories/new)
2. **Email** — contact [@tobocop2](https://github.com/tobocop2) directly

### What to expect

- Acknowledgment within 72 hours
- Status update within 1 week
- We will coordinate disclosure timing with you

## Dependency advisories

### LiteLLM (`lilbee[litellm]` extra)

lilbee uses `litellm` **only as a client SDK** — `litellm.completion()` / `litellm.embedding()` calls to hosted model providers, routed through a single adapter (`src/lilbee/providers/litellm_sdk.py`, the only module that imports `litellm`). lilbee does **not** run the LiteLLM Proxy server, does not expose any of its admin or test endpoints (`/guardrails/test_custom_code`, `/prompts/test`, the proxy key-verification path, the MCP stdio test endpoints), and does not configure LiteLLM guardrails or custom guardrail code.

Consequently, advisories in the LiteLLM **Proxy server** — e.g. GHSA-xqmj-j6mv-4862, GHSA-r75f-5x8p-qvmc, GHSA-v4p8-mg3p-g94g, GHSA-wxxx-gvqv-xp7p (CVE-2026-40217) — do not affect lilbee, and are dismissed in this repository's Dependabot alerts as "vulnerable code not used". An advisory that affects the LiteLLM *client SDK* path (completion/embedding request handling) would be in scope and patched.
