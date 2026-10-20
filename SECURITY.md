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

### Provider client (`lilbee[providers]` extra)

lilbee reaches hosted model providers through a client library, used **only as a client SDK** inside one adapter module (`src/lilbee/providers/backends/`, the only package that imports it). lilbee does **not** run a proxy server.

Earlier releases used LiteLLM for this role. LiteLLM is no longer a dependency, so its Proxy server advisories (GHSA-xqmj-j6mv-4862, GHSA-r75f-5x8p-qvmc, GHSA-v4p8-mg3p-g94g, GHSA-wxxx-gvqv-xp7p) do not apply to current releases. An advisory that affects the client library's request handling is in scope and patched.
