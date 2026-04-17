# 3. SSRF protection and URL validation

## Status
Accepted

## Context
Web crawling accepts user-provided URLs. Without validation, a user (or MCP client) could request crawling of internal network addresses, cloud metadata endpoints (169.254.x.x), or non-HTTP protocols.

## Decision
Comprehensive URL validation via shared `validate_crawl_url()` and `require_valid_crawl_url()`, applied consistently across CLI, MCP, TUI, and REST API:

- DNS resolution check (hostname must resolve)
- Block private/reserved IPs: 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, ::1
- Enforce HTTP/HTTPS only (reject file://, data://, ftp://)
- Validate hostname exists and is non-empty

## Consequences
- Cannot crawl localhost or private network addresses
- Validation is centralized, reducing risk of bypass via alternative entry points
- DNS resolution adds a small latency cost per crawl request
