# Obsidian Plugin — Feature Parity Update

Plugin repo: `/Users/tobias/projects/obsidian-lilbee`
Branch: `feature/server-api-parity` (PR #7 → main)
Also: `feature/catalog-families` has model removal + enriched API (needs merge into server-api-parity)

## What the TUI now has that the plugin needs to match

### 1. Model dropdowns
TUI now has Select widgets for chat/embed/vision model switching (commit `39703ee`).
Plugin currently has model switching in settings only — should also have it in the chat header or sidebar.

### 2. Editable settings
TUI settings screen is now editable (commit `271b502`). Press Enter to edit inline.
Plugin has editable settings in its settings tab — already done. No action needed.

### 3. Quality tier labels
TUI shows "balanced", "high quality", "compact" instead of Q4_K_M etc.
Plugin has `quality_tier` field from enriched API — already rendering it. No action needed.

### 4. Clean display names
TUI shows "Qwen 2.5 7B" not "Qwen2.5-7B-Instruct-GGUF".
Plugin has `display_name` from enriched API — already using it. No action needed.

### 5. Model removal from catalog
TUI has `d` key to delete installed models from catalog.
Plugin has Remove button in catalog modal (commit `d19d69a` on catalog-families, cherry-picked to server-api-parity at `5b9a0fd`). Done.

### 6. HF token management
TUI has `/login` command + `huggingface_hub.get_token()` fallback.
Plugin has `hfToken` settings field that passes `HF_TOKEN` env var to server process. Done.

## What the plugin has that the TUI doesn't
- Document removal modal (search + select + delete)
- Crawl modal (URL + depth + max pages form)
- Setup wizard (5-step guided onboarding)
- Auto-sync with debounce on vault changes
- Binary manager (downloads/updates lilbee executable)
- Chat export to vault markdown

## Plugin PRs to merge
1. **PR #7** (`feature/server-api-parity` → `main`) — the big release: Ollama removal, catalog, crawl, documents, settings redesign. MERGEABLE.
2. **PR #9** (`feature/catalog-families` → `server-api-parity`) — hierarchical families, setup wizard, model removal. Has enriched API fields. NEEDS REBASE after server-api-parity changes.

## Server API endpoints the plugin uses
All exist and work. Key ones:
- `GET /api/models/catalog` — returns enriched models (display_name, quality_tier)
- `POST /api/models/pull` — SSE download progress
- `DELETE /api/models/{model}` — model removal
- `PUT /api/models/chat|vision|embedding` — model switching
- `POST /api/chat/stream` — SSE chat with reasoning
- `POST /api/sync` — SSE sync progress
- `POST /api/add` — SSE file add
- `PATCH /api/config` — update settings
- `GET /api/documents` — document listing
- `POST /api/documents/remove` — document deletion

## Action items for plugin parity
1. Merge PR #7 into main
2. Rebase PR #9 onto updated server-api-parity, then merge
3. Consider adding model Select dropdowns in chat view header (matching TUI)
4. No other gaps — plugin is ahead of TUI in some areas (document modal, crawl modal)
