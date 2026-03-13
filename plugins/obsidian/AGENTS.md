# lilbee Obsidian Plugin — Development Guide

## Project

Obsidian plugin for [lilbee](../../), a local RAG knowledge base. Communicates with the lilbee HTTP server (`lilbee serve`) over localhost. TypeScript, esbuild, Vitest.

## Commands

```bash
npm run build        # Production build → main.js
npm run dev          # Watch mode (esbuild)
npm test             # Vitest with v8 coverage (100% required)
npm run test:watch   # Vitest watch mode
```

From the repo root: `make plugin-build`, `make plugin-test`, `make plugin-dev`.

## Architecture

```
src/
  main.ts           # Plugin entry: lifecycle, commands, auto-sync
  types.ts          # All interfaces + DEFAULT_SETTINGS
  api.ts            # LilbeeClient — typed HTTP client with SSE streaming
  settings.ts       # Settings tab + model management UI
  views/
    results.ts      # Render helpers: document cards, source chips
    search-modal.ts # Search/ask modal (Modal)
    chat-view.ts    # Chat sidebar (ItemView)
```

**Key design decisions:**
- `LilbeeClient` is the only module that calls `fetch`. All other modules go through it.
- Streaming endpoints (chat, sync, model pull) use `async*` generators yielding `SSEEvent`.
- Views create DOM in `onOpen()`, never in the constructor.
- All CSS classes are namespaced `lilbee-*`.
- Settings are merged with `DEFAULT_SETTINGS` on load for forward compatibility.

## Code Style

- **No `any`** — define interfaces in `types.ts`; use `unknown` + cast at SSE boundaries only.
- **Null guards** — always check container elements exist before manipulating DOM.
- **Error handling** — wrap all API calls in try/catch, show `new Notice(message)` on failure.
- **Constants** — named constants for magic numbers (`MAX_EXCERPT_CHARS`, `SEARCH_DEBOUNCE_MS`). Use `as const` objects for string literal sets (`SSE_EVENT`, `JSON_HEADERS`).
- **Obsidian DOM helpers** — use `createDiv()`, `createEl()`, `setText()`, `addClass()`. No `innerHTML`.
- **Import order** — obsidian imports first, then local modules.
- **Strict TypeScript** — `noImplicitAny`, `strictNullChecks` enabled in tsconfig.

## TypeScript Best Practices

### Const Assertions for String Unions
Use `as const` objects for string literal sets shared across modules. Prefer this over enums:
```typescript
export const SSE_EVENT = {
    TOKEN: "token",
    DONE: "done",
} as const;

// Type: "token" | "done"
type SSEEventType = (typeof SSE_EVENT)[keyof typeof SSE_EVENT];
```

### Discriminated Unions
Model state transitions with discriminated unions, not boolean flags:
```typescript
type StreamEvent =
    | { type: "token"; data: string }
    | { type: "done"; data: null }
    | { type: "error"; data: string };
```

### Error Handling
- Wrap all `fetch` calls through a centralized `assertOk()` pattern.
- Use `try/catch` at the call site, never inside the HTTP client.
- Surface errors to users via `new Notice()`, never `console.error` alone.

### Function Length
- Target max 30 lines per function. Extract helpers when exceeding this.
- Settings display code may exceed this for fluent builder chains — acceptable when each builder call is a single statement.

### Exports
- Export only what other modules need. Keep helpers and internal functions unexported.
- Exception: export for testing is acceptable but should be noted with a comment.

### Null Handling
- Prefer `T | null` over `T | undefined` for data model fields (matches Python server).
- Use `undefined` only for optional function parameters.
- Guard nullable DOM references before use: `if (!this.el) return;`

### Type Assertions
- Minimize `as` casts. Use them only at serialization boundaries (SSE event data).
- Never use `as any` in production code. Use `as unknown as T` in tests for mock compatibility.

### Immutable Defaults
- Copy default objects with spread: `{ ...DEFAULT_SETTINGS }` to avoid shared mutation.
- Use `as const` for constant objects to prevent accidental mutation.

## Testing

### Requirements
- **100% coverage** on statements, functions, lines (branches: 99% threshold due to one v8 artifact).
- Every new public function needs tests. Every branch needs coverage.
- Tests run in Node, not a browser — there is no real DOM.

### Mock Architecture
`tests/__mocks__/obsidian.ts` provides the entire Obsidian API mock:
- `MockElement` — simulates Obsidian's enhanced HTMLElement (`createDiv`, `createEl`, `empty`, `setText`, `addClass`, `addEventListener`).
- Test helpers on MockElement: `find(cls)`, `findAll(cls)`, `trigger(event, ...args)`.
- `App`, `Plugin`, `Modal`, `ItemView`, `PluginSettingTab`, `Notice`, `Setting` — all mocked.
- `Notice.instances` array + `Notice.clear()` for asserting on user notifications.

Vitest aliases `obsidian` to this mock via `vitest.config.ts`.

### Test Patterns
```typescript
// Factory helpers reduce boilerplate
const container = new MockElement() as unknown as HTMLElement;
const app = new App();

// Async generator mocks for streaming
async function* mockStream(events: SSEEvent[]) {
    for (const e of events) yield e;
}

// Collect async generator results
async function collect<T>(gen: AsyncGenerator<T>): Promise<T[]> {
    const items: T[] = [];
    for await (const item of gen) items.push(item);
    return items;
}

// Trigger events on MockElements
element.trigger("click", mockEvent);

// Assert notices
expect(Notice.instances.map(n => n.message)).toContain("expected message");
```

- Use `vi.useFakeTimers()` for debounce tests.
- Use `vi.stubGlobal("fetch", vi.fn())` for API tests.
- Access private methods via `(instance as any).method()` when needed.
- Call `Notice.clear()` in `beforeEach`.

### What NOT to Mock
- Don't mock `types.ts` — use real interfaces.
- Don't mock between view modules — let `renderDocumentResult` run for real in search-modal tests.

## Obsidian API Notes

- `ItemView.containerEl.children[1]` is the content area — standard convention but undocumented.
- `app.workspace.openLinkText(path, "")` opens files by path.
- `app.vault.on("create" | "modify" | "delete" | "rename", callback)` for vault events.
- `Plugin.registerEvent()` for auto-cleanup of event listeners on unload.
- `addStatusBarItem()` returns an element for status text; use `setText()`.

## Build

esbuild bundles `src/main.ts` → `main.js` (CJS, ES2022). Externals: `obsidian`, `electron`, codemirror/lezer packages, Node builtins. The output `main.js` and `manifest.json` are what gets installed into `.obsidian/plugins/lilbee/`.

## Server Dependency

The plugin requires `lilbee serve` running on localhost (default port 7433). Without it, all API calls fail gracefully with user-visible Notice messages.
