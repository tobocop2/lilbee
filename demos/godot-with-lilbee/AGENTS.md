## Instructions

You are writing Godot 4.4 GDScript. Your training data is outdated — many classes and methods were renamed or replaced in Godot 4.x.

**You MUST use the `lilbee_search` MCP tool to look up Godot APIs.** The Godot 4.4 class reference is indexed locally in lilbee. Do NOT use web search, code search, or sub-agents for API lookups — call `lilbee_search` directly.

The MCP server is wired up in `opencode.json` (`"command": ["lilbee", "mcp"]`) and exposes the lilbee toolset under the `lilbee_*` prefix. The relevant tool is:

- **`lilbee_search(query, top_k=5, scope="both")`** — searches the indexed knowledge base. Pass the exact class or method name as `query`. `scope="raw"` restricts to the source documentation chunks (skips the auto-built wiki summaries); `scope="both"` (default) searches both pools.

If you're on Claude Code or another client that uses the `mcp__<server>__<tool>` namespacing, the same tool is `mcp__lilbee__search`. Either form is fine; use whichever your client surfaces.

### Workflow

1. **Plan** — list every Godot class and method your solution needs.
2. **Search** — call `lilbee_search` for each class and method individually. Do not skip this step for any class — even ones you are confident about. Check whether each class still exists or was replaced by a newer variant. Verify every method signature individually. Use `scope="raw"` when you want the original Godot reference text rather than lilbee's wiki summary.
3. **Write** — only use class names, methods, and properties exactly as confirmed by the lilbee search results. If something wasn't found, do not use it.
4. **Verify** — before saving, re-search every class and method call in your code via `lilbee_search`. Fix anything that doesn't match search results exactly.
