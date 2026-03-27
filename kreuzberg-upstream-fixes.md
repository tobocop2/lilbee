# Kreuzberg Upstream Fixes Needed

## ChunkingConfig stub missing fields

The `_internal_bindings.pyi` stub for `ChunkingConfig` is missing parameters that exist at runtime in kreuzberg 4.6.2:

- `chunker_type` — used to select markdown-aware chunking (`chunker_type="markdown"`)
- `prepend_heading_context` — prepends heading hierarchy path to chunks
- `sizing_type` — exists at runtime but not in stub
- `sizing_model` — exists at runtime but not in stub
- `sizing_cache_dir` — exists at runtime but not in stub

### Current stub (incomplete):

```python
class ChunkingConfig:
    max_chars: int
    max_overlap: int
    embedding: EmbeddingConfig | None
    preset: str | None
```

### Should be:

```python
class ChunkingConfig:
    max_chars: int
    max_overlap: int
    embedding: EmbeddingConfig | None
    preset: str | None
    chunker_type: str | None
    sizing_type: str | None
    sizing_model: str | None
    sizing_cache_dir: str | None
    prepend_heading_context: bool | None
```

### Impact

lilbee has to use `# type: ignore[call-arg]` on these kwargs in `src/lilbee/chunk.py`. Once the stub is fixed, those ignores can be removed.

### Location in kreuzberg

The stub file is likely auto-generated from the Rust bindings. Check wherever `_internal_bindings.pyi` is generated — the Rust `ChunkingConfig` struct has these fields but they're not being emitted to the Python stub.
