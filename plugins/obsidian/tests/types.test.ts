import { vi, describe, it, expect } from "vitest";
import { DEFAULT_SETTINGS, SSE_EVENT, JSON_HEADERS } from "../src/types";
import type {
    Excerpt,
    DocumentResult,
    AskResponse,
    Source,
    ModelInfo,
    ModelCatalog,
    ModelsResponse,
    StatusResponse,
    SyncDone,
    PullProgress,
    SSEEvent,
    Message,
    LilbeeSettings,
} from "../src/types";

describe("DEFAULT_SETTINGS", () => {
    it("has the correct serverUrl", () => {
        expect(DEFAULT_SETTINGS.serverUrl).toBe("http://127.0.0.1:7433");
    });

    it("has topK of 5", () => {
        expect(DEFAULT_SETTINGS.topK).toBe(5);
    });

    it("has syncMode set to manual", () => {
        expect(DEFAULT_SETTINGS.syncMode).toBe("manual");
    });

    it("has syncDebounceMs of 5000", () => {
        expect(DEFAULT_SETTINGS.syncDebounceMs).toBe(5000);
    });

    it("is a plain object with exactly the expected keys", () => {
        const keys = Object.keys(DEFAULT_SETTINGS).sort();
        expect(keys).toEqual(["binaryPath", "manageServer", "ollamaUrl", "serverUrl", "syncDebounceMs", "syncMode", "topK"].sort());
    });
});

// Interface shape tests — TypeScript compiles these so runtime values
// are only needed for the constant; the interface tests below verify
// that objects conforming to each interface can be assigned correctly.

describe("Excerpt interface", () => {
    it("accepts all fields including nulls", () => {
        const e: Excerpt = {
            content: "some text",
            page_start: 1,
            page_end: 2,
            line_start: 10,
            line_end: 20,
            relevance: 0.95,
        };
        expect(e.content).toBe("some text");
        expect(e.relevance).toBe(0.95);
    });

    it("accepts null for optional page/line fields", () => {
        const e: Excerpt = {
            content: "text",
            page_start: null,
            page_end: null,
            line_start: null,
            line_end: null,
            relevance: 0.5,
        };
        expect(e.page_start).toBeNull();
        expect(e.line_end).toBeNull();
    });
});

describe("DocumentResult interface", () => {
    it("holds a list of excerpts", () => {
        const excerpt: Excerpt = {
            content: "chunk",
            page_start: null,
            page_end: null,
            line_start: null,
            line_end: null,
            relevance: 0.8,
        };
        const doc: DocumentResult = {
            source: "file.md",
            content_type: "text/markdown",
            excerpts: [excerpt],
            best_relevance: 0.8,
        };
        expect(doc.excerpts).toHaveLength(1);
        expect(doc.source).toBe("file.md");
    });
});

describe("Source interface", () => {
    it("accepts required fields only", () => {
        const s: Source = {
            source: "doc.pdf",
            content_type: "application/pdf",
            distance: 0.12,
            chunk: "some text",
        };
        expect(s.page_start).toBeUndefined();
        expect(s.distance).toBe(0.12);
    });

    it("accepts all optional fields", () => {
        const s: Source = {
            source: "doc.pdf",
            content_type: "application/pdf",
            distance: 0.1,
            chunk: "text",
            page_start: 1,
            page_end: 2,
            line_start: 5,
            line_end: 10,
        };
        expect(s.page_end).toBe(2);
        expect(s.line_start).toBe(5);
    });
});

describe("AskResponse interface", () => {
    it("holds answer and sources", () => {
        const r: AskResponse = {
            answer: "42",
            sources: [],
        };
        expect(r.answer).toBe("42");
        expect(r.sources).toEqual([]);
    });
});

describe("ModelInfo interface", () => {
    it("captures all model metadata", () => {
        const m: ModelInfo = {
            name: "llama3",
            size_gb: 4.7,
            min_ram_gb: 8,
            description: "A good model",
            installed: true,
        };
        expect(m.installed).toBe(true);
    });
});

describe("ModelCatalog interface", () => {
    it("has active model and catalog list", () => {
        const c: ModelCatalog = {
            active: "llama3",
            catalog: [],
            installed: ["llama3"],
        };
        expect(c.active).toBe("llama3");
    });
});

describe("ModelsResponse interface", () => {
    it("has chat and vision catalogs", () => {
        const catalog: ModelCatalog = { active: "m", catalog: [], installed: [] };
        const r: ModelsResponse = { chat: catalog, vision: catalog };
        expect(r.chat).toBe(catalog);
        expect(r.vision).toBe(catalog);
    });
});

describe("StatusResponse interface", () => {
    it("holds config, sources, and total_chunks", () => {
        const s: StatusResponse = {
            config: { model: "llama3" },
            sources: [{ filename: "a.md", chunk_count: 3 }],
            total_chunks: 3,
        };
        expect(s.total_chunks).toBe(3);
        expect(s.sources[0].filename).toBe("a.md");
    });
});

describe("SyncDone interface", () => {
    it("holds sync result summary", () => {
        const d: SyncDone = {
            added: ["a.md"],
            updated: [],
            removed: ["b.md"],
            unchanged: 10,
            failed: [],
        };
        expect(d.added).toContain("a.md");
        expect(d.unchanged).toBe(10);
    });
});

describe("PullProgress interface", () => {
    it("tracks model pull progress", () => {
        const p: PullProgress = {
            model: "llama3",
            status: "downloading",
            completed: 512,
            total: 4096,
        };
        expect(p.completed).toBe(512);
    });
});

describe("SSEEvent interface", () => {
    it("holds event name and arbitrary data", () => {
        const e: SSEEvent = { event: "progress", data: { pct: 50 } };
        expect(e.event).toBe("progress");
    });
});

describe("Message interface", () => {
    it("accepts user role", () => {
        const m: Message = { role: "user", content: "hello" };
        expect(m.role).toBe("user");
    });

    it("accepts assistant role", () => {
        const m: Message = { role: "assistant", content: "hi" };
        expect(m.role).toBe("assistant");
    });

    it("accepts system role", () => {
        const m: Message = { role: "system", content: "you are helpful" };
        expect(m.role).toBe("system");
    });
});

describe("LilbeeSettings interface", () => {
    it("accepts manual syncMode", () => {
        const s: LilbeeSettings = {
            serverUrl: "http://localhost:7433",
            topK: 3,
            syncMode: "manual",
            syncDebounceMs: 2000,
        };
        expect(s.syncMode).toBe("manual");
    });

    it("accepts auto syncMode", () => {
        const s: LilbeeSettings = {
            serverUrl: "http://localhost:7433",
            topK: 3,
            syncMode: "auto",
            syncDebounceMs: 2000,
        };
        expect(s.syncMode).toBe("auto");
    });
});

describe("SSE_EVENT constants", () => {
    it("has all expected event types", () => {
        expect(SSE_EVENT.TOKEN).toBe("token");
        expect(SSE_EVENT.SOURCES).toBe("sources");
        expect(SSE_EVENT.DONE).toBe("done");
        expect(SSE_EVENT.ERROR).toBe("error");
        expect(SSE_EVENT.PROGRESS).toBe("progress");
        expect(SSE_EVENT.MESSAGE).toBe("message");
        expect(SSE_EVENT.FILE_START).toBe("file_start");
        expect(SSE_EVENT.EXTRACT).toBe("extract");
        expect(SSE_EVENT.EMBED).toBe("embed");
        expect(SSE_EVENT.FILE_DONE).toBe("file_done");
    });
});

describe("JSON_HEADERS constant", () => {
    it("has Content-Type application/json", () => {
        expect(JSON_HEADERS["Content-Type"]).toBe("application/json");
    });
});
