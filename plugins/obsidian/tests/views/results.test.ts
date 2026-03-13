import { vi, describe, it, expect, beforeEach } from "vitest";
import { App, MockElement } from "../__mocks__/obsidian";
import { renderDocumentResult, renderSourceChip } from "../../src/views/results";
import type { DocumentResult, Source } from "../../src/types";

function makeContainer(): MockElement {
    return new MockElement("div");
}

function makeApp(): App {
    return new App();
}

function makeDocumentResult(overrides: Partial<DocumentResult> = {}): DocumentResult {
    return {
        source: "notes/foo.md",
        content_type: "markdown",
        excerpts: [],
        best_relevance: 0.75,
        ...overrides,
    };
}

function makeExcerpt(
    content: string,
    page_start: number | null = null,
    page_end: number | null = null,
    line_start: number | null = null,
    line_end: number | null = null,
) {
    return { content, page_start, page_end, line_start, line_end, relevance: 0.8 };
}

describe("renderDocumentResult — card structure", () => {
    it("creates a card div with the correct class", () => {
        const container = makeContainer();
        renderDocumentResult(container as unknown as HTMLElement, makeDocumentResult(), makeApp());
        const card = container.find("lilbee-document-card");
        expect(card).not.toBeNull();
    });

    it("creates a header div inside the card", () => {
        const container = makeContainer();
        renderDocumentResult(container as unknown as HTMLElement, makeDocumentResult(), makeApp());
        const header = container.find("lilbee-document-card-header");
        expect(header).not.toBeNull();
    });

    it("renders the source link with correct text and class", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ source: "vault/my-note.md" }),
            makeApp(),
        );
        const link = container.find("lilbee-document-source");
        expect(link).not.toBeNull();
        expect(link!.textContent).toBe("vault/my-note.md");
        expect(link!.tagName).toBe("A");
    });

    it("renders the content-type badge with correct text", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ content_type: "pdf" }),
            makeApp(),
        );
        const badge = container.find("lilbee-content-badge");
        expect(badge).not.toBeNull();
        expect(badge!.textContent).toBe("pdf");
    });

    it("renders a relevance bar container and bar", () => {
        const container = makeContainer();
        renderDocumentResult(container as unknown as HTMLElement, makeDocumentResult(), makeApp());
        expect(container.find("lilbee-relevance-bar-container")).not.toBeNull();
        expect(container.find("lilbee-relevance-bar")).not.toBeNull();
    });
});

describe("renderDocumentResult — relevance bar width", () => {
    it("sets bar width to rounded percentage of best_relevance", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ best_relevance: 0.756 }),
            makeApp(),
        );
        const bar = container.find("lilbee-relevance-bar")!;
        expect(bar.style["width"]).toBe("76%");
    });

    it("clamps best_relevance > 1 to 100%", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ best_relevance: 1.5 }),
            makeApp(),
        );
        const bar = container.find("lilbee-relevance-bar")!;
        expect(bar.style["width"]).toBe("100%");
    });

    it("clamps best_relevance < 0 to 0%", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ best_relevance: -0.3 }),
            makeApp(),
        );
        const bar = container.find("lilbee-relevance-bar")!;
        expect(bar.style["width"]).toBe("0%");
    });

    it("handles best_relevance of exactly 1 → 100%", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ best_relevance: 1 }),
            makeApp(),
        );
        const bar = container.find("lilbee-relevance-bar")!;
        expect(bar.style["width"]).toBe("100%");
    });

    it("handles best_relevance of exactly 0 → 0%", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ best_relevance: 0 }),
            makeApp(),
        );
        const bar = container.find("lilbee-relevance-bar")!;
        expect(bar.style["width"]).toBe("0%");
    });
});

describe("renderDocumentResult — source link click", () => {
    it("calls app.workspace.openLinkText with the source on click", () => {
        const container = makeContainer();
        const app = makeApp();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ source: "vault/my-note.md" }),
            app,
        );
        const link = container.find("lilbee-document-source")!;
        const mockEvent = { preventDefault: vi.fn() };
        link.trigger("click", mockEvent);
        expect(mockEvent.preventDefault).toHaveBeenCalled();
        expect(app.workspace.openLinkText).toHaveBeenCalledWith("vault/my-note.md", "");
    });
});

describe("renderDocumentResult — excerpts", () => {
    it("renders no excerpt elements when excerpts array is empty", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [] }),
            makeApp(),
        );
        expect(container.findAll("lilbee-excerpt")).toHaveLength(0);
    });

    it("renders up to 3 excerpts even when 5 are provided", () => {
        const container = makeContainer();
        const excerpts = [1, 2, 3, 4, 5].map((i) => makeExcerpt(`Excerpt ${i}`));
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts }),
            makeApp(),
        );
        expect(container.findAll("lilbee-excerpt")).toHaveLength(3);
    });

    it("renders exactly 3 excerpts when exactly 3 are provided", () => {
        const container = makeContainer();
        const excerpts = [1, 2, 3].map((i) => makeExcerpt(`Excerpt ${i}`));
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts }),
            makeApp(),
        );
        expect(container.findAll("lilbee-excerpt")).toHaveLength(3);
    });

    it("renders fewer than 3 excerpts when fewer are provided", () => {
        const container = makeContainer();
        const excerpts = [makeExcerpt("Only one")];
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts }),
            makeApp(),
        );
        expect(container.findAll("lilbee-excerpt")).toHaveLength(1);
    });

    it("renders excerpt content as a <p> element", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("Hello world")] }),
            makeApp(),
        );
        const excerptEl = container.find("lilbee-excerpt")!;
        const p = excerptEl.children.find((c) => c.tagName === "P");
        expect(p).toBeDefined();
        expect(p!.textContent).toBe("Hello world");
    });

    it("does NOT render a location span when all location fields are null", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("No location")] }),
            makeApp(),
        );
        const excerptEl = container.find("lilbee-excerpt")!;
        const loc = excerptEl.find("lilbee-location");
        expect(loc).toBeNull();
    });

    it("renders a location span when page_start is set", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("Has page", 3)] }),
            makeApp(),
        );
        const excerptEl = container.find("lilbee-excerpt")!;
        const loc = excerptEl.find("lilbee-location");
        expect(loc).not.toBeNull();
        expect(loc!.textContent).toBe("p. 3");
    });

    it("renders a location span when line_start is set", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({
                excerpts: [makeExcerpt("Has line", null, null, 10)],
            }),
            makeApp(),
        );
        const excerptEl = container.find("lilbee-excerpt")!;
        const loc = excerptEl.find("lilbee-location");
        expect(loc).not.toBeNull();
        expect(loc!.textContent).toBe("line 10");
    });
});

describe("renderDocumentResult — truncate", () => {
    const shortText = "Short text";
    const longText = "x".repeat(201);

    it("does not truncate text at or below 200 chars", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt(shortText)] }),
            makeApp(),
        );
        const p = container.find("lilbee-excerpt")!.children.find((c) => c.tagName === "P")!;
        expect(p.textContent).toBe(shortText);
    });

    it("does not truncate text exactly 200 chars long", () => {
        const exactly200 = "a".repeat(200);
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt(exactly200)] }),
            makeApp(),
        );
        const p = container.find("lilbee-excerpt")!.children.find((c) => c.tagName === "P")!;
        expect(p.textContent).toBe(exactly200);
    });

    it("truncates text over 200 chars and appends '...'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt(longText)] }),
            makeApp(),
        );
        const p = container.find("lilbee-excerpt")!.children.find((c) => c.tagName === "P")!;
        expect(p.textContent).toBe("x".repeat(200) + "...");
    });
});

describe("formatLocation — page branches", () => {
    it("page_start only → 'p. X'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", 5, null)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("p. 5");
    });

    it("page_start === page_end → 'p. X' (same page)", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", 7, 7)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("p. 7");
    });

    it("page_start !== page_end → 'pp. X–Y'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", 2, 4)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("pp. 2\u20134");
    });
});

describe("formatLocation — line branches", () => {
    it("line_start only → 'line X'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", null, null, 10, null)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("line 10");
    });

    it("line_start === line_end → 'line X'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", null, null, 10, 10)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("line 10");
    });

    it("line_start !== line_end → 'lines X–Y'", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", null, null, 10, 20)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")!.textContent).toBe("lines 10\u201320");
    });

    it("all null → no location element rendered", () => {
        const container = makeContainer();
        renderDocumentResult(
            container as unknown as HTMLElement,
            makeDocumentResult({ excerpts: [makeExcerpt("text", null, null, null, null)] }),
            makeApp(),
        );
        expect(container.find("lilbee-location")).toBeNull();
    });
});

describe("renderSourceChip — basic", () => {
    it("creates a span with class lilbee-source-chip", () => {
        const container = makeContainer();
        const source: Source = {
            source: "docs/readme.md",
            content_type: "markdown",
            distance: 0.1,
            chunk: "some text",
            page_start: null,
            page_end: null,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        const chip = container.find("lilbee-source-chip");
        expect(chip).not.toBeNull();
        expect(chip!.tagName).toBe("SPAN");
    });

    it("sets chip text to source name when no location fields", () => {
        const container = makeContainer();
        const source: Source = {
            source: "docs/readme.md",
            content_type: "markdown",
            distance: 0.1,
            chunk: "some text",
            page_start: null,
            page_end: null,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("docs/readme.md");
    });

    it("appends page location when page_start is provided (only)", () => {
        const container = makeContainer();
        const source: Source = {
            source: "book.pdf",
            content_type: "pdf",
            distance: 0.2,
            chunk: "...",
            page_start: 42,
            page_end: null,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("book.pdf (p. 42)");
    });

    it("appends page range when page_start and different page_end provided", () => {
        const container = makeContainer();
        const source: Source = {
            source: "book.pdf",
            content_type: "pdf",
            distance: 0.2,
            chunk: "...",
            page_start: 10,
            page_end: 12,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("book.pdf (pp. 10\u201312)");
    });

    it("appends 'p. X' when page_start === page_end", () => {
        const container = makeContainer();
        const source: Source = {
            source: "book.pdf",
            content_type: "pdf",
            distance: 0.2,
            chunk: "...",
            page_start: 5,
            page_end: 5,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("book.pdf (p. 5)");
    });

    it("appends line location when line_start is provided (only)", () => {
        const container = makeContainer();
        const source: Source = {
            source: "main.py",
            content_type: "code",
            distance: 0.05,
            chunk: "def foo():",
            page_start: null,
            page_end: null,
            line_start: 99,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("main.py (line 99)");
    });

    it("appends line range when line_start and different line_end provided", () => {
        const container = makeContainer();
        const source: Source = {
            source: "main.py",
            content_type: "code",
            distance: 0.05,
            chunk: "def foo():",
            page_start: null,
            page_end: null,
            line_start: 10,
            line_end: 20,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("main.py (lines 10\u201320)");
    });

    it("appends 'line X' when line_start === line_end", () => {
        const container = makeContainer();
        const source: Source = {
            source: "main.py",
            content_type: "code",
            distance: 0.05,
            chunk: "def foo():",
            page_start: null,
            page_end: null,
            line_start: 15,
            line_end: 15,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("main.py (line 15)");
    });

    it("shows no location suffix when all location fields are null", () => {
        const container = makeContainer();
        const source: Source = {
            source: "notes/plain.md",
            content_type: "markdown",
            distance: 0.3,
            chunk: "body text",
            page_start: null,
            page_end: null,
            line_start: null,
            line_end: null,
        };
        renderSourceChip(container as unknown as HTMLElement, source);
        expect(container.find("lilbee-source-chip")!.textContent).toBe("notes/plain.md");
    });
});
