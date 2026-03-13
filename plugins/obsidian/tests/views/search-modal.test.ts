import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { App } from "../__mocks__/obsidian";
import type LilbeePlugin from "../../src/main";
import type { DocumentResult, AskResponse } from "../../src/types";

// Mock the results module before importing SearchModal
vi.mock("../../src/views/results", () => ({
    renderDocumentResult: vi.fn(),
    renderSourceChip: vi.fn(),
}));

import { SearchModal } from "../../src/views/search-modal";
import { renderDocumentResult, renderSourceChip } from "../../src/views/results";

function makePlugin(): LilbeePlugin {
    return {
        api: {
            search: vi.fn(),
            ask: vi.fn(),
        },
        settings: {
            serverUrl: "http://localhost:7433",
            topK: 5,
            syncMode: "manual" as const,
            syncDebounceMs: 5000,
        },
    } as unknown as LilbeePlugin;
}

function makeResult(overrides: Partial<DocumentResult> = {}): DocumentResult {
    return {
        source: "doc.md",
        content_type: "text",
        excerpts: [],
        best_relevance: 0.9,
        ...overrides,
    };
}

function makeAskResponse(overrides: Partial<AskResponse> = {}): AskResponse {
    return {
        answer: "This is the answer.",
        sources: [],
        ...overrides,
    };
}

describe("SearchModal", () => {
    let app: App;
    let plugin: LilbeePlugin;

    beforeEach(() => {
        vi.useFakeTimers();
        app = new App();
        plugin = makePlugin();
        vi.mocked(renderDocumentResult).mockReset();
        vi.mocked(renderSourceChip).mockReset();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    describe("constructor", () => {
        it("defaults mode to 'search'", () => {
            const modal = new SearchModal(app, plugin);
            expect((modal as any).mode).toBe("search");
        });

        it("accepts 'ask' mode", () => {
            const modal = new SearchModal(app, plugin, "ask");
            expect((modal as any).mode).toBe("ask");
        });

        it("accepts explicit 'search' mode", () => {
            const modal = new SearchModal(app, plugin, "search");
            expect((modal as any).mode).toBe("search");
        });
    });

    describe("onOpen (search mode)", () => {
        let modal: SearchModal;

        beforeEach(() => {
            modal = new SearchModal(app, plugin, "search");
            modal.open();
        });

        it("adds lilbee-modal class to contentEl", () => {
            expect(modal.contentEl.classList.contains("lilbee-modal")).toBe(true);
        });

        it("renders h2 with 'Search knowledge base'", () => {
            const h2 = modal.contentEl.find("lilbee-modal")
                ? modal.contentEl.children.find(el => el.tagName === "H2")
                : null;
            // Search through direct children
            const headings = modal.contentEl.children.filter(c => c.tagName === "H2");
            expect(headings.length).toBe(1);
            expect(headings[0].textContent).toBe("Search knowledge base");
        });

        it("creates input with correct placeholder", () => {
            const inputs = modal.contentEl.children.filter(c => c.tagName === "INPUT");
            expect(inputs.length).toBe(1);
            expect(inputs[0].placeholder).toBe("Type to search...");
        });

        it("applies lilbee-search-input CSS class to input", () => {
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;
            expect(input.classList.contains("lilbee-search-input")).toBe(true);
        });

        it("creates a results container div", () => {
            const container = modal.contentEl.find("lilbee-modal-results");
            expect(container).not.toBeNull();
        });

        it("renders empty state on open", () => {
            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState).not.toBeNull();
            expect(emptyState!.textContent).toBe("Enter a query to begin.");
        });

        it("debounces search on input event", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "hello";
            input.trigger("input");

            // Not yet called before debounce fires
            expect(plugin.api.search).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(300);
            expect(plugin.api.search).toHaveBeenCalledWith("hello", 5);
        });

        it("resets debounce timer on rapid input events", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "h";
            input.trigger("input");
            await vi.advanceTimersByTimeAsync(100);

            input.value = "hello";
            input.trigger("input");
            await vi.advanceTimersByTimeAsync(300);

            // Only called once (second debounce)
            expect(plugin.api.search).toHaveBeenCalledTimes(1);
            expect(plugin.api.search).toHaveBeenCalledWith("hello", 5);
        });

        it("does not trigger search for empty input value", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "   "; // whitespace only
            input.trigger("input");
            await vi.advanceTimersByTimeAsync(300);

            // runSearch("") -> shows empty state, no API call
            expect(plugin.api.search).not.toHaveBeenCalled();
            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState!.textContent).toBe("Enter a query to begin.");
        });

        it("focuses input after setTimeout(0)", () => {
            // Advancing timers lets the focus setTimeout fire without errors
            vi.advanceTimersByTime(0);
            // No crash = pass (focus is a no-op on MockElement)
        });
    });

    describe("onOpen (ask mode)", () => {
        let modal: SearchModal;

        beforeEach(() => {
            modal = new SearchModal(app, plugin, "ask");
            modal.open();
        });

        it("renders h2 with 'Ask a question'", () => {
            const headings = modal.contentEl.children.filter(c => c.tagName === "H2");
            expect(headings[0].textContent).toBe("Ask a question");
        });

        it("creates input with 'Ask anything...' placeholder", () => {
            const inputs = modal.contentEl.children.filter(c => c.tagName === "INPUT");
            expect(inputs[0].placeholder).toBe("Ask anything...");
        });

        it("Enter key with non-empty value triggers runAsk", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(makeAskResponse());
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "What is lilbee?";
            input.trigger("keydown", { key: "Enter" });
            await vi.runAllTimersAsync();

            expect(plugin.api.ask).toHaveBeenCalledWith("What is lilbee?", 5);
        });

        it("Enter key with empty value does nothing", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(makeAskResponse());
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "";
            input.trigger("keydown", { key: "Enter" });
            await vi.runAllTimersAsync();

            expect(plugin.api.ask).not.toHaveBeenCalled();
        });

        it("non-Enter key does nothing", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(makeAskResponse());
            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;

            input.value = "What is lilbee?";
            input.trigger("keydown", { key: "a" });
            await vi.runAllTimersAsync();

            expect(plugin.api.ask).not.toHaveBeenCalled();
        });
    });

    describe("onClose", () => {
        it("empties contentEl", () => {
            const modal = new SearchModal(app, plugin);
            modal.open();
            // contentEl has children after open
            expect(modal.contentEl.children.length).toBeGreaterThan(0);

            modal.close();
            expect(modal.contentEl.children.length).toBe(0);
        });

        it("clears debounce timer on close", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);
            const modal = new SearchModal(app, plugin, "search");
            modal.open();

            const input = modal.contentEl.children.find(c => c.tagName === "INPUT")!;
            input.value = "query";
            input.trigger("input");

            // Close before debounce fires
            modal.close();

            await vi.advanceTimersByTimeAsync(300);
            // Search should not have been called because timer was cleared
            expect(plugin.api.search).not.toHaveBeenCalled();
        });

        it("handles close with no active debounce timer gracefully", () => {
            const modal = new SearchModal(app, plugin);
            modal.open();
            // No input event, so no timer set
            expect(() => modal.close()).not.toThrow();
        });
    });

    describe("runSearch", () => {
        let modal: SearchModal;

        beforeEach(() => {
            modal = new SearchModal(app, plugin, "search");
            modal.open();
        });

        it("shows loading then renders results", async () => {
            const result = makeResult();
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([result]);

            await (modal as any).runSearch("query");

            expect(renderDocumentResult).toHaveBeenCalledWith(
                (modal as any).resultsContainer,
                result,
                app,
            );
        });

        it("shows 'No results found.' when results array is empty", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);

            await (modal as any).runSearch("query");

            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState!.textContent).toBe("No results found.");
        });

        it("shows error message when API throws", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fail"));

            await (modal as any).runSearch("query");

            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState!.textContent).toBe("Error: could not connect to lilbee server.");
        });

        it("shows empty state for empty query string", async () => {
            await (modal as any).runSearch("");

            expect(plugin.api.search).not.toHaveBeenCalled();
            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState!.textContent).toBe("Enter a query to begin.");
        });

        it("renders multiple results", async () => {
            const results = [makeResult({ source: "a.md" }), makeResult({ source: "b.md" })];
            (plugin.api.search as ReturnType<typeof vi.fn>).mockResolvedValue(results);

            await (modal as any).runSearch("query");

            expect(renderDocumentResult).toHaveBeenCalledTimes(2);
        });

        it("guards against null resultsContainer after results arrive", async () => {
            (plugin.api.search as ReturnType<typeof vi.fn>).mockImplementation(async () => {
                // Null out the container while the API call is in-flight
                (modal as any).resultsContainer = null;
                return [makeResult()];
            });

            // Should not throw
            await expect((modal as any).runSearch("query")).resolves.toBeUndefined();
        });
    });

    describe("runAsk", () => {
        let modal: SearchModal;

        beforeEach(() => {
            modal = new SearchModal(app, plugin, "ask");
            modal.open();
        });

        it("shows loading then renders answer", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(
                makeAskResponse({ answer: "Forty-two." }),
            );

            await (modal as any).runAsk("What is the answer?");

            const answerEl = modal.contentEl.find("lilbee-ask-answer");
            expect(answerEl).not.toBeNull();
            expect(answerEl!.textContent).toBe("Forty-two.");
        });

        it("does not render sources section when sources array is empty", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(
                makeAskResponse({ sources: [] }),
            );

            await (modal as any).runAsk("question");

            const sourcesEl = modal.contentEl.find("lilbee-ask-sources");
            expect(sourcesEl).toBeNull();
        });

        it("renders sources section with chips when sources are present", async () => {
            const source = {
                source: "doc.md",
                content_type: "text",
                distance: 0.1,
                chunk: "chunk text",
                page_start: 1,
            };
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(
                makeAskResponse({ sources: [source, source] }),
            );

            await (modal as any).runAsk("question");

            expect(renderSourceChip).toHaveBeenCalledTimes(2);
            const sourcesEl = modal.contentEl.find("lilbee-ask-sources");
            expect(sourcesEl).not.toBeNull();
        });

        it("renders 'Sources: ' label before source chips", async () => {
            const source = {
                source: "doc.md",
                content_type: "text",
                distance: 0.1,
                chunk: "chunk",
            };
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockResolvedValue(
                makeAskResponse({ sources: [source] }),
            );

            await (modal as any).runAsk("question");

            const sourcesEl = modal.contentEl.find("lilbee-ask-sources")!;
            const label = sourcesEl.children.find(c => c.textContent === "Sources: ");
            expect(label).not.toBeUndefined();
        });

        it("shows error message when API throws", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fail"));

            await (modal as any).runAsk("question");

            const emptyState = modal.contentEl.find("lilbee-empty-state");
            expect(emptyState!.textContent).toBe("Error: could not connect to lilbee server.");
        });

        it("guards against null resultsContainer after answer arrives", async () => {
            (plugin.api.ask as ReturnType<typeof vi.fn>).mockImplementation(async () => {
                (modal as any).resultsContainer = null;
                return makeAskResponse();
            });

            await expect((modal as any).runAsk("question")).resolves.toBeUndefined();
        });
    });

    describe("renderEmptyState with null resultsContainer", () => {
        it("returns early without throwing when resultsContainer is null", () => {
            const modal = new SearchModal(app, plugin);
            modal.open();
            (modal as any).resultsContainer = null;
            expect(() => (modal as any).renderEmptyState("msg")).not.toThrow();
        });
    });

    describe("renderLoading with null resultsContainer", () => {
        it("returns early without throwing when resultsContainer is null", () => {
            const modal = new SearchModal(app, plugin);
            modal.open();
            (modal as any).resultsContainer = null;
            expect(() => (modal as any).renderLoading()).not.toThrow();
        });
    });
});
