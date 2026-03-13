import { App, Modal } from "obsidian";
import type LilbeePlugin from "../main";
import type { DocumentResult } from "../types";
import { renderDocumentResult, renderSourceChip } from "./results";

const SEARCH_DEBOUNCE_MS = 300;

export class SearchModal extends Modal {
    private plugin: LilbeePlugin;
    private mode: "search" | "ask";
    private debounceTimer: ReturnType<typeof setTimeout> | null = null;
    private resultsContainer: HTMLElement | null = null;

    constructor(app: App, plugin: LilbeePlugin, mode: "search" | "ask" = "search") {
        super(app);
        this.plugin = plugin;
        this.mode = mode;
    }

    onOpen(): void {
        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass("lilbee-modal");

        const title = this.mode === "search" ? "Search knowledge base" : "Ask a question";
        contentEl.createEl("h2", { text: title });

        const input = contentEl.createEl("input", {
            type: "text",
            cls: "lilbee-search-input",
            placeholder: this.mode === "search" ? "Type to search..." : "Ask anything...",
        });

        this.resultsContainer = contentEl.createDiv({ cls: "lilbee-modal-results" });
        this.renderEmptyState("Enter a query to begin.");

        if (this.mode === "search") {
            input.addEventListener("input", () => {
                if (this.debounceTimer) clearTimeout(this.debounceTimer);
                this.debounceTimer = setTimeout(() => {
                    this.runSearch(input.value.trim());
                }, SEARCH_DEBOUNCE_MS);
            });
        } else {
            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && input.value.trim()) {
                    this.runAsk(input.value.trim());
                }
            });
        }

        // Focus input after open
        setTimeout(() => input.focus(), 0);
    }

    onClose(): void {
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        const { contentEl } = this;
        contentEl.empty();
    }

    private renderEmptyState(message: string): void {
        if (!this.resultsContainer) return;
        this.resultsContainer.empty();
        this.resultsContainer.createEl("p", {
            text: message,
            cls: "lilbee-empty-state",
        });
    }

    private renderLoading(): void {
        if (!this.resultsContainer) return;
        this.resultsContainer.empty();
        this.resultsContainer.createDiv({ cls: "lilbee-loading" });
    }

    private async runSearch(query: string): Promise<void> {
        if (!query) {
            this.renderEmptyState("Enter a query to begin.");
            return;
        }
        this.renderLoading();
        try {
            const results: DocumentResult[] = await this.plugin.api.search(
                query,
                this.plugin.settings.topK,
            );
            if (!this.resultsContainer) return;
            this.resultsContainer.empty();
            if (results.length === 0) {
                this.renderEmptyState("No results found.");
                return;
            }
            for (const result of results) {
                renderDocumentResult(this.resultsContainer, result, this.app);
            }
        } catch {
            this.renderEmptyState("Error: could not connect to lilbee server.");
        }
    }

    private async runAsk(question: string): Promise<void> {
        this.renderLoading();
        try {
            const response = await this.plugin.api.ask(question, this.plugin.settings.topK);
            if (!this.resultsContainer) return;
            this.resultsContainer.empty();

            this.resultsContainer.createEl("p", {
                text: response.answer,
                cls: "lilbee-ask-answer",
            });

            if (response.sources.length > 0) {
                const sourcesEl = this.resultsContainer.createDiv({ cls: "lilbee-ask-sources" });
                sourcesEl.createEl("span", { text: "Sources: " });
                for (const source of response.sources) {
                    renderSourceChip(sourcesEl, source);
                }
            }
        } catch {
            this.renderEmptyState("Error: could not connect to lilbee server.");
        }
    }
}
