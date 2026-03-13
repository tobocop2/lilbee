import { App } from "obsidian";
import type { DocumentResult, Source } from "../types";

const MAX_EXCERPT_CHARS = 200;
const MAX_EXCERPTS = 3;

function formatLocation(excerpt: { page_start: number | null; page_end: number | null; line_start: number | null; line_end: number | null }): string | null {
    if (excerpt.page_start !== null) {
        return excerpt.page_end !== null && excerpt.page_end !== excerpt.page_start
            ? `pp. ${excerpt.page_start}–${excerpt.page_end}`
            : `p. ${excerpt.page_start}`;
    }
    if (excerpt.line_start !== null) {
        return excerpt.line_end !== null && excerpt.line_end !== excerpt.line_start
            ? `lines ${excerpt.line_start}–${excerpt.line_end}`
            : `line ${excerpt.line_start}`;
    }
    return null;
}

function truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + "...";
}

export function renderDocumentResult(container: HTMLElement, result: DocumentResult, app: App): void {
    const card = container.createDiv({ cls: "lilbee-document-card" });

    // Header: filename + content type badge
    const header = card.createDiv({ cls: "lilbee-document-card-header" });
    const link = header.createEl("a", {
        text: result.source,
        cls: "lilbee-document-source",
    });
    link.addEventListener("click", (e) => {
        e.preventDefault();
        app.workspace.openLinkText(result.source, "");
    });

    header.createEl("span", {
        text: result.content_type,
        cls: "lilbee-content-badge",
    });

    // Relevance bar
    const barContainer = card.createDiv({ cls: "lilbee-relevance-bar-container" });
    const bar = barContainer.createDiv({ cls: "lilbee-relevance-bar" });
    const pct = Math.round(Math.max(0, Math.min(1, result.best_relevance)) * 100);
    bar.style.width = `${pct}%`;

    // Excerpts (up to MAX_EXCERPTS)
    const excerpts = result.excerpts.slice(0, MAX_EXCERPTS);
    for (const excerpt of excerpts) {
        const excerptEl = card.createDiv({ cls: "lilbee-excerpt" });
        excerptEl.createEl("p", { text: truncate(excerpt.content, MAX_EXCERPT_CHARS) });
        const loc = formatLocation(excerpt);
        if (loc) {
            excerptEl.createEl("span", { text: loc, cls: "lilbee-location" });
        }
    }
}

export function renderSourceChip(container: HTMLElement, source: Source): void {
    const chip = container.createEl("span", { cls: "lilbee-source-chip" });

    let label = source.source;
    const loc = formatLocation(source);
    if (loc) {
        label += ` (${loc})`;
    }
    chip.setText(label);
}
