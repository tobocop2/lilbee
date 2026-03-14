import { FuzzySuggestModal, ItemView, MarkdownRenderer, Menu, Notice, setIcon, type TFile, WorkspaceLeaf } from "obsidian";
import type LilbeePlugin from "../main";
import { SSE_EVENT } from "../types";
import type { Message, ModelCatalog, Source, SSEEvent } from "../types";
import { renderSourceChip } from "./results";
import { buildModelOptions, SEPARATOR_KEY } from "../settings";

interface OpenDialogResult {
    canceled: boolean;
    filePaths: string[];
}

/** Thin wrapper around Electron's dialog — exported for test stubbing. */
export const electronDialog = {
    /* v8 ignore start -- requires Electron runtime */
    showOpenDialog(opts: Record<string, unknown>): Promise<OpenDialogResult> {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const electron = require("electron") as {
            remote: { dialog: { showOpenDialog(o: Record<string, unknown>): Promise<OpenDialogResult> } };
        };
        return electron.remote.dialog.showOpenDialog(opts);
    },
    /* v8 ignore stop */
};

export const VIEW_TYPE_CHAT = "lilbee-chat";

interface ProgressInfo {
    label: string;
    current: number;
    total: number;
}

const PROGRESS_EXTRACTORS: Record<string, (data: any) => ProgressInfo> = {
    [SSE_EVENT.FILE_START]: (d) => ({
        label: `Indexing ${d.current_file}/${d.total_files} — ${d.file}`,
        current: d.current_file,
        total: d.total_files,
    }),
    [SSE_EVENT.EXTRACT]: (d) => ({
        label: `Extracting ${d.file} (page ${d.page}/${d.total_pages})`,
        current: d.page,
        total: d.total_pages,
    }),
    [SSE_EVENT.EMBED]: (d) => ({
        label: `Embedding ${d.file} (${d.chunk}/${d.total_chunks} chunks)`,
        current: d.chunk,
        total: d.total_chunks,
    }),
    [SSE_EVENT.PROGRESS]: (d) => ({
        label: `Indexing ${d.current}/${d.total} — ${d.file}`,
        current: d.current,
        total: d.total,
    }),
    [SSE_EVENT.PULL]: (d) => ({
        label: `Pulling ${d.model} — ${Math.round((d.current / d.total) * 100)}%`,
        current: d.current,
        total: d.total,
    }),
};

function extractString(data: unknown, field: string): string {
    if (typeof data === "object" && data !== null && field in data) {
        return String((data as Record<string, unknown>)[field]);
    }
    return String(data);
}

export class ChatView extends ItemView {
    private plugin: LilbeePlugin;
    private history: Message[] = [];
    private messagesEl: HTMLElement | null = null;
    private sendBtn: HTMLButtonElement | null = null;
    private sending = false;
    private connectionDot: HTMLElement | null = null;
    private progressBanner: HTMLElement | null = null;
    private progressLabel: HTMLElement | null = null;
    private progressBar: HTMLElement | null = null;

    constructor(leaf: WorkspaceLeaf, plugin: LilbeePlugin) {
        super(leaf);
        this.plugin = plugin;
    }

    getViewType(): string {
        return VIEW_TYPE_CHAT;
    }

    getDisplayText(): string {
        return "lilbee Chat";
    }

    getIcon(): string {
        return "message-circle";
    }

    async onOpen(): Promise<void> {
        const container = this.containerEl.children[1] as HTMLElement;
        container.empty();
        container.addClass("lilbee-chat-container");

        this.createToolbar(container);
        this.createProgressBanner(container);
        this.messagesEl = container.createDiv({ cls: "lilbee-chat-messages" });
        this.createInputArea(container);

        this.plugin.onProgress = (event) => this.handleProgress(event);
    }

    async onClose(): Promise<void> {
        if (this.plugin.onProgress) {
            this.plugin.onProgress = null;
        }
    }

    private createToolbar(container: HTMLElement): void {
        const toolbar = container.createDiv({ cls: "lilbee-chat-toolbar" });

        this.connectionDot = toolbar.createDiv({ cls: "lilbee-connection-dot" });
        this.pingHealth();

        const modelSelect = toolbar.createEl("select", {
            cls: "lilbee-chat-model-select",
        }) as HTMLSelectElement;
        this.populateModelSelector(modelSelect);

        const clearBtn = toolbar.createEl("button", {
            text: "Clear chat",
            cls: "lilbee-chat-clear",
        });
        clearBtn.addEventListener("click", () => this.clearChat());
    }

    private createProgressBanner(container: HTMLElement): void {
        this.progressBanner = container.createDiv({ cls: "lilbee-progress-banner" });
        this.progressBanner.style.display = "none";
        this.progressLabel = this.progressBanner.createDiv({ cls: "lilbee-progress-label" });
        const barContainer = this.progressBanner.createDiv({ cls: "lilbee-progress-bar-container" });
        this.progressBar = barContainer.createDiv({ cls: "lilbee-progress-bar" });
    }

    private createInputArea(container: HTMLElement): void {
        const inputArea = container.createDiv({ cls: "lilbee-chat-input" });

        const addBtn = inputArea.createEl("button", { cls: "lilbee-chat-add-file" });
        addBtn.setAttribute("aria-label", "Add file");
        setIcon(addBtn, "paperclip");
        addBtn.addEventListener("click", (e) => this.openFilePicker(e));

        const textarea = inputArea.createEl("textarea", {
            placeholder: "Ask something...",
            cls: "lilbee-chat-textarea",
        });
        this.sendBtn = inputArea.createEl("button", {
            text: "Send",
            cls: "lilbee-chat-send",
        }) as HTMLButtonElement;

        const handleSend = (): void => {
            const text = textarea.value.trim();
            if (!text) return;
            textarea.value = "";
            void this.sendMessage(text);
        };

        this.sendBtn.addEventListener("click", handleSend);
        textarea.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });
    }

    private pingHealth(): void {
        this.plugin.api.health().then(() => {
            this.setConnectionStatus(true);
        }).catch(() => {
            this.setConnectionStatus(false);
        });
    }

    private setConnectionStatus(connected: boolean): void {
        if (!this.connectionDot) return;
        this.connectionDot.removeClass("connected", "disconnected");
        this.connectionDot.addClass(connected ? "connected" : "disconnected");
    }

    private chatCatalog: ModelCatalog | null = null;

    private populateModelSelector(selectEl: HTMLSelectElement): void {
        this.plugin.api.listModels().then((models) => {
            this.chatCatalog = models.chat;
            const options = buildModelOptions(models.chat, "chat");
            for (const [value, label] of Object.entries(options)) {
                const option = selectEl.createEl("option", { text: label });
                (option as HTMLOptionElement).value = value;
                if (value === SEPARATOR_KEY) {
                    (option as HTMLOptionElement).disabled = true;
                }
                if (value === models.chat.active) {
                    (option as HTMLOptionElement).selected = true;
                }
            }
            this.setConnectionStatus(true);
        }).catch(() => {
            selectEl.createEl("option", { text: "(offline)" });
            this.setConnectionStatus(false);
        });

        selectEl.addEventListener("change", () => {
            if (!selectEl.value || selectEl.value === SEPARATOR_KEY) return;
            const uninstalled = this.chatCatalog?.catalog.find(
                (m) => m.name === selectEl.value && !m.installed,
            );
            if (uninstalled) {
                this.autoPullAndSetChat(uninstalled, selectEl);
                return;
            }
            this.plugin.api.setChatModel(selectEl.value).then(() => {
                this.plugin.activeModel = selectEl.value;
                this.plugin.fetchActiveModel();
            }).catch(() => {
                new Notice("lilbee: failed to switch model");
            });
        });
    }

    private autoPullAndSetChat(
        model: { name: string },
        selectEl: HTMLSelectElement,
    ): void {
        new Notice(`Pulling ${model.name}...`);
        (async () => {
            try {
                for await (const event of this.plugin.api.pullModel(model.name)) {
                    if (event.event === SSE_EVENT.PROGRESS) {
                        const data = event.data as { completed: number; total: number };
                        if (data.total > 0) {
                            const pct = Math.round((data.completed / data.total) * 100);
                            this.showProgress(
                                `Pulling ${model.name} — ${pct}%`,
                                data.completed,
                                data.total,
                            );
                        }
                    }
                }
                this.hideProgress();
                await this.plugin.api.setChatModel(model.name);
                this.plugin.activeModel = model.name;
                this.plugin.fetchActiveModel();
                new Notice(`Model ${model.name} pulled and activated`);
                this.refreshModelSelector(selectEl);
            } catch {
                new Notice(`Failed to pull ${model.name}`);
                this.hideProgress();
            }
        })();
    }

    private refreshModelSelector(selectEl: HTMLSelectElement): void {
        selectEl.empty();
        this.populateModelSelector(selectEl);
    }

    private clearChat(): void {
        this.history = [];
        if (this.messagesEl) this.messagesEl.empty();
    }

    private async sendMessage(text: string): Promise<void> {
        if (!this.messagesEl || this.sending) return;
        this.sending = true;
        if (this.sendBtn) this.sendBtn.disabled = true;

        const userBubble = this.messagesEl.createDiv({ cls: "lilbee-chat-message user" });
        userBubble.createEl("p", { text });
        this.history.push({ role: "user", content: text });

        const assistantBubble = this.messagesEl.createDiv({ cls: "lilbee-chat-message assistant" });
        const spinner = assistantBubble.createDiv({ cls: "lilbee-loading" });
        spinner.textContent = "Thinking...";
        const textEl = assistantBubble.createDiv({ cls: "lilbee-chat-content" });
        textEl.style.display = "none";
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;

        const state = { fullContent: "", sources: [] as Source[], renderPending: false };

        const revealContent = (): void => {
            if (spinner.parentElement) spinner.remove();
            textEl.style.display = "";
        };

        const scrollToBottom = (): void => {
            if (this.messagesEl) this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        };

        const scheduleRender = (): void => {
            if (state.renderPending) return;
            state.renderPending = true;
            requestAnimationFrame(() => {
                state.renderPending = false;
                void this.renderMarkdown(textEl, state.fullContent).then(scrollToBottom);
            });
        };

        try {
            for await (const event of this.plugin.api.chatStream(
                text,
                this.history.slice(0, -1),
                this.plugin.settings.topK,
            )) {
                this.setConnectionStatus(true);
                this.handleStreamEvent(event, textEl, assistantBubble, state, revealContent, scheduleRender);
            }
        } catch {
            revealContent();
            this.setConnectionStatus(false);
            textEl.textContent = "Server unavailable — retries exhausted. Is lilbee running?";
            textEl.addClass("lilbee-chat-error");
        } finally {
            this.sending = false;
            if (this.sendBtn) this.sendBtn.disabled = false;
        }
    }

    private handleStreamEvent(
        event: SSEEvent,
        textEl: HTMLElement,
        assistantBubble: HTMLElement,
        state: { fullContent: string; sources: Source[] },
        revealContent: () => void,
        scheduleRender: () => void,
    ): void {
        switch (event.event) {
            case SSE_EVENT.TOKEN: {
                revealContent();
                state.fullContent += extractString(event.data, "token");
                scheduleRender();
                break;
            }
            case SSE_EVENT.SOURCES:
                state.sources.push(...(event.data as Source[]));
                break;
            case SSE_EVENT.DONE: {
                revealContent();
                void this.renderMarkdown(textEl, state.fullContent);
                if (state.sources.length > 0) this.renderSources(assistantBubble, state.sources);
                this.history.push({ role: "assistant", content: state.fullContent });
                break;
            }
            case SSE_EVENT.ERROR: {
                const errMsg = extractString(event.data, "message");
                revealContent();
                textEl.textContent = errMsg;
                textEl.addClass("lilbee-chat-error");
                new Notice(`lilbee: ${errMsg}`);
                break;
            }
        }
    }

    private async renderMarkdown(el: HTMLElement, markdown: string): Promise<void> {
        el.empty();
        await MarkdownRenderer.render(this.app, markdown, el, "", this.plugin);
        el.addClass("markdown-rendered");
    }

    private openFilePicker(event: MouseEvent): void {
        const menu = new Menu();
        menu.addItem((item) => {
            item.setTitle("From vault")
                .setIcon("vault")
                .onClick(() => {
                    new VaultFilePickerModal(this.app, this.plugin).open();
                });
        });
        menu.addItem((item) => {
            item.setTitle("Files from disk")
                .setIcon("file-plus")
                .onClick(() => this.openNativeFilePicker(false));
        });
        menu.addItem((item) => {
            item.setTitle("Folder from disk")
                .setIcon("folder-plus")
                .onClick(() => this.openNativeFilePicker(true));
        });
        menu.showAtMouseEvent(event);
    }

    private openNativeFilePicker(directory: boolean): void {
        const properties = directory
            ? ["openDirectory"]
            : ["openFile", "multiSelections"];
        electronDialog.showOpenDialog({ properties }).then((result) => {
            if (result.canceled || result.filePaths.length === 0) return;
            void this.plugin.addExternalFiles(result.filePaths);
        }).catch(() => {
            new Notice("lilbee: could not open file picker");
        });
    }

    handleProgress(event: SSEEvent): void {
        if (event.event === SSE_EVENT.DONE) {
            this.hideProgress();
            return;
        }

        const extractor = PROGRESS_EXTRACTORS[event.event];
        if (!extractor) return;

        const info = extractor(event.data);
        this.showProgress(info.label, info.current, info.total);
    }

    showProgress(label: string, current: number, total: number): void {
        if (!this.progressBanner || !this.progressLabel || !this.progressBar) return;
        this.progressBanner.style.display = "";
        this.progressLabel.textContent = label;
        this.progressBar.style.width = `${Math.round((current / total) * 100)}%`;
    }

    hideProgress(): void {
        if (!this.progressBanner || !this.progressBar) return;
        this.progressBanner.style.display = "none";
        this.progressBar.style.width = "0%";
    }

    private renderSources(container: HTMLElement, sources: Source[]): void {
        const sourcesEl = container.createDiv({ cls: "lilbee-chat-sources" });
        const details = sourcesEl.createEl("details");
        details.createEl("summary", { text: "Sources" });
        const chipsEl = details.createDiv({ cls: "lilbee-chat-source-chips" });
        for (const source of sources) {
            renderSourceChip(chipsEl, source);
        }
    }
}

export class VaultFilePickerModal extends FuzzySuggestModal<TFile> {
    private plugin: LilbeePlugin;

    constructor(app: import("obsidian").App, plugin: LilbeePlugin) {
        super(app);
        this.plugin = plugin;
        this.setPlaceholder("Pick a vault file to add to lilbee...");
    }

    getItems(): TFile[] {
        return this.app.vault.getFiles();
    }

    getItemText(item: TFile): string {
        return item.path;
    }

    onChooseItem(item: TFile): void {
        void this.plugin.addToLilbee(item);
    }
}
