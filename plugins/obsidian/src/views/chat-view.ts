import { FuzzySuggestModal, ItemView, MarkdownRenderer, Menu, Notice, setIcon, type TFile, WorkspaceLeaf } from "obsidian";
import type LilbeePlugin from "../main";
import { SSE_EVENT } from "../types";
import type { Message, Source, SSEEvent } from "../types";
import { renderSourceChip } from "./results";

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

        const toolbar = container.createDiv({ cls: "lilbee-chat-toolbar" });

        // Connection status dot
        this.connectionDot = toolbar.createDiv({ cls: "lilbee-connection-dot" });
        this.pingHealth();

        // Model selector
        const modelSelect = toolbar.createEl("select", { cls: "lilbee-chat-model-select" });
        this.populateModelSelector(modelSelect);

        const clearBtn = toolbar.createEl("button", {
            text: "Clear chat",
            cls: "lilbee-chat-clear",
        });
        clearBtn.addEventListener("click", () => this.clearChat());

        // Progress banner (hidden by default)
        this.progressBanner = container.createDiv({ cls: "lilbee-progress-banner" });
        this.progressBanner.style.display = "none";
        this.progressLabel = this.progressBanner.createDiv({ cls: "lilbee-progress-label" });
        const barContainer = this.progressBanner.createDiv({ cls: "lilbee-progress-bar-container" });
        this.progressBar = barContainer.createDiv({ cls: "lilbee-progress-bar" });

        // Messages list
        this.messagesEl = container.createDiv({ cls: "lilbee-chat-messages" });

        // Input area
        const inputArea = container.createDiv({ cls: "lilbee-chat-input" });

        // Paperclip add-file button
        const addBtn = inputArea.createEl("button", {
            cls: "lilbee-chat-add-file",
        });
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

        // Register for progress events from the plugin
        this.plugin.onProgress = (event) => this.handleProgress(event);
    }

    async onClose(): Promise<void> {
        if (this.plugin.onProgress) {
            this.plugin.onProgress = null;
        }
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

    private populateModelSelector(selectEl: HTMLElement): void {
        this.plugin.api.listModels().then((models) => {
            for (const name of models.chat.installed) {
                const option = selectEl.createEl("option", { text: name });
                (option as any).value = name;
                if (name === models.chat.active) {
                    (option as any).selected = true;
                }
            }
            this.setConnectionStatus(true);
        }).catch(() => {
            selectEl.createEl("option", { text: "(offline)" });
            this.setConnectionStatus(false);
        });

        selectEl.addEventListener("change", () => {
            const value = (selectEl as any).value;
            if (value) {
                this.plugin.api.setChatModel(value).then(() => {
                    this.plugin.activeModel = value;
                    this.plugin.fetchActiveModel();
                }).catch(() => {
                    new Notice("lilbee: failed to switch model");
                });
            }
        });
    }

    private clearChat(): void {
        this.history = [];
        if (this.messagesEl) this.messagesEl.empty();
    }

    private async sendMessage(text: string): Promise<void> {
        if (!this.messagesEl || this.sending) return;
        this.sending = true;
        if (this.sendBtn) this.sendBtn.disabled = true;

        // Render user bubble
        const userBubble = this.messagesEl.createDiv({
            cls: "lilbee-chat-message user",
        });
        userBubble.createEl("p", { text });

        // Push to history
        this.history.push({ role: "user", content: text });

        // Render assistant bubble with loading spinner
        const assistantBubble = this.messagesEl.createDiv({
            cls: "lilbee-chat-message assistant",
        });
        const spinner = assistantBubble.createDiv({ cls: "lilbee-loading" });
        spinner.textContent = "Thinking...";
        const textEl = assistantBubble.createDiv({ cls: "lilbee-chat-content" });
        textEl.style.display = "none";
        if (this.messagesEl) this.messagesEl.scrollTop = this.messagesEl.scrollHeight;

        let fullContent = "";
        const sources: Source[] = [];
        let renderPending = false;

        const scrollToBottom = (): void => {
            if (this.messagesEl) {
                this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
            }
        };

        const scheduleRender = (): void => {
            if (renderPending) return;
            renderPending = true;
            requestAnimationFrame(() => {
                renderPending = false;
                void this.renderMarkdown(textEl, fullContent).then(scrollToBottom);
            });
        };

        try {
            for await (const event of this.plugin.api.chatStream(
                text,
                this.history.slice(0, -1),
                this.plugin.settings.topK,
            )) {
                this.setConnectionStatus(true);
                if (event.event === SSE_EVENT.TOKEN) {
                    if (spinner.parentElement) spinner.remove();
                    textEl.style.display = "";
                    const raw = event.data;
                    const token = typeof raw === "object" && raw !== null && "token" in raw
                        ? String((raw as Record<string, unknown>).token)
                        : String(raw);
                    fullContent += token;
                    scheduleRender();
                } else if (event.event === SSE_EVENT.SOURCES) {
                    const data = event.data as Source[];
                    sources.push(...data);
                } else if (event.event === SSE_EVENT.DONE) {
                    if (spinner.parentElement) spinner.remove();
                    textEl.style.display = "";
                    await this.renderMarkdown(textEl, fullContent);
                    if (sources.length > 0) {
                        this.renderSources(assistantBubble, sources);
                    }
                    this.history.push({ role: "assistant", content: fullContent });
                } else if (event.event === SSE_EVENT.ERROR) {
                    const errData = event.data;
                    const errMsg = typeof errData === "object" && errData !== null && "message" in errData
                        ? String((errData as Record<string, unknown>).message)
                        : String(errData);
                    if (spinner.parentElement) spinner.remove();
                    textEl.style.display = "";
                    textEl.textContent = errMsg;
                    textEl.addClass("lilbee-chat-error");
                    new Notice(`lilbee: ${errMsg}`);
                }
            }
        } catch {
            if (spinner.parentElement) spinner.remove();
            textEl.style.display = "";
            this.setConnectionStatus(false);
            textEl.textContent = "Server unavailable — retries exhausted. Is lilbee running?";
            textEl.addClass("lilbee-chat-error");
        } finally {
            this.sending = false;
            if (this.sendBtn) this.sendBtn.disabled = false;
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
        if (!this.progressBanner || !this.progressLabel || !this.progressBar) return;

        switch (event.event) {
            case SSE_EVENT.FILE_START: {
                const data = event.data as { file: string; current_file: number; total_files: number };
                this.progressBanner.style.display = "";
                this.progressLabel.textContent = `Indexing ${data.current_file}/${data.total_files} — ${data.file}`;
                const pct = Math.round((data.current_file / data.total_files) * 100);
                this.progressBar.style.width = `${pct}%`;
                break;
            }
            case SSE_EVENT.EXTRACT: {
                const data = event.data as { file: string; page: number; total_pages: number };
                this.progressBanner.style.display = "";
                this.progressLabel.textContent = `Extracting ${data.file} (page ${data.page}/${data.total_pages})`;
                const pct = Math.round((data.page / data.total_pages) * 100);
                this.progressBar.style.width = `${pct}%`;
                break;
            }
            case SSE_EVENT.EMBED: {
                const data = event.data as { file: string; chunk: number; total_chunks: number };
                this.progressBanner.style.display = "";
                this.progressLabel.textContent = `Embedding ${data.file} (${data.chunk}/${data.total_chunks} chunks)`;
                const pct = Math.round((data.chunk / data.total_chunks) * 100);
                this.progressBar.style.width = `${pct}%`;
                break;
            }
            case SSE_EVENT.PROGRESS: {
                const data = event.data as { file: string; current: number; total: number };
                this.progressBanner.style.display = "";
                this.progressLabel.textContent = `Indexing ${data.current}/${data.total} — ${data.file}`;
                const pct = Math.round((data.current / data.total) * 100);
                this.progressBar.style.width = `${pct}%`;
                break;
            }
            case SSE_EVENT.DONE:
                this.progressBanner.style.display = "none";
                this.progressBar.style.width = "0%";
                break;
        }
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
