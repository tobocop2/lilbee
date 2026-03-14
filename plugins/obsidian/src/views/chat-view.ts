import { FuzzySuggestModal, ItemView, MarkdownRenderer, Menu, Notice, type TFile, WorkspaceLeaf } from "obsidian";
import type LilbeePlugin from "../main";
import { SSE_EVENT } from "../types";
import type { Message, Source } from "../types";
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

        // Model selector
        const modelSelect = toolbar.createEl("select", { cls: "lilbee-chat-model-select" });
        this.populateModelSelector(modelSelect);

        const addBtn = toolbar.createEl("button", {
            text: "+ Add file",
            cls: "lilbee-chat-add-file",
        });
        addBtn.addEventListener("click", (e) => this.openFilePicker(e));

        const clearBtn = toolbar.createEl("button", {
            text: "Clear chat",
            cls: "lilbee-chat-clear",
        });
        clearBtn.addEventListener("click", () => this.clearChat());

        // Messages list
        this.messagesEl = container.createDiv({ cls: "lilbee-chat-messages" });

        // Input area
        const inputArea = container.createDiv({ cls: "lilbee-chat-input" });
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

    private populateModelSelector(selectEl: HTMLElement): void {
        this.plugin.api.listModels().then((models) => {
            for (const name of models.chat.installed) {
                const option = selectEl.createEl("option", { text: name });
                (option as any).value = name;
                if (name === models.chat.active) {
                    (option as any).selected = true;
                }
            }
        }).catch(() => {
            selectEl.createEl("option", { text: "(offline)" });
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
                    new Notice(`lilbee: ${errMsg}`);
                }
            }
        } catch {
            if (spinner.parentElement) spinner.remove();
            textEl.style.display = "";
            new Notice("lilbee: chat error — cannot connect to server");
            textEl.textContent = "Error: could not connect to the lilbee server.";
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
