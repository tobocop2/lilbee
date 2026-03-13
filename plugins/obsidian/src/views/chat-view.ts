import { ItemView, Notice, WorkspaceLeaf } from "obsidian";
import type LilbeePlugin from "../main";
import { SSE_EVENT } from "../types";
import type { Message, Source } from "../types";
import { renderSourceChip } from "./results";

export const VIEW_TYPE_CHAT = "lilbee-chat";

export class ChatView extends ItemView {
    private plugin: LilbeePlugin;
    private history: Message[] = [];
    private messagesEl: HTMLElement | null = null;
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

        // Clear chat button
        const toolbar = container.createDiv({ cls: "lilbee-chat-toolbar" });

        // Model selector
        const modelSelect = toolbar.createEl("select", { cls: "lilbee-chat-model-select" });
        this.populateModelSelector(modelSelect);

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
        const sendBtn = inputArea.createEl("button", {
            text: "Send",
            cls: "lilbee-chat-send",
        });

        const handleSend = (): void => {
            const text = textarea.value.trim();
            if (!text) return;
            textarea.value = "";
            void this.sendMessage(text);
        };

        sendBtn.addEventListener("click", handleSend);
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

        // Render user bubble
        const userBubble = this.messagesEl.createDiv({
            cls: "lilbee-chat-message user",
        });
        userBubble.createEl("p", { text });

        // Push to history
        this.history.push({ role: "user", content: text });

        // Render assistant bubble (initially empty)
        const assistantBubble = this.messagesEl.createDiv({
            cls: "lilbee-chat-message assistant",
        });
        const textEl = assistantBubble.createEl("p");
        assistantBubble.scrollIntoView({ behavior: "smooth" });

        let fullContent = "";
        const sources: Source[] = [];

        try {
            for await (const event of this.plugin.api.chatStream(
                text,
                this.history.slice(0, -1), // current question is sent separately; don't duplicate in history
                this.plugin.settings.topK,
            )) {
                if (event.event === SSE_EVENT.TOKEN) {
                    const token = String(event.data);
                    fullContent += token;
                    textEl.textContent = fullContent;
                    assistantBubble.scrollIntoView({ behavior: "smooth" });
                } else if (event.event === SSE_EVENT.SOURCES) {
                    const data = event.data as Source[];
                    sources.push(...data);
                } else if (event.event === SSE_EVENT.DONE) {
                    // Sources section
                    if (sources.length > 0) {
                        this.renderSources(assistantBubble, sources);
                    }
                    // Push assistant message to history
                    this.history.push({ role: "assistant", content: fullContent });
                } else if (event.event === SSE_EVENT.ERROR) {
                    new Notice(`lilbee: ${String(event.data)}`);
                }
            }
        } catch {
            new Notice("lilbee: chat error — cannot connect to server");
            textEl.textContent = "Error: could not connect to the lilbee server.";
        } finally {
            this.sending = false;
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
