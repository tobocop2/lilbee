import { Notice, Plugin } from "obsidian";
import { LilbeeClient } from "./api";
import { LilbeeSettingTab } from "./settings";
import { DEFAULT_SETTINGS, SSE_EVENT, type LilbeeSettings, type SSEEvent, type SyncDone } from "./types";
import { ChatView, VIEW_TYPE_CHAT } from "./views/chat-view";
import { SearchModal } from "./views/search-modal";

export default class LilbeePlugin extends Plugin {
    settings: LilbeeSettings = { ...DEFAULT_SETTINGS };
    api: LilbeeClient = new LilbeeClient(DEFAULT_SETTINGS.serverUrl);
    private syncTimeout: ReturnType<typeof setTimeout> | null = null;
    private statusBarEl: HTMLElement | null = null;
    private autoSyncRefs: { id: string }[] = [];

    async onload(): Promise<void> {
        await this.loadSettings();
        this.api = new LilbeeClient(this.settings.serverUrl);

        // Status bar
        this.statusBarEl = this.addStatusBarItem();
        this.statusBarEl.setText("lilbee: ready");

        // Register views
        this.registerView(VIEW_TYPE_CHAT, (leaf) => new ChatView(leaf, this));

        // Settings tab
        this.addSettingTab(new LilbeeSettingTab(this.app, this));

        // Commands
        this.addCommand({
            id: "lilbee:search",
            name: "Search knowledge base",
            callback: () => new SearchModal(this.app, this).open(),
        });

        this.addCommand({
            id: "lilbee:ask",
            name: "Ask a question",
            callback: () => {
                // Quick ask via search modal in ask mode
                const modal = new SearchModal(this.app, this, "ask");
                modal.open();
            },
        });

        this.addCommand({
            id: "lilbee:chat",
            name: "Open chat",
            callback: () => this.activateChatView(),
        });

        this.addCommand({
            id: "lilbee:sync",
            name: "Sync vault",
            callback: () => this.triggerSync(),
        });

        this.addCommand({
            id: "lilbee:status",
            name: "Show status",
            callback: async () => {
                try {
                    const status = await this.api.status();
                    new Notice(
                        `lilbee: ${status.sources.length} documents, ${status.total_chunks} chunks`,
                    );
                } catch {
                    new Notice("lilbee: cannot connect to server");
                }
            },
        });

        // Auto-sync watcher
        if (this.settings.syncMode === "auto") {
            this.registerAutoSync();
        }
    }

    onunload(): void {
        if (this.syncTimeout) {
            clearTimeout(this.syncTimeout);
        }
    }

    async loadSettings(): Promise<void> {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings(): Promise<void> {
        await this.saveData(this.settings);
        this.api = new LilbeeClient(this.settings.serverUrl);
        this.updateAutoSync();
    }

    private updateAutoSync(): void {
        if (this.settings.syncMode === "auto" && this.autoSyncRefs.length === 0) {
            this.registerAutoSync();
        } else if (this.settings.syncMode === "manual" && this.autoSyncRefs.length > 0) {
            this.unregisterAutoSync();
        }
    }

    private unregisterAutoSync(): void {
        // Obsidian's registerEvent() ties cleanup to plugin unload only —
        // there is no unregisterEvent(). Clearing refs prevents re-registration
        // but existing listeners remain active until the plugin is reloaded.
        this.autoSyncRefs = [];
    }

    private async activateChatView(): Promise<void> {
        const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE_CHAT);
        if (existing.length > 0) {
            this.app.workspace.revealLeaf(existing[0]);
            return;
        }
        const leaf = this.app.workspace.getRightLeaf(false);
        if (leaf) {
            await leaf.setViewState({ type: VIEW_TYPE_CHAT, active: true });
            this.app.workspace.revealLeaf(leaf);
        }
    }

    private registerAutoSync(): void {
        const handler = () => this.debouncedSync();
        const vault = this.app.vault;
        const refs = [
            vault.on("create", handler),
            vault.on("modify", handler),
            vault.on("delete", handler),
            vault.on("rename", handler),
        ];
        for (const ref of refs) {
            this.autoSyncRefs.push(ref as { id: string });
            this.registerEvent(ref);
        }
    }

    private debouncedSync(): void {
        if (this.syncTimeout) {
            clearTimeout(this.syncTimeout);
        }
        this.syncTimeout = setTimeout(() => {
            this.triggerSync();
        }, this.settings.syncDebounceMs);
    }

    async triggerSync(): Promise<void> {
        if (!this.statusBarEl) return;
        this.statusBarEl.setText("lilbee: syncing...");

        try {
            let lastEvent: SSEEvent | null = null;
            for await (const event of this.api.syncStream()) {
                if (event.event === SSE_EVENT.PROGRESS) {
                    const data = event.data as {
                        file: string;
                        current: number;
                        total: number;
                    };
                    this.statusBarEl.setText(
                        `lilbee: indexing ${data.current}/${data.total} — ${data.file}`,
                    );
                }
                lastEvent = event;
            }

            if (lastEvent?.event === SSE_EVENT.DONE) {
                const done = lastEvent.data as SyncDone;
                const parts: string[] = [];
                if (done.added.length > 0) parts.push(`${done.added.length} added`);
                if (done.updated.length > 0) parts.push(`${done.updated.length} updated`);
                if (done.removed.length > 0) parts.push(`${done.removed.length} removed`);
                if (done.failed.length > 0) parts.push(`${done.failed.length} failed`);
                if (parts.length > 0) {
                    new Notice(`lilbee: synced — ${parts.join(", ")}`);
                }
            }
        } catch {
            new Notice("lilbee: sync failed — cannot connect to server");
        }

        this.statusBarEl.setText("lilbee: ready");
    }
}
