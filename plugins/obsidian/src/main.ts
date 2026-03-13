import { Notice, Plugin, type TAbstractFile } from "obsidian";
import { LilbeeClient } from "./api";
import { OLLAMA_STATE, OllamaDetector, type OllamaState } from "./ollama-detector";
import { SERVER_STATE, ServerManager, vaultPort, type ServerState } from "./server-manager";
import { LilbeeSettingTab } from "./settings";
import { DEFAULT_SETTINGS, SSE_EVENT, type LilbeeSettings, type SSEEvent, type SyncDone } from "./types";
import { ChatView, VIEW_TYPE_CHAT } from "./views/chat-view";
import { SearchModal } from "./views/search-modal";

export default class LilbeePlugin extends Plugin {
    settings: LilbeeSettings = { ...DEFAULT_SETTINGS };
    api: LilbeeClient = new LilbeeClient(DEFAULT_SETTINGS.serverUrl);
    ollamaDetector: OllamaDetector | null = null;
    activeModel = "";
    statusBarEl: HTMLElement | null = null;
    private syncTimeout: ReturnType<typeof setTimeout> | null = null;
    private autoSyncRefs: { id: string }[] = [];
    private serverManager: ServerManager | null = null;

    async onload(): Promise<void> {
        await this.loadSettings();
        this.api = new LilbeeClient(this.settings.serverUrl);

        // Status bar
        this.statusBarEl = this.addStatusBarItem();
        this.setStatusReady();

        // Managed server
        if (this.settings.manageServer) {
            this.startManagedServer();
        }

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

        // File explorer context menu
        this.registerEvent(
            this.app.workspace.on("file-menu" as any, (menu: any, file: TAbstractFile) => {
                menu.addItem((item: any) => {
                    item.setTitle("Add to lilbee")
                        .setIcon("plus-circle")
                        .onClick(() => this.addToLilbee(file));
                });
            }),
        );

        // Fetch models to populate activeModel
        this.fetchActiveModel();

        // Ollama detector — runs regardless of manageServer
        this.ollamaDetector = new OllamaDetector({
            ollamaUrl: this.settings.ollamaUrl,
            onStateChange: (state) => this.onOllamaStateChange(state),
        });
        this.ollamaDetector.startPolling();

        // Auto-sync watcher
        if (this.settings.syncMode === "auto") {
            this.registerAutoSync();
        }
    }

    onunload(): void {
        if (this.syncTimeout) {
            clearTimeout(this.syncTimeout);
        }
        this.ollamaDetector?.stopPolling();
        this.stopManagedServer();
    }

    async loadSettings(): Promise<void> {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings(): Promise<void> {
        await this.saveData(this.settings);
        this.api = new LilbeeClient(this.settings.serverUrl);
        this.updateAutoSync();
        this.recreateOllamaDetector();
    }

    private updateStatusBar(text: string): void {
        if (!this.statusBarEl) return;
        const model = this.activeModel ? ` (${this.activeModel})` : "";
        this.statusBarEl.setText(`${text}${model}`);
    }

    private setStatusReady(): void {
        this.updateStatusBar("lilbee: ready");
    }

    fetchActiveModel(): void {
        this.api.listModels().then((models) => {
            this.activeModel = models.chat.active;
            this.setStatusReady();
        }).catch(() => {});
    }

    async addToLilbee(file: TAbstractFile): Promise<void> {
        if (!this.statusBarEl) return;
        const adapter = this.app.vault.adapter as unknown as { getBasePath(): string };
        const absolutePath = `${adapter.getBasePath()}/${file.path}`;

        this.updateStatusBar("lilbee: adding...");

        try {
            let lastEvent: SSEEvent | null = null;
            for await (const event of this.api.addFiles([absolutePath])) {
                this.handleProgressEvent(event);
                lastEvent = event;
            }

            if (lastEvent?.event === SSE_EVENT.DONE) {
                const done = lastEvent.data as SyncDone;
                const parts: string[] = [];
                if (done.added.length > 0) parts.push(`${done.added.length} added`);
                if (done.failed.length > 0) parts.push(`${done.failed.length} failed`);
                if (parts.length > 0) {
                    new Notice(`lilbee: ${parts.join(", ")}`);
                }
            }
        } catch {
            new Notice("lilbee: add failed — cannot connect to server");
        }

        this.setStatusReady();
    }

    private handleProgressEvent(event: SSEEvent): void {
        if (!this.statusBarEl) return;
        switch (event.event) {
            case SSE_EVENT.FILE_START: {
                const data = event.data as { file: string; current_file: number; total_files: number };
                this.updateStatusBar(`lilbee: indexing ${data.current_file}/${data.total_files} — ${data.file}`);
                break;
            }
            case SSE_EVENT.EXTRACT: {
                const data = event.data as { file: string; page: number; total_pages: number };
                this.updateStatusBar(`lilbee: extracting ${data.file} (page ${data.page}/${data.total_pages})`);
                break;
            }
            case SSE_EVENT.EMBED: {
                const data = event.data as { file: string; chunk: number; total_chunks: number };
                this.updateStatusBar(`lilbee: embedding ${data.file} (${data.chunk}/${data.total_chunks} chunks)`);
                break;
            }
            case SSE_EVENT.PROGRESS: {
                const data = event.data as { file: string; current: number; total: number };
                this.updateStatusBar(`lilbee: indexing ${data.current}/${data.total} — ${data.file}`);
                break;
            }
        }
    }

    private recreateOllamaDetector(): void {
        this.ollamaDetector?.stopPolling();
        this.ollamaDetector = new OllamaDetector({
            ollamaUrl: this.settings.ollamaUrl,
            onStateChange: (state) => this.onOllamaStateChange(state),
        });
        this.ollamaDetector.startPolling();
    }

    private startManagedServer(): void {
        const adapter = this.app.vault.adapter as unknown as { getBasePath(): string };
        const vaultPath = adapter.getBasePath();
        const dataDir = `${vaultPath}/.lilbee`;
        const port = vaultPort(vaultPath);

        this.settings.serverUrl = `http://127.0.0.1:${port}`;
        this.api = new LilbeeClient(this.settings.serverUrl);

        this.serverManager = new ServerManager({
            binaryPath: this.settings.binaryPath,
            dataDir,
            host: "127.0.0.1",
            port,
            onStateChange: (state, detail) => this.onServerStateChange(state, detail),
        });

        this.serverManager.start().catch(() => {});
    }

    private stopManagedServer(): void {
        if (!this.serverManager) return;
        this.serverManager.stop().catch(() => {});
        this.serverManager = null;
    }

    async restartServer(): Promise<void> {
        if (!this.serverManager) return;
        await this.serverManager.restart();
    }

    private onOllamaStateChange(state: OllamaState): void {
        if (!this.statusBarEl) return;
        if (state === OLLAMA_STATE.UNREACHABLE) {
            this.updateStatusBar("lilbee: ready (Ollama offline)");
            new Notice(
                "Ollama is not running. Sync, ask, and chat require Ollama.\n" +
                    "Start Ollama or install it from https://ollama.com",
                0,
            );
        } else if (state === OLLAMA_STATE.REACHABLE) {
            this.setStatusReady();
        }
    }

    private onServerStateChange(state: ServerState, detail?: string): void {
        if (!this.statusBarEl) return;
        switch (state) {
            case SERVER_STATE.STOPPED:
                this.updateStatusBar("lilbee: stopped");
                break;
            case SERVER_STATE.STARTING:
                this.updateStatusBar("lilbee: starting...");
                break;
            case SERVER_STATE.READY:
                this.setStatusReady();
                break;
            case SERVER_STATE.ERROR:
                this.updateStatusBar("lilbee: error");
                new Notice(`lilbee: ${detail ?? "server error"}`);
                break;
        }
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
        if (this.ollamaDetector?.state === OLLAMA_STATE.UNREACHABLE) {
            new Notice(
                "Cannot sync: Ollama is not running.\n" +
                    "Start Ollama or install it from https://ollama.com",
            );
            return;
        }
        this.updateStatusBar("lilbee: syncing...");

        try {
            let lastEvent: SSEEvent | null = null;
            for await (const event of this.api.syncStream()) {
                this.handleProgressEvent(event);
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

        this.setStatusReady();
    }
}
