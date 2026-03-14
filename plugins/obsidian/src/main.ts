import { Notice, Plugin, type TAbstractFile } from "obsidian";
import { LilbeeClient } from "./api";
import { HEALTH_STATE, HealthDetector, type HealthState } from "./health-detector";
import { LilbeeSettingTab } from "./settings";
import { DEFAULT_SETTINGS, SSE_EVENT, type LilbeeSettings, type SSEEvent, type SyncDone } from "./types";
import { ChatView, VIEW_TYPE_CHAT } from "./views/chat-view";
import { SearchModal } from "./views/search-modal";

export default class LilbeePlugin extends Plugin {
    settings: LilbeeSettings = { ...DEFAULT_SETTINGS };
    api: LilbeeClient = new LilbeeClient(DEFAULT_SETTINGS.serverUrl);
    ollamaDetector: HealthDetector | null = null;
    serverDetector: HealthDetector | null = null;
    activeModel = "";
    activeVisionModel = "";
    statusBarEl: HTMLElement | null = null;
    private syncTimeout: ReturnType<typeof setTimeout> | null = null;
    private autoSyncRefs: { id: string }[] = [];

    async onload(): Promise<void> {
        await this.loadSettings();
        this.api = new LilbeeClient(this.settings.serverUrl);

        // Status bar
        this.statusBarEl = this.addStatusBarItem();
        this.setStatusReady();

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
            id: "lilbee:add-file",
            name: "Add current file to lilbee",
            checkCallback: (checking) => {
                const file = this.app.workspace.getActiveFile();
                if (!file) return false;
                if (!checking) void this.addToLilbee(file);
                return true;
            },
        });

        this.addCommand({
            id: "lilbee:add-folder",
            name: "Add current folder to lilbee",
            checkCallback: (checking) => {
                const file = this.app.workspace.getActiveFile();
                const folder = file?.parent;
                if (!folder) return false;
                if (!checking) void this.addToLilbee(folder);
                return true;
            },
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

        // Health detectors — check once on startup, then only on operation failure
        this.serverDetector = new HealthDetector({
            url: `${this.settings.serverUrl}/api/health`,
            onStateChange: (state) => this.onServerHealthChange(state),
        });
        void this.serverDetector.check();

        this.ollamaDetector = new HealthDetector({
            url: this.settings.ollamaUrl,
            onStateChange: (state) => this.onOllamaStateChange(state),
        });
        void this.ollamaDetector.check();

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
        this.recreateDetectors();
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
            this.activeVisionModel = models.vision.active;
            this.setStatusReady();
        }).catch(() => {});
    }

    async addExternalFiles(paths: string[]): Promise<void> {
        if (!this.statusBarEl || paths.length === 0) return;
        const label = paths.length === 1 ? paths[0].split("/").pop() : `${paths.length} files`;
        new Notice(`lilbee: adding ${label}...`);
        await this.runAdd(paths);
    }

    async addToLilbee(file: TAbstractFile): Promise<void> {
        if (!this.statusBarEl) return;
        const adapter = this.app.vault.adapter as unknown as { getBasePath(): string };
        const absolutePath = `${adapter.getBasePath()}/${file.path}`;
        new Notice(`lilbee: adding ${file.name ?? file.path}...`);
        await this.runAdd([absolutePath]);
    }

    private async runAdd(paths: string[]): Promise<void> {
        this.updateStatusBar("lilbee: adding...");

        try {
            let lastEvent: SSEEvent | null = null;
            for await (const event of this.api.addFiles(paths, false, this.activeVisionModel || undefined)) {
                this.handleProgressEvent(event);
                lastEvent = event;
            }

            if (lastEvent?.event === SSE_EVENT.DONE) {
                const done = lastEvent.data as SyncDone;
                const parts: string[] = [];
                if (done.added.length > 0) parts.push(`${done.added.length} added`);
                if (done.failed.length > 0) parts.push(`${done.failed.length} failed`);
                new Notice(parts.length > 0
                    ? `lilbee: ${parts.join(", ")}`
                    : "lilbee: nothing new to add");
            }
        } catch (err) {
            const msg = err instanceof Error ? err.message : "cannot connect to server";
            new Notice(`lilbee: add failed — ${msg}`);
            await this.checkHealthOnError();
            return;
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

    private recreateDetectors(): void {
        this.serverDetector = new HealthDetector({
            url: `${this.settings.serverUrl}/api/health`,
            onStateChange: (state) => this.onServerHealthChange(state),
        });

        this.ollamaDetector = new HealthDetector({
            url: this.settings.ollamaUrl,
            onStateChange: (state) => this.onOllamaStateChange(state),
        });
    }

    /** Run after an operation fails to update status bar if server/ollama went down. */
    private async checkHealthOnError(): Promise<void> {
        await this.serverDetector?.check();
        await this.ollamaDetector?.check();
    }

    private onOllamaStateChange(state: HealthState): void {
        if (!this.statusBarEl) return;
        if (state === HEALTH_STATE.UNREACHABLE) {
            this.updateStatusBar("lilbee: ready (Ollama offline)");
            new Notice(
                "Ollama is not running. Sync, ask, and chat require Ollama.\n" +
                    "Start Ollama or install it from https://ollama.com",
            );
        } else if (state === HEALTH_STATE.REACHABLE) {
            this.setStatusReady();
        }
    }

    private onServerHealthChange(state: HealthState): void {
        if (!this.statusBarEl) return;
        if (state === HEALTH_STATE.UNREACHABLE) {
            this.updateStatusBar("lilbee: server offline");
            new Notice("lilbee server is not running. Start it with: lilbee serve");
        } else if (state === HEALTH_STATE.REACHABLE) {
            this.setStatusReady();
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
        this.updateStatusBar("lilbee: syncing...");

        try {
            let lastEvent: SSEEvent | null = null;
            for await (const event of this.api.syncStream(!!this.activeVisionModel)) {
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
            await this.checkHealthOnError();
            return;
        }

        this.setStatusReady();
    }
}
