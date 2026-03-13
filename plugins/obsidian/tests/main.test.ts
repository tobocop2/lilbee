import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { Notice } from "obsidian";
import { App, WorkspaceLeaf } from "./__mocks__/obsidian";
import { SSE_EVENT } from "../src/types";

vi.mock("../src/api", () => ({
    LilbeeClient: vi.fn().mockImplementation(() => ({
        status: vi.fn(),
        syncStream: vi.fn(),
        search: vi.fn(),
        ask: vi.fn(),
        chatStream: vi.fn(),
        listModels: vi.fn().mockRejectedValue(new Error("offline")),
        pullModel: vi.fn(),
        setChatModel: vi.fn(),
        setVisionModel: vi.fn(),
        health: vi.fn(),
        addFiles: vi.fn(),
    })),
}));

vi.mock("../src/ollama-detector", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../src/ollama-detector")>();
    return {
        ...actual,
        OllamaDetector: vi.fn().mockImplementation(() => ({
            startPolling: vi.fn(),
            stopPolling: vi.fn(),
            check: vi.fn().mockResolvedValue("unknown"),
            state: "unknown",
        })),
    };
});

vi.mock("../src/server-manager", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../src/server-manager")>();
    return {
        ...actual,
        ServerManager: vi.fn().mockImplementation(() => ({
            start: vi.fn().mockResolvedValue(undefined),
            stop: vi.fn().mockResolvedValue(undefined),
            restart: vi.fn().mockResolvedValue(undefined),
            state: "stopped",
        })),
        vaultPort: vi.fn().mockReturnValue(7500),
    };
});

// We also need to mock the views to avoid loading heavy deps
vi.mock("../src/views/chat-view", () => ({
    VIEW_TYPE_CHAT: "lilbee-chat",
    ChatView: vi.fn().mockImplementation(() => ({})),
}));

vi.mock("../src/views/search-modal", () => ({
    SearchModal: vi.fn().mockImplementation(() => ({ open: vi.fn() })),
}));

async function createPlugin() {
    const { default: LilbeePlugin } = await import("../src/main");
    const app = new App();
    // Plugin constructor calls super(app, manifest)
    const plugin = new LilbeePlugin(app as any, {
        id: "lilbee",
        name: "lilbee",
        version: "0.1.0",
        minAppVersion: "1.0.0",
        author: "test",
        description: "test",
    } as any);
    return plugin;
}

describe("LilbeePlugin", () => {
    beforeEach(() => {
        Notice.clear();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    describe("onload()", () => {
        it("loads settings, creates API client, sets up status bar", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            expect(plugin.addStatusBarItem).toHaveBeenCalled();
            expect(plugin.addSettingTab).toHaveBeenCalled();
            expect(plugin.registerView).toHaveBeenCalled();
        });

        it("adds all five commands", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            expect(plugin.addCommand).toHaveBeenCalledTimes(5);
            const ids = (plugin.addCommand as ReturnType<typeof vi.fn>).mock.calls.map(
                (c: any[]) => c[0].id,
            );
            expect(ids).toContain("lilbee:search");
            expect(ids).toContain("lilbee:ask");
            expect(ids).toContain("lilbee:chat");
            expect(ids).toContain("lilbee:sync");
            expect(ids).toContain("lilbee:status");
        });

        it("sets status bar text to 'lilbee: ready'", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("with manual sync mode: registers only file-menu event (no vault events)", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "manual" });
            await plugin.onload();
            // 1 for file-menu
            expect(plugin.registerEvent).toHaveBeenCalledTimes(1);
        });

        it("with auto sync mode: registers vault events + file-menu", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "auto" });
            await plugin.onload();
            // 4 vault events + 1 file-menu = 5
            expect(plugin.registerEvent).toHaveBeenCalledTimes(5);
        });

        it("recreates API client with loaded serverUrl", async () => {
            const { LilbeeClient } = await import("../src/api");
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ serverUrl: "http://custom:9999" });
            await plugin.onload();
            // LilbeeClient should have been called with the custom URL
            expect(LilbeeClient).toHaveBeenCalledWith("http://custom:9999");
        });
    });

    describe("onunload()", () => {
        it("clears sync timeout if one is active", async () => {
            vi.useFakeTimers();
            const plugin = await createPlugin();
            await plugin.onload();
            // Schedule a debounced sync
            (plugin as any).debouncedSync();
            expect((plugin as any).syncTimeout).not.toBeNull();
            const clearSpy = vi.spyOn(globalThis, "clearTimeout");
            plugin.onunload();
            expect(clearSpy).toHaveBeenCalled();
        });

        it("does not throw when no timeout is active", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            expect(() => plugin.onunload()).not.toThrow();
        });
    });

    describe("loadSettings()", () => {
        it("merges saved data over defaults", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ topK: 15, syncMode: "auto" });
            await plugin.loadSettings();
            expect(plugin.settings.topK).toBe(15);
            expect(plugin.settings.syncMode).toBe("auto");
            // Default preserved for unset fields
            expect(plugin.settings.serverUrl).toBe("http://127.0.0.1:7433");
        });

        it("uses defaults when loadData returns null/empty", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue(null);
            await plugin.loadSettings();
            expect(plugin.settings.topK).toBe(5);
            expect(plugin.settings.syncMode).toBe("manual");
        });
    });

    describe("saveSettings()", () => {
        it("calls saveData and recreates the API client", async () => {
            const { LilbeeClient } = await import("../src/api");
            const plugin = await createPlugin();
            await plugin.onload();
            const callsBefore = (LilbeeClient as ReturnType<typeof vi.fn>).mock.calls.length;

            plugin.settings.serverUrl = "http://newserver:8080";
            await plugin.saveSettings();

            expect(plugin.saveData).toHaveBeenCalledWith(plugin.settings);
            const callsAfter = (LilbeeClient as ReturnType<typeof vi.fn>).mock.calls.length;
            expect(callsAfter).toBeGreaterThan(callsBefore);
        });

        it("calls updateAutoSync after saving", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const updateSpy = vi.spyOn(plugin as any, "updateAutoSync");
            await plugin.saveSettings();

            expect(updateSpy).toHaveBeenCalled();
        });
    });

    describe("updateAutoSync()", () => {
        it("registers vault events when switching from manual to auto", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "manual" });
            await plugin.onload();

            // Only file-menu registered initially
            expect(plugin.registerEvent).toHaveBeenCalledTimes(1);

            // Switch to auto and save
            plugin.settings.syncMode = "auto";
            await plugin.saveSettings();

            // 1 file-menu + 4 vault events = 5
            expect(plugin.registerEvent).toHaveBeenCalledTimes(5);
        });

        it("clears autoSyncRefs when switching from auto to manual", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "auto" });
            await plugin.onload();

            // autoSyncRefs should be populated after auto-sync registration
            expect((plugin as any).autoSyncRefs.length).toBe(4);

            // Switch to manual and save
            plugin.settings.syncMode = "manual";
            await plugin.saveSettings();

            expect((plugin as any).autoSyncRefs.length).toBe(0);
        });

        it("does not re-register events when already in auto mode", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "auto" });
            await plugin.onload();

            const callsBefore = (plugin.registerEvent as ReturnType<typeof vi.fn>).mock.calls.length;

            // Save again with auto still active — should not register vault events again
            await plugin.saveSettings();

            expect((plugin.registerEvent as ReturnType<typeof vi.fn>).mock.calls.length).toBe(callsBefore);
        });
    });

    describe("settings initialisation", () => {
        it("settings is a separate object from DEFAULT_SETTINGS", async () => {
            const { default: LilbeePlugin } = await import("../src/main");
            const { DEFAULT_SETTINGS } = await import("../src/types");
            const app = new App();
            const plugin = new LilbeePlugin(app as any, {
                id: "lilbee",
                name: "lilbee",
                version: "0.1.0",
                minAppVersion: "1.0.0",
                author: "test",
                description: "test",
            } as any);

            expect(plugin.settings).not.toBe(DEFAULT_SETTINGS);
        });
    });

    describe("activateChatView()", () => {
        it("reveals existing leaf when chat view is already open", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const leaf = new WorkspaceLeaf(plugin.app as any);
            plugin.app.workspace.getLeavesOfType = vi.fn().mockReturnValue([leaf]);

            await (plugin as any).activateChatView();

            expect(plugin.app.workspace.revealLeaf).toHaveBeenCalledWith(leaf);
            expect(plugin.app.workspace.getRightLeaf).not.toHaveBeenCalled();
        });

        it("sets view state on right leaf when no chat view exists", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.app.workspace.getLeavesOfType = vi.fn().mockReturnValue([]);
            const leaf = new WorkspaceLeaf(plugin.app as any);
            plugin.app.workspace.getRightLeaf = vi.fn().mockReturnValue(leaf);

            await (plugin as any).activateChatView();

            expect(leaf.setViewState).toHaveBeenCalledWith({ type: "lilbee-chat", active: true });
            expect(plugin.app.workspace.revealLeaf).toHaveBeenCalledWith(leaf);
        });

        it("does not crash when getRightLeaf returns null", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.app.workspace.getLeavesOfType = vi.fn().mockReturnValue([]);
            plugin.app.workspace.getRightLeaf = vi.fn().mockReturnValue(null);

            await expect((plugin as any).activateChatView()).resolves.not.toThrow();
        });
    });

    describe("debouncedSync()", () => {
        it("schedules triggerSync after debounce delay", async () => {
            vi.useFakeTimers();
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncDebounceMs: 1000 });
            await plugin.onload();

            const triggerSpy = vi.spyOn(plugin, "triggerSync").mockResolvedValue(undefined);
            (plugin as any).debouncedSync();

            expect(triggerSpy).not.toHaveBeenCalled();
            vi.advanceTimersByTime(1000);
            expect(triggerSpy).toHaveBeenCalledTimes(1);
        });

        it("cancels previous timer when called again", async () => {
            vi.useFakeTimers();
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncDebounceMs: 500 });
            await plugin.onload();

            const triggerSpy = vi.spyOn(plugin, "triggerSync").mockResolvedValue(undefined);
            (plugin as any).debouncedSync();
            vi.advanceTimersByTime(200);
            (plugin as any).debouncedSync();
            vi.advanceTimersByTime(500);

            // Should only fire once (the second call)
            expect(triggerSpy).toHaveBeenCalledTimes(1);
        });
    });

    describe("triggerSync()", () => {
        it("updates status bar during sync and resets to 'ready'", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* noEvents() {}
            plugin.api.syncStream = vi.fn().mockReturnValue(noEvents());

            await plugin.triggerSync();

            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("updates status bar text for progress events", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const statusTexts: string[] = [];
            const origSetText = (plugin as any).statusBarEl!.setText.bind((plugin as any).statusBarEl);
            (plugin as any).statusBarEl!.setText = (text: string) => {
                statusTexts.push(text);
                origSetText(text);
            };

            async function* withProgress() {
                yield { event: SSE_EVENT.PROGRESS, data: { file: "notes.md", current: 1, total: 5 } };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withProgress());

            await plugin.triggerSync();

            expect(statusTexts.some((t) => t.includes("1/5") && t.includes("notes.md"))).toBe(true);
        });

        it("shows Notice with all stats when done event has populated arrays", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* withDone() {
                yield {
                    event: SSE_EVENT.DONE,
                    data: {
                        added: ["a.md"],
                        updated: ["b.md"],
                        removed: ["c.md"],
                        failed: ["d.md"],
                        unchanged: 0,
                    },
                };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withDone());

            await plugin.triggerSync();

            const msg = Notice.instances[0]?.message ?? "";
            expect(msg).toContain("1 added");
            expect(msg).toContain("1 updated");
            expect(msg).toContain("1 removed");
            expect(msg).toContain("1 failed");
        });

        it("does NOT show Notice when done event has all empty arrays", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* withEmptyDone() {
                yield {
                    event: SSE_EVENT.DONE,
                    data: { added: [], updated: [], removed: [], failed: [], unchanged: 10 },
                };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withEmptyDone());

            await plugin.triggerSync();

            expect(Notice.instances.length).toBe(0);
        });

        it("does NOT show Notice when last event is not 'done'", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* withProgress() {
                yield { event: SSE_EVENT.PROGRESS, data: { file: "x.md", current: 1, total: 1 } };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withProgress());

            await plugin.triggerSync();

            expect(Notice.instances.length).toBe(0);
        });

        it("shows error Notice and resets status bar on API error", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.syncStream = vi.fn().mockImplementation(() => {
                throw new Error("connection refused");
            });

            await plugin.triggerSync();

            expect(Notice.instances.some((n) => n.message.includes("sync failed"))).toBe(true);
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("returns early when statusBarEl is null", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            // syncStream should never be called if we return early
            const syncStreamSpy = vi.spyOn(plugin.api, "syncStream");
            await plugin.triggerSync();

            expect(syncStreamSpy).not.toHaveBeenCalled();
        });
    });

    describe("commands", () => {
        async function getCommandCallback(plugin: Awaited<ReturnType<typeof createPlugin>>, id: string) {
            const calls = (plugin.addCommand as ReturnType<typeof vi.fn>).mock.calls as Array<[{ id: string; callback: () => void | Promise<void> }]>;
            const call = calls.find((c) => c[0].id === id);
            return call?.[0].callback;
        }

        it("lilbee:search opens SearchModal", async () => {
            const { SearchModal } = await import("../src/views/search-modal");
            const plugin = await createPlugin();
            await plugin.onload();

            const cb = await getCommandCallback(plugin, "lilbee:search");
            cb?.();

            expect(SearchModal).toHaveBeenCalled();
            const instance = (SearchModal as ReturnType<typeof vi.fn>).mock.results[0].value;
            expect(instance.open).toHaveBeenCalled();
        });

        it("lilbee:ask opens SearchModal in 'ask' mode", async () => {
            const { SearchModal } = await import("../src/views/search-modal");
            const plugin = await createPlugin();
            await plugin.onload();

            const cb = await getCommandCallback(plugin, "lilbee:ask");
            cb?.();

            // Should be called with 'ask' as third argument
            const calls = (SearchModal as ReturnType<typeof vi.fn>).mock.calls;
            const askCall = calls.find((c: any[]) => c[2] === "ask");
            expect(askCall).toBeDefined();
        });

        it("lilbee:chat calls activateChatView", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const activateSpy = vi
                .spyOn(plugin as any, "activateChatView")
                .mockResolvedValue(undefined);
            const cb = await getCommandCallback(plugin, "lilbee:chat");
            cb?.();

            expect(activateSpy).toHaveBeenCalled();
        });

        it("lilbee:sync calls triggerSync", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const syncSpy = vi.spyOn(plugin, "triggerSync").mockResolvedValue(undefined);
            const cb = await getCommandCallback(plugin, "lilbee:sync");
            cb?.();

            expect(syncSpy).toHaveBeenCalled();
        });

        it("lilbee:status shows status Notice on success", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.status = vi.fn().mockResolvedValue({
                sources: [{ filename: "a.md", chunk_count: 3 }, { filename: "b.md", chunk_count: 2 }],
                total_chunks: 5,
                config: {},
            });

            const cb = await getCommandCallback(plugin, "lilbee:status");
            await cb?.();

            expect(Notice.instances.some((n) => n.message.includes("2 documents") && n.message.includes("5 chunks"))).toBe(true);
        });

        it("lilbee:status shows error Notice on API failure", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.status = vi.fn().mockRejectedValue(new Error("timeout"));

            const cb = await getCommandCallback(plugin, "lilbee:status");
            await cb?.();

            expect(Notice.instances.some((n) => n.message.includes("cannot connect"))).toBe(true);
        });
    });

    describe("registerAutoSync()", () => {
        it("vault event callbacks call debouncedSync", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "auto" });

            const debouncedSpy = vi.spyOn(plugin as any, "debouncedSync").mockImplementation(() => {});

            await plugin.onload();

            const vaultOnCalls = (plugin.app.vault.on as ReturnType<typeof vi.fn>).mock.calls as Array<[string, () => void]>;
            expect(vaultOnCalls.length).toBe(4);

            // Trigger the callback from the first vault.on call
            vaultOnCalls[0][1]();
            expect(debouncedSpy).toHaveBeenCalledTimes(1);
        });
    });

    describe("managed server", () => {
        it("onload() with manageServer: true creates ServerManager and starts it", async () => {
            const { ServerManager } = await import("../src/server-manager");
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();

            expect(ServerManager).toHaveBeenCalled();
            const instance = (ServerManager as ReturnType<typeof vi.fn>).mock.results[0].value;
            expect(instance.start).toHaveBeenCalled();
        });

        it("onload() with manageServer: false skips ServerManager", async () => {
            const { ServerManager } = await import("../src/server-manager");
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: false });
            (ServerManager as ReturnType<typeof vi.fn>).mockClear();
            await plugin.onload();

            expect(ServerManager).not.toHaveBeenCalled();
        });

        it("onunload() stops managed server", async () => {
            const { ServerManager } = await import("../src/server-manager");
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();

            const instance = (ServerManager as ReturnType<typeof vi.fn>).mock.results[0].value;
            plugin.onunload();

            expect(instance.stop).toHaveBeenCalled();
        });

        it("onunload() does not crash when no server manager", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: false });
            await plugin.onload();
            expect(() => plugin.onunload()).not.toThrow();
        });

        it("status bar updates per server state", async () => {
            const { ServerManager } = await import("../src/server-manager");
            let onStateChange: (state: string, detail?: string) => void = () => {};
            (ServerManager as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    start: vi.fn().mockResolvedValue(undefined),
                    stop: vi.fn().mockResolvedValue(undefined),
                    restart: vi.fn().mockResolvedValue(undefined),
                    state: "stopped",
                };
            });

            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();

            onStateChange("starting");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: starting...");

            onStateChange("ready");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");

            onStateChange("error", "port in use");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: error");
            expect(Notice.instances.some((n) => n.message.includes("port in use"))).toBe(true);

            onStateChange("stopped");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: stopped");
        });

        it("error Notice uses default message when detail is undefined", async () => {
            const { ServerManager } = await import("../src/server-manager");
            let onStateChange: (state: string, detail?: string) => void = () => {};
            (ServerManager as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    start: vi.fn().mockResolvedValue(undefined),
                    stop: vi.fn().mockResolvedValue(undefined),
                    restart: vi.fn().mockResolvedValue(undefined),
                    state: "stopped",
                };
            });

            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();

            onStateChange("error");
            expect(Notice.instances.some((n) => n.message.includes("server error"))).toBe(true);
        });

        it("restartServer() calls serverManager.restart()", async () => {
            const { ServerManager } = await import("../src/server-manager");
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();

            const instance = (ServerManager as ReturnType<typeof vi.fn>).mock.results[0].value;
            await plugin.restartServer();
            expect(instance.restart).toHaveBeenCalled();
        });

        it("restartServer() no-ops when no server manager", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: false });
            await plugin.onload();
            await expect(plugin.restartServer()).resolves.not.toThrow();
        });

        it("onServerStateChange no-ops when statusBarEl is null", async () => {
            const { ServerManager } = await import("../src/server-manager");
            let onStateChange: (state: string, detail?: string) => void = () => {};
            (ServerManager as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    start: vi.fn().mockResolvedValue(undefined),
                    stop: vi.fn().mockResolvedValue(undefined),
                    restart: vi.fn().mockResolvedValue(undefined),
                    state: "stopped",
                };
            });

            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ manageServer: true });
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            expect(() => onStateChange("error", "test")).not.toThrow();
        });
    });

    describe("ollama detector", () => {
        it("onload creates OllamaDetector and starts polling", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            const plugin = await createPlugin();
            await plugin.onload();

            expect(OllamaDetector).toHaveBeenCalled();
            const instance = (OllamaDetector as ReturnType<typeof vi.fn>).mock.results[0].value;
            expect(instance.startPolling).toHaveBeenCalled();
        });

        it("onunload stops polling", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            const plugin = await createPlugin();
            await plugin.onload();

            const instance = (OllamaDetector as ReturnType<typeof vi.fn>).mock.results[0].value;
            plugin.onunload();
            expect(instance.stopPolling).toHaveBeenCalled();
        });

        it("onOllamaStateChange('unreachable') updates status bar and shows Notice", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            let onStateChange: (state: string) => void = () => {};
            (OllamaDetector as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    startPolling: vi.fn(),
                    stopPolling: vi.fn(),
                    state: "unknown",
                };
            });

            const plugin = await createPlugin();
            await plugin.onload();

            onStateChange("unreachable");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready (Ollama offline)");
            expect(Notice.instances.some((n) => n.message.includes("Ollama is not running"))).toBe(true);
            expect(Notice.instances.some((n) => n.duration === 0)).toBe(true);
        });

        it("onOllamaStateChange('reachable') restores status bar to ready", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            let onStateChange: (state: string) => void = () => {};
            (OllamaDetector as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    startPolling: vi.fn(),
                    stopPolling: vi.fn(),
                    state: "unknown",
                };
            });

            const plugin = await createPlugin();
            await plugin.onload();

            onStateChange("reachable");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("onOllamaStateChange no-ops when statusBarEl is null", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            let onStateChange: (state: string) => void = () => {};
            (OllamaDetector as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                onStateChange = opts.onStateChange;
                return {
                    startPolling: vi.fn(),
                    stopPolling: vi.fn(),
                    state: "unknown",
                };
            });

            const plugin = await createPlugin();
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            expect(() => onStateChange("unreachable")).not.toThrow();
        });

        it("saveSettings recreates detector with new ollamaUrl", async () => {
            const { OllamaDetector } = await import("../src/ollama-detector");
            let latestOnStateChange: (state: string) => void = () => {};
            (OllamaDetector as ReturnType<typeof vi.fn>).mockImplementation((opts: any) => {
                latestOnStateChange = opts.onStateChange;
                return {
                    startPolling: vi.fn(),
                    stopPolling: vi.fn(),
                    state: "unknown",
                };
            });

            const plugin = await createPlugin();
            await plugin.onload();

            const callsBefore = (OllamaDetector as ReturnType<typeof vi.fn>).mock.calls.length;
            const oldInstance = (OllamaDetector as ReturnType<typeof vi.fn>).mock.results.at(-1)!.value;

            plugin.settings.ollamaUrl = "http://remote:11434";
            await plugin.saveSettings();

            expect(oldInstance.stopPolling).toHaveBeenCalled();
            const callsAfter = (OllamaDetector as ReturnType<typeof vi.fn>).mock.calls.length;
            expect(callsAfter).toBeGreaterThan(callsBefore);

            // Verify the new detector's callback works
            latestOnStateChange("reachable");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("triggerSync early-returns when Ollama unreachable", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            // Set detector state to unreachable
            (plugin.ollamaDetector as any).state = "unreachable";

            const syncStreamSpy = vi.spyOn(plugin.api, "syncStream");
            await plugin.triggerSync();

            expect(syncStreamSpy).not.toHaveBeenCalled();
            expect(Notice.instances.some((n) => n.message.includes("Cannot sync"))).toBe(true);
        });

        it("triggerSync proceeds when Ollama reachable", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            (plugin.ollamaDetector as any).state = "reachable";
            async function* noEvents() {}
            plugin.api.syncStream = vi.fn().mockReturnValue(noEvents());

            await plugin.triggerSync();
            expect(plugin.api.syncStream).toHaveBeenCalled();
        });
    });

    describe("file-menu integration", () => {
        it("registers file-menu event on load", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            // workspace.on should have been called with "file-menu"
            const workspaceOnCalls = (plugin.app.workspace.on as ReturnType<typeof vi.fn>).mock.calls as Array<[string, ...unknown[]]>;
            const fileMenuCall = workspaceOnCalls.find((c) => c[0] === "file-menu");
            expect(fileMenuCall).toBeDefined();
        });

        it("file-menu callback invokes addToLilbee", async () => {
            const plugin = await createPlugin();
            const addSpy = vi.spyOn(plugin as any, "addToLilbee").mockResolvedValue(undefined);
            await plugin.onload();

            // Find the file-menu callback registered via workspace.on
            const workspaceOnCalls = (plugin.app.workspace.on as ReturnType<typeof vi.fn>).mock.calls as Array<[string, ...unknown[]]>;
            const fileMenuCall = workspaceOnCalls.find((c) => c[0] === "file-menu");
            const callback = fileMenuCall![1] as (menu: any, file: any) => void;

            // Simulate the menu structure Obsidian provides
            let menuItemCallback: (() => void) | null = null;
            const fakeMenu = {
                addItem: (cb: (item: any) => void) => {
                    const fakeItem = {
                        setTitle: () => fakeItem,
                        setIcon: () => fakeItem,
                        onClick: (fn: () => void) => { menuItemCallback = fn; return fakeItem; },
                    };
                    cb(fakeItem);
                },
            };
            const fakeFile = { path: "notes/test.md", name: "test.md" };
            callback(fakeMenu, fakeFile);

            expect(menuItemCallback).not.toBeNull();
            menuItemCallback!();
            expect(addSpy).toHaveBeenCalledWith(fakeFile);
        });

        it("addToLilbee calls api.addFiles with absolute path", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* noEvents() {}
            plugin.api.addFiles = vi.fn().mockReturnValue(noEvents());

            await (plugin as any).addToLilbee({ path: "notes/test.md", name: "test.md" });

            expect(plugin.api.addFiles).toHaveBeenCalledWith(["/test/vault/notes/test.md"]);
        });

        it("addToLilbee shows summary Notice on done event", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* withDone() {
                yield {
                    event: SSE_EVENT.DONE,
                    data: { added: ["test.md"], updated: [], removed: [], failed: [], unchanged: 0 },
                };
            }
            plugin.api.addFiles = vi.fn().mockReturnValue(withDone());

            await (plugin as any).addToLilbee({ path: "test.md", name: "test.md" });

            expect(Notice.instances.some((n) => n.message.includes("1 added"))).toBe(true);
        });

        it("addToLilbee shows error Notice on API failure", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.addFiles = vi.fn().mockImplementation(() => {
                throw new Error("connection refused");
            });

            await (plugin as any).addToLilbee({ path: "test.md", name: "test.md" });

            expect(Notice.instances.some((n) => n.message.includes("add failed"))).toBe(true);
        });

        it("addToLilbee returns early when statusBarEl is null", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            const addFilesSpy = vi.spyOn(plugin.api, "addFiles");
            await (plugin as any).addToLilbee({ path: "test.md", name: "test.md" });

            expect(addFilesSpy).not.toHaveBeenCalled();
        });

        it("addToLilbee does not show Notice when done has empty arrays", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* emptyDone() {
                yield {
                    event: SSE_EVENT.DONE,
                    data: { added: [], updated: [], removed: [], failed: [], unchanged: 1 },
                };
            }
            plugin.api.addFiles = vi.fn().mockReturnValue(emptyDone());

            await (plugin as any).addToLilbee({ path: "test.md", name: "test.md" });

            expect(Notice.instances.length).toBe(0);
        });

        it("addToLilbee shows failed count in Notice", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            async function* withFailed() {
                yield {
                    event: SSE_EVENT.DONE,
                    data: { added: [], updated: [], removed: [], failed: ["bad.pdf"], unchanged: 0 },
                };
            }
            plugin.api.addFiles = vi.fn().mockReturnValue(withFailed());

            await (plugin as any).addToLilbee({ path: "bad.pdf", name: "bad.pdf" });

            expect(Notice.instances.some((n) => n.message.includes("1 failed"))).toBe(true);
        });
    });

    describe("granular progress events", () => {
        it("file_start updates status bar with file-level progress", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const statusTexts: string[] = [];
            const origSetText = (plugin as any).statusBarEl!.setText.bind((plugin as any).statusBarEl);
            (plugin as any).statusBarEl!.setText = (text: string) => {
                statusTexts.push(text);
                origSetText(text);
            };

            async function* withFileStart() {
                yield { event: SSE_EVENT.FILE_START, data: { file: "paper.pdf", current_file: 3, total_files: 10 } };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withFileStart());

            await plugin.triggerSync();

            expect(statusTexts.some((t) => t.includes("3/10") && t.includes("paper.pdf"))).toBe(true);
        });

        it("extract updates status bar with page-level progress", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const statusTexts: string[] = [];
            const origSetText = (plugin as any).statusBarEl!.setText.bind((plugin as any).statusBarEl);
            (plugin as any).statusBarEl!.setText = (text: string) => {
                statusTexts.push(text);
                origSetText(text);
            };

            async function* withExtract() {
                yield { event: SSE_EVENT.EXTRACT, data: { file: "paper.pdf", page: 5, total_pages: 50 } };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withExtract());

            await plugin.triggerSync();

            expect(statusTexts.some((t) => t.includes("extracting") && t.includes("page 5/50"))).toBe(true);
        });

        it("embed updates status bar with chunk-level progress", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            const statusTexts: string[] = [];
            const origSetText = (plugin as any).statusBarEl!.setText.bind((plugin as any).statusBarEl);
            (plugin as any).statusBarEl!.setText = (text: string) => {
                statusTexts.push(text);
                origSetText(text);
            };

            async function* withEmbed() {
                yield { event: SSE_EVENT.EMBED, data: { file: "paper.pdf", chunk: 30, total_chunks: 100 } };
            }
            plugin.api.syncStream = vi.fn().mockReturnValue(withEmbed());

            await plugin.triggerSync();

            expect(statusTexts.some((t) => t.includes("embedding") && t.includes("30/100"))).toBe(true);
        });

        it("handleProgressEvent no-ops when statusBarEl is null", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            expect(() => {
                (plugin as any).handleProgressEvent({ event: SSE_EVENT.FILE_START, data: { file: "x", current_file: 1, total_files: 1 } });
            }).not.toThrow();
        });
    });

    describe("active model in status bar", () => {
        it("fetchActiveModel sets activeModel and updates status bar", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.listModels = vi.fn().mockResolvedValue({
                chat: { active: "qwen3:8b", installed: ["qwen3:8b"], catalog: [] },
                vision: { active: "", installed: [], catalog: [] },
            });

            plugin.fetchActiveModel();
            // Wait for the promise to resolve
            await new Promise((r) => setTimeout(r, 0));

            expect(plugin.activeModel).toBe("qwen3:8b");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready (qwen3:8b)");
        });

        it("fetchActiveModel silently fails on API error", async () => {
            const plugin = await createPlugin();
            await plugin.onload();

            plugin.api.listModels = vi.fn().mockRejectedValue(new Error("offline"));

            plugin.fetchActiveModel();
            await new Promise((r) => setTimeout(r, 0));

            expect(plugin.activeModel).toBe("");
            expect((plugin as any).statusBarEl?.textContent).toBe("lilbee: ready");
        });

        it("status bar includes model name during sync", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            plugin.activeModel = "llama3";

            const statusTexts: string[] = [];
            const origSetText = (plugin as any).statusBarEl!.setText.bind((plugin as any).statusBarEl);
            (plugin as any).statusBarEl!.setText = (text: string) => {
                statusTexts.push(text);
                origSetText(text);
            };

            async function* noEvents() {}
            plugin.api.syncStream = vi.fn().mockReturnValue(noEvents());

            await plugin.triggerSync();

            // "syncing..." text should include model name
            expect(statusTexts.some((t) => t.includes("syncing") && t.includes("llama3"))).toBe(true);
        });

        it("updateStatusBar no-ops when statusBarEl is null", async () => {
            const plugin = await createPlugin();
            await plugin.onload();
            (plugin as any).statusBarEl = null;

            expect(() => {
                (plugin as any).updateStatusBar("test");
            }).not.toThrow();
        });
    });
});
