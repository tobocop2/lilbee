import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { Notice } from "obsidian";
import { App, WorkspaceLeaf } from "./__mocks__/obsidian";

vi.mock("../src/api", () => ({
    LilbeeClient: vi.fn().mockImplementation(() => ({
        status: vi.fn(),
        syncStream: vi.fn(),
        search: vi.fn(),
        ask: vi.fn(),
        chatStream: vi.fn(),
        listModels: vi.fn(),
        pullModel: vi.fn(),
        setChatModel: vi.fn(),
        setVisionModel: vi.fn(),
    })),
}));

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

        it("with manual sync mode: does not register vault events", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "manual" });
            await plugin.onload();
            expect(plugin.registerEvent).not.toHaveBeenCalled();
        });

        it("with auto sync mode: registers vault events for create/modify/delete/rename", async () => {
            const plugin = await createPlugin();
            plugin.loadData = vi.fn().mockResolvedValue({ syncMode: "auto" });
            await plugin.onload();
            expect(plugin.registerEvent).toHaveBeenCalledTimes(4);
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

            // Confirm no vault events registered initially
            expect(plugin.registerEvent).not.toHaveBeenCalled();

            // Switch to auto and save
            plugin.settings.syncMode = "auto";
            await plugin.saveSettings();

            expect(plugin.registerEvent).toHaveBeenCalledTimes(4);
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

            // Save again with auto still active — should not register again
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
                yield { event: "progress", data: { file: "notes.md", current: 1, total: 5 } };
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
                    event: "done",
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
                    event: "done",
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
                yield { event: "progress", data: { file: "x.md", current: 1, total: 1 } };
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

            // Each registerEvent call comes from app.vault.on; we need to trigger them.
            // registerEvent is called with the result of vault.on; the vault.on mock returns
            // { id: "mock-vault-event" }. We need to capture the callbacks passed to vault.on.
            const vaultOnCalls = (plugin.app.vault.on as ReturnType<typeof vi.fn>).mock.calls as Array<[string, () => void]>;
            expect(vaultOnCalls.length).toBe(4);

            // Trigger the callback from the first vault.on call
            vaultOnCalls[0][1]();
            expect(debouncedSpy).toHaveBeenCalledTimes(1);
        });
    });
});
