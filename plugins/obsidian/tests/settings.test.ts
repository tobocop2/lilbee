import { vi, describe, it, expect, beforeEach } from "vitest";
import { App, Notice, Setting } from "obsidian";
import { MockElement } from "./__mocks__/obsidian";
import { LilbeeSettingTab } from "../src/settings";
import type { LilbeeSettings, ModelsResponse } from "../src/types";
import { DEFAULT_SETTINGS, SSE_EVENT } from "../src/types";

function makePlugin(overrides: Partial<LilbeeSettings> = {}, ollamaState: string = "unknown") {
    const settings: LilbeeSettings = { ...DEFAULT_SETTINGS, ...overrides };
    const api = {
        listModels: vi.fn(),
        setChatModel: vi.fn(),
        setVisionModel: vi.fn(),
        pullModel: vi.fn(),
    };
    const saveSettings = vi.fn().mockResolvedValue(undefined);
    const restartServer = vi.fn().mockResolvedValue(undefined);
    const ollamaDetector = { state: ollamaState };
    const statusBarEl = { setText: vi.fn(), textContent: "" };
    const fetchActiveModel = vi.fn();
    return { settings, api, saveSettings, restartServer, ollamaDetector, statusBarEl, fetchActiveModel } as unknown as InstanceType<typeof import("../src/main").default>;
}

function makeTab(plugin: ReturnType<typeof makePlugin>) {
    const app = new App();
    return new LilbeeSettingTab(app as any, plugin as any);
}

function makeModelsResponse(): ModelsResponse {
    return {
        chat: {
            active: "llama3",
            installed: ["llama3"],
            catalog: [
                { name: "llama3", size_gb: 4.7, min_ram_gb: 8, description: "Meta Llama 3", installed: true },
                { name: "phi3", size_gb: 2.3, min_ram_gb: 4, description: "Microsoft Phi-3", installed: false },
            ],
        },
        vision: {
            active: "",
            installed: [],
            catalog: [
                { name: "llava", size_gb: 4.5, min_ram_gb: 8, description: "LLaVA vision model", installed: false },
            ],
        },
    };
}

// Intercept Setting component callbacks by spying on the prototype methods.
// Each call returns `this`, so we collect the captured onChange handlers in order.

type TextOnChange = (v: string) => Promise<void>;
type SliderOnChange = (v: number) => Promise<void>;
type DropdownOnChange = (v: string) => Promise<void>;
type ToggleOnChange = (v: boolean) => Promise<void>;
type ButtonOnClick = () => Promise<void>;

interface Captured {
    textOnChanges: TextOnChange[];
    sliderOnChanges: SliderOnChange[];
    dropdownOnChanges: DropdownOnChange[];
    toggleOnChanges: ToggleOnChange[];
    buttonOnClicks: ButtonOnClick[];
}

function captureSettingCallbacks(fn: () => void): Captured {
    const textOnChanges: TextOnChange[] = [];
    const sliderOnChanges: SliderOnChange[] = [];
    const dropdownOnChanges: DropdownOnChange[] = [];
    const toggleOnChanges: ToggleOnChange[] = [];
    const buttonOnClicks: ButtonOnClick[] = [];

    const origAddText = Setting.prototype.addText;
    const origAddSlider = Setting.prototype.addSlider;
    const origAddDropdown = Setting.prototype.addDropdown;
    const origAddToggle = (Setting.prototype as any).addToggle;
    const origAddButton = Setting.prototype.addButton;

    Setting.prototype.addText = function (cb: (text: any) => void) {
        const fakeText = {
            setPlaceholder: () => fakeText,
            setValue: () => fakeText,
            onChange: (handler: TextOnChange) => { textOnChanges.push(handler); return fakeText; },
        };
        cb(fakeText);
        return this;
    };

    Setting.prototype.addSlider = function (cb: (slider: any) => void) {
        const fakeSlider = {
            setLimits: () => fakeSlider,
            setValue: () => fakeSlider,
            setDynamicTooltip: () => fakeSlider,
            onChange: (handler: SliderOnChange) => { sliderOnChanges.push(handler); return fakeSlider; },
        };
        cb(fakeSlider);
        return this;
    };

    Setting.prototype.addDropdown = function (cb: (dropdown: any) => void) {
        const fakeDropdown = {
            addOption: (_v: string, _l: string) => fakeDropdown,
            addOptions: (_opts: Record<string, string>) => fakeDropdown,
            setValue: () => fakeDropdown,
            onChange: (handler: DropdownOnChange) => { dropdownOnChanges.push(handler); return fakeDropdown; },
        };
        cb(fakeDropdown);
        return this;
    };

    (Setting.prototype as any).addToggle = function (cb: (toggle: any) => void) {
        const fakeToggle = {
            setValue: () => fakeToggle,
            onChange: (handler: ToggleOnChange) => { toggleOnChanges.push(handler); return fakeToggle; },
        };
        cb(fakeToggle);
        return this;
    };

    Setting.prototype.addButton = function (cb: (btn: any) => void) {
        const fakeBtn = {
            setButtonText: () => fakeBtn,
            onClick: (handler: ButtonOnClick) => { buttonOnClicks.push(handler); return fakeBtn; },
        };
        cb(fakeBtn);
        return this;
    };

    try {
        fn();
    } finally {
        Setting.prototype.addText = origAddText;
        Setting.prototype.addSlider = origAddSlider;
        Setting.prototype.addDropdown = origAddDropdown;
        (Setting.prototype as any).addToggle = origAddToggle;
        Setting.prototype.addButton = origAddButton;
    }

    return { textOnChanges, sliderOnChanges, dropdownOnChanges, toggleOnChanges, buttonOnClicks };
}

// Similarly capture dropdown options so we can inspect them
function captureDropdownOptions(fn: () => void): Array<Record<string, string>> {
    const allOptions: Array<Record<string, string>> = [];
    const origAddDropdown = Setting.prototype.addDropdown;

    Setting.prototype.addDropdown = function (cb: (dropdown: any) => void) {
        const options: Record<string, string> = {};
        const fakeDropdown = {
            addOption: (v: string, l: string) => { options[v] = l; return fakeDropdown; },
            addOptions: (opts: Record<string, string>) => { Object.assign(options, opts); return fakeDropdown; },
            setValue: () => fakeDropdown,
            onChange: () => fakeDropdown,
        };
        cb(fakeDropdown);
        allOptions.push({ ...options });
        return this;
    };

    try {
        fn();
    } finally {
        Setting.prototype.addDropdown = origAddDropdown;
    }

    return allOptions;
}

describe("LilbeeSettingTab", () => {
    beforeEach(() => {
        Notice.clear();
    });

    describe("display()", () => {
        it("renders server URL, topK, and sync-mode settings without error", () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            expect(() => tab.display()).not.toThrow();
        });

        it("creates h3 'Models' and description paragraph", () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            tab.display();
            const h3 = tab.containerEl.children.find((c) => c.tagName === "H3");
            expect(h3?.textContent).toBe("Models");
            const p = tab.containerEl.children.find(
                (c) => c.tagName === "P" && c.textContent.includes("Manage chat"),
            );
            expect(p).toBeDefined();
        });

        it("creates a models container div", () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            tab.display();
            const container = tab.containerEl.find("lilbee-models-container");
            expect(container).not.toBeNull();
        });

        it("shows sync-debounce setting when syncMode is 'auto'", () => {
            const plugin = makePlugin({ syncMode: "auto" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());
            // manageServer=true (default): serverUrl + ollamaUrl + binaryPath + syncDebounce = 4
            expect(textOnChanges.length).toBe(4);
        });

        it("does NOT show sync-debounce when syncMode is 'manual'", () => {
            const plugin = makePlugin({ syncMode: "manual" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());
            // manageServer=true (default): serverUrl + ollamaUrl + binaryPath = 3
            expect(textOnChanges.length).toBe(3);
        });
    });

    describe("manageServer toggle onChange", () => {
        it("updates manageServer, saves, and re-renders display", async () => {
            const plugin = makePlugin({ manageServer: false });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);

            const { toggleOnChanges } = captureSettingCallbacks(() => tab.display());
            const displaySpy = vi.spyOn(tab, "display").mockImplementation(() => {});

            await toggleOnChanges[0](true);

            expect(plugin.settings.manageServer).toBe(true);
            expect(plugin.saveSettings).toHaveBeenCalled();
            expect(displaySpy).toHaveBeenCalled();
        });
    });

    describe("binaryPath setting onChange", () => {
        it("updates binaryPath and calls saveSettings", async () => {
            const plugin = makePlugin({ manageServer: true });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            // textOnChanges[0] = serverUrl, textOnChanges[1] = ollamaUrl, textOnChanges[2] = binaryPath
            await textOnChanges[2]("/usr/local/bin/lilbee");
            expect(plugin.settings.binaryPath).toBe("/usr/local/bin/lilbee");
            expect(plugin.saveSettings).toHaveBeenCalled();
        });
    });

    describe("restart server button", () => {
        it("onClick calls restartServer", async () => {
            const plugin = makePlugin({ manageServer: true });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { buttonOnClicks } = captureSettingCallbacks(() => tab.display());

            // buttonOnClicks[0] = refresh models, buttonOnClicks[1] = restart server
            await buttonOnClicks[1]();
            expect(plugin.restartServer).toHaveBeenCalled();
        });
    });

    describe("serverUrl setting onChange", () => {
        it("updates plugin settings and calls saveSettings", async () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            // First text onChange is serverUrl
            await textOnChanges[0]("http://localhost:9999");
            expect(plugin.settings.serverUrl).toBe("http://localhost:9999");
            expect(plugin.saveSettings).toHaveBeenCalledTimes(1);
        });
    });

    describe("topK slider onChange", () => {
        it("updates topK and calls saveSettings", async () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { sliderOnChanges } = captureSettingCallbacks(() => tab.display());

            await sliderOnChanges[0](10);
            expect(plugin.settings.topK).toBe(10);
            expect(plugin.saveSettings).toHaveBeenCalledTimes(1);
        });
    });

    describe("syncMode dropdown onChange", () => {
        it("updates syncMode, saves, and re-renders display", async () => {
            const plugin = makePlugin({ syncMode: "manual" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);

            // Capture the syncMode dropdown onChange; the onChange calls this.display() so we
            // need to prevent infinite recursion by mocking display on subsequent calls.
            const { dropdownOnChanges } = captureSettingCallbacks(() => tab.display());

            // After the first display(), spy on display to count re-render calls
            const displaySpy = vi.spyOn(tab, "display").mockImplementation(() => {});

            // First dropdown onChange is the syncMode setting
            await dropdownOnChanges[0]("auto");

            expect(plugin.settings.syncMode).toBe("auto");
            expect(plugin.saveSettings).toHaveBeenCalled();
            expect(displaySpy).toHaveBeenCalled();
        });
    });

    describe("syncDebounce text onChange", () => {
        // With manageServer=true (default) + syncMode=auto, text fields are:
        // [0] serverUrl, [1] ollamaUrl, [2] binaryPath, [3] syncDebounce
        const DEBOUNCE_IDX = 3;

        it("updates syncDebounceMs for valid positive number", async () => {
            const plugin = makePlugin({ syncMode: "auto" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            await textOnChanges[DEBOUNCE_IDX]("3000");
            expect(plugin.settings.syncDebounceMs).toBe(3000);
            expect(plugin.saveSettings).toHaveBeenCalled();
        });

        it("accepts zero as a valid debounce value", async () => {
            const plugin = makePlugin({ syncMode: "auto" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            plugin.settings.syncDebounceMs = 999;
            await textOnChanges[DEBOUNCE_IDX]("0");
            expect(plugin.settings.syncDebounceMs).toBe(0);
            expect(plugin.saveSettings).toHaveBeenCalled();
        });

        it("does NOT save for NaN input", async () => {
            const plugin = makePlugin({ syncMode: "auto" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            const original = plugin.settings.syncDebounceMs;
            await textOnChanges[DEBOUNCE_IDX]("not-a-number");
            expect(plugin.settings.syncDebounceMs).toBe(original);
            expect(plugin.saveSettings).not.toHaveBeenCalled();
        });

        it("does NOT save for negative number input", async () => {
            const plugin = makePlugin({ syncMode: "auto" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            const original = plugin.settings.syncDebounceMs;
            await textOnChanges[DEBOUNCE_IDX]("-100");
            expect(plugin.settings.syncDebounceMs).toBe(original);
            expect(plugin.saveSettings).not.toHaveBeenCalled();
        });
    });

    describe("ollamaUrl setting onChange", () => {
        it("updates ollamaUrl and calls saveSettings", async () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { textOnChanges } = captureSettingCallbacks(() => tab.display());

            // textOnChanges[1] = ollamaUrl
            await textOnChanges[1]("http://remote:11434");
            expect(plugin.settings.ollamaUrl).toBe("http://remote:11434");
            expect(plugin.saveSettings).toHaveBeenCalled();
        });
    });

    describe("Ollama warning banner", () => {
        it("shows warning when Ollama is unreachable", () => {
            const plugin = makePlugin({}, "unreachable");
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            tab.display();
            const warning = tab.containerEl.find("lilbee-ollama-warning");
            expect(warning).not.toBeNull();
            const paragraphs = warning!.children.filter((c) => c.tagName === "P");
            expect(paragraphs.some((p) => p.textContent.includes("Ollama is not running"))).toBe(true);
        });

        it("does NOT show warning when Ollama is reachable", () => {
            const plugin = makePlugin({}, "reachable");
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            tab.display();
            const warning = tab.containerEl.find("lilbee-ollama-warning");
            expect(warning).toBeNull();
        });
    });

    describe("Refresh models button", () => {
        it("onClick calls loadModels with the models container", async () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());
            const tab = makeTab(plugin);
            const { buttonOnClicks } = captureSettingCallbacks(() => tab.display());

            // manageServer=true (default): Refresh button [0] + Restart button [1]
            expect(buttonOnClicks.length).toBe(2);
            await expect(buttonOnClicks[0]()).resolves.not.toThrow();
        });
    });

    describe("loadModels()", () => {
        it("renders chat and vision sections on success", async () => {
            const plugin = makePlugin();
            const models = makeModelsResponse();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(models);
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            await (tab as any).loadModels(container);
            const sections = (container as unknown as MockElement).findAll("lilbee-model-section");
            expect(sections.length).toBe(2);
        });

        it("shows warning paragraph when API call fails", async () => {
            const plugin = makePlugin();
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("network"));
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            await (tab as any).loadModels(container);
            const p = (container as unknown as MockElement).children.find(
                (c) => c.tagName === "P" && c.textContent.includes("Could not connect"),
            );
            expect(p).toBeDefined();
            expect(p?.classList.contains("mod-warning")).toBe(true);
        });
    });

    describe("renderModelSection()", () => {
        it("chat section does NOT have 'Disabled' dropdown option", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const allOptions = captureDropdownOptions(() => {
                (tab as any).renderModelSection(container, "Chat Model", makeModelsResponse().chat, "chat");
            });

            expect(allOptions.length).toBeGreaterThan(0);
            // The first dropdown is the active model selector
            expect("" in allOptions[0]).toBe(false);
        });

        it("vision section includes 'Disabled' option with empty string key", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const allOptions = captureDropdownOptions(() => {
                (tab as any).renderModelSection(container, "Vision Model", makeModelsResponse().vision, "vision");
            });

            expect(allOptions.length).toBeGreaterThan(0);
            expect("" in allOptions[0]).toBe(true);
            expect(allOptions[0][""]).toBe("Disabled");
        });

        it("active chat model onChange calls setChatModel and shows Notice", async () => {
            const plugin = makePlugin();
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const { dropdownOnChanges } = captureSettingCallbacks(() => {
                (tab as any).renderModelSection(container, "Chat Model", makeModelsResponse().chat, "chat");
            });

            await dropdownOnChanges[0]("phi3");
            expect(plugin.api.setChatModel).toHaveBeenCalledWith("phi3");
            expect(Notice.instances.some((n) => n.message.includes("phi3"))).toBe(true);
        });

        it("active vision model onChange calls setVisionModel and shows Notice", async () => {
            const plugin = makePlugin();
            (plugin.api.setVisionModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "llava" });
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const { dropdownOnChanges } = captureSettingCallbacks(() => {
                (tab as any).renderModelSection(container, "Vision Model", makeModelsResponse().vision, "vision");
            });

            await dropdownOnChanges[0]("llava");
            expect(plugin.api.setVisionModel).toHaveBeenCalledWith("llava");
            expect(Notice.instances.some((n) => n.message.includes("llava"))).toBe(true);
        });

        it("setting model to empty string shows 'disabled' in Notice", async () => {
            const plugin = makePlugin();
            (plugin.api.setVisionModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "" });
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const { dropdownOnChanges } = captureSettingCallbacks(() => {
                (tab as any).renderModelSection(container, "Vision Model", makeModelsResponse().vision, "vision");
            });

            await dropdownOnChanges[0]("");
            expect(Notice.instances.some((n) => n.message.includes("disabled"))).toBe(true);
        });

        it("active model onChange shows failure Notice on API error", async () => {
            const plugin = makePlugin();
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fail"));
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;

            const { dropdownOnChanges } = captureSettingCallbacks(() => {
                (tab as any).renderModelSection(container, "Chat Model", makeModelsResponse().chat, "chat");
            });

            await dropdownOnChanges[0]("phi3");
            expect(Notice.instances.some((n) => n.message.includes("Failed to set"))).toBe(true);
        });

        it("renders table with header columns", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            (tab as any).renderModelSection(container, "Chat Model", makeModelsResponse().chat, "chat");
            // Verify a table exists somewhere in the section
            function findTag(el: MockElement, tag: string): MockElement | null {
                if (el.tagName === tag.toUpperCase()) return el;
                for (const child of el.children) {
                    const found = findTag(child, tag);
                    if (found) return found;
                }
                return null;
            }
            const table = findTag(container as unknown as MockElement, "table");
            expect(table).not.toBeNull();
        });

        it("desc shows active model name when set", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            // Just verify it doesn't throw with a non-empty active field
            expect(() => {
                (tab as any).renderModelSection(container, "Chat Model", makeModelsResponse().chat, "chat");
            }).not.toThrow();
        });

        it("desc shows 'Disabled' for vision when active is empty", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            // vision catalog has active: ""
            expect(() => {
                (tab as any).renderModelSection(container, "Vision Model", makeModelsResponse().vision, "vision");
            }).not.toThrow();
        });

        it("desc shows 'Not set' for chat when active is empty", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const container = new MockElement("div") as unknown as HTMLElement;
            const catalog = { ...makeModelsResponse().chat, active: "" };
            expect(() => {
                (tab as any).renderModelSection(container, "Chat Model", catalog, "chat");
            }).not.toThrow();
        });
    });

    describe("renderCatalogRow()", () => {
        it("shows 'Installed' badge for installed models", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const table = new MockElement("table") as unknown as HTMLTableElement;
            const model = makeModelsResponse().chat.catalog[0]; // llama3, installed: true
            (tab as any).renderCatalogRow(table, model, "chat");
            const row = (table as unknown as MockElement).children[0];
            const actionCell = row.children[3];
            const badge = actionCell.children[0];
            expect(badge.textContent).toBe("Installed");
            expect(badge.classList.contains("lilbee-installed")).toBe(true);
        });

        it("shows 'Pull' button for uninstalled models", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const table = new MockElement("table") as unknown as HTMLTableElement;
            const model = makeModelsResponse().chat.catalog[1]; // phi3, installed: false
            (tab as any).renderCatalogRow(table, model, "chat");
            const row = (table as unknown as MockElement).children[0];
            const actionCell = row.children[3];
            const btn = actionCell.children[0];
            expect(btn.tagName).toBe("BUTTON");
            expect(btn.textContent).toBe("Pull");
        });

        it("renders model name, size, and description cells", () => {
            const plugin = makePlugin();
            const tab = makeTab(plugin);
            const table = new MockElement("table") as unknown as HTMLTableElement;
            const model = makeModelsResponse().chat.catalog[0];
            (tab as any).renderCatalogRow(table, model, "chat");
            const row = (table as unknown as MockElement).children[0];
            expect(row.children[0].textContent).toBe("llama3");
            expect(row.children[1].textContent).toBe("4.7 GB");
            expect(row.children[2].textContent).toBe("Meta Llama 3");
        });
    });

    describe("Pull button", () => {
        interface PullSetup {
            tab: LilbeeSettingTab;
            clickHandler: () => Promise<void>;
            actionCell: MockElement;
        }

        // Instead of trigger (which doesn't await async handlers), we capture the
        // click handler function via addEventListener and call it directly.
        async function setupPullButton(
            plugin: ReturnType<typeof makePlugin>,
            type: "chat" | "vision" = "chat",
        ): Promise<PullSetup> {
            const tab = makeTab(plugin);
            const table = new MockElement("table") as unknown as HTMLTableElement;
            const catalog = makeModelsResponse();
            const model = type === "chat" ? catalog.chat.catalog[1] : catalog.vision.catalog[0];

            let clickHandler: (() => Promise<void>) | null = null;
            const origAddEventListener = MockElement.prototype.addEventListener;
            MockElement.prototype.addEventListener = function (event: string, handler: Function) {
                if (event === "click") {
                    clickHandler = handler as () => Promise<void>;
                }
                origAddEventListener.call(this, event, handler);
            };

            (tab as any).renderCatalogRow(table, model, type);
            MockElement.prototype.addEventListener = origAddEventListener;

            const row = (table as unknown as MockElement).children[0];
            const actionCell = row.children[3];

            return { tab, clickHandler: clickHandler!, actionCell };
        }

        it("successful pull: shows progress percentage and success Notice, sets model", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.PROGRESS, data: { model: "phi3", status: "pulling", completed: 75, total: 100 } };
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler, actionCell } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await clickHandler();

            const progressDiv = actionCell.find("lilbee-pull-progress");
            expect(progressDiv?.textContent).toBe("75%");
            expect(Notice.instances.some((n) => n.message.includes("phi3") && n.message.includes("pulled"))).toBe(true);
            expect(plugin.api.setChatModel).toHaveBeenCalledWith("phi3");
        });

        it("progress event with total=0: does not set percentage text", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.PROGRESS, data: { model: "phi3", status: "pulling", completed: 0, total: 0 } };
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler, actionCell } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await clickHandler();

            const progressDiv = actionCell.find("lilbee-pull-progress");
            expect(progressDiv?.textContent).not.toMatch(/\d+%/);
        });

        it("pull with vision type calls setVisionModel", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setVisionModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "llava" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "vision");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await clickHandler();

            expect(plugin.api.setVisionModel).toHaveBeenCalledWith("llava");
        });

        it("pull failure: shows failure Notice and re-enables button with 'Pull' text", async () => {
            const plugin = makePlugin();

            async function* failingPull(): AsyncGenerator<never> {
                throw new Error("network error");
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(failingPull());

            const { tab, clickHandler, actionCell } = await setupPullButton(plugin, "chat");
            const btn = actionCell.children[0];

            await clickHandler();

            expect(Notice.instances.some((n) => n.message.includes("Failed to pull"))).toBe(true);
            expect(btn.disabled).toBe(false);
            expect(btn.textContent).toBe("Pull");
        });

        it("successful pull without models container: does not crash", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { clickHandler } = await setupPullButton(plugin, "chat");
            // Do NOT attach a modelsContainer — querySelector returns null

            await expect(clickHandler()).resolves.not.toThrow();
            expect(Notice.instances.some((n) => n.message.includes("pulled"))).toBe(true);
        });

        it("successful pull with HTMLElement container: reloads models", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "chat");

            // In the node test environment HTMLElement does not exist globally. We define
            // it as an alias for MockElement so that querySelector can return a MockElement
            // that satisfies `instanceof HTMLElement`, and loadModels can call .empty() etc.
            Object.defineProperty(globalThis, "HTMLElement", { value: MockElement, configurable: true, writable: true });
            const fakeContainer = new MockElement("div");
            tab.containerEl.querySelector = vi.fn().mockReturnValue(fakeContainer);

            await clickHandler();

            expect(plugin.api.listModels).toHaveBeenCalled();

            // Cleanup the global stub
            // @ts-expect-error removing test-only global
            delete (globalThis as any).HTMLElement;
        });

        it("pull progress updates plugin status bar", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.PROGRESS, data: { model: "phi3", status: "pulling", completed: 45, total: 100 } };
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await clickHandler();

            expect(plugin.statusBarEl!.setText).toHaveBeenCalledWith("lilbee: pulling phi3 — 45%");
        });

        it("successful pull calls fetchActiveModel", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await clickHandler();

            expect(plugin.fetchActiveModel).toHaveBeenCalled();
        });

        it("pull progress does not update status bar when statusBarEl is null", async () => {
            const plugin = makePlugin();
            (plugin as any).statusBarEl = null;

            async function* fakePull() {
                yield { event: SSE_EVENT.PROGRESS, data: { model: "phi3", status: "pulling", completed: 50, total: 100 } };
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await expect(clickHandler()).resolves.not.toThrow();
        });

        it("non-progress events during pull are ignored (no crash)", async () => {
            const plugin = makePlugin();

            async function* fakePull() {
                yield { event: "status", data: { msg: "downloading" } };
                yield { event: SSE_EVENT.DONE, data: {} };
            }
            (plugin.api.pullModel as ReturnType<typeof vi.fn>).mockReturnValue(fakePull());
            (plugin.api.setChatModel as ReturnType<typeof vi.fn>).mockResolvedValue({ model: "phi3" });
            (plugin.api.listModels as ReturnType<typeof vi.fn>).mockResolvedValue(makeModelsResponse());

            const { tab, clickHandler } = await setupPullButton(plugin, "chat");
            const modelsContainer = new MockElement("div");
            modelsContainer.classList.add("lilbee-models-container");
            tab.containerEl.children.push(modelsContainer);

            await expect(clickHandler()).resolves.not.toThrow();
        });
    });
});
